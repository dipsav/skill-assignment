"""Skill-based queue simulation with an arrival-normalized surge experiment.

This module replaces the old global-state simulation and batch functions. It
has no dependency on the original Request, Server, Event, or statistics globals.
Requires Python >= 3.10 and NumPy. Matplotlib is needed only by the example.

Counts used to size phases are EXPECTED arrivals, not exact stopping counts.
The total includes warm-up, baseline, surge, and recovery. Recovery receives
whatever remains after the first three counts. Customers are never discarded
at a phase boundary; the observation stops at a fixed, derived time horizon.

The engine uses x = baseline_arrival_rate * physical_time internally. Output
trace coordinates use tau = x - x_at_surge_start. Each trace value is an exact
time-weighted BIN AVERAGE, not an instantaneous sample or moving average.

Policy: FIFO within a skill; lowest cost/rate score across skills; optionally
preempt a strictly lower-priority job (larger score). Scores are rounded to six
decimals, as in the original. Eligible idle servers are chosen by server ID;
preemption candidates by (-running-job score, incoming-skill score, server ID).
After every processed arrival or departure, waiting jobs are reconsidered
until no eligible direct strict-priority swap remains. No global matching
or augmenting-path migration is performed. Service work is
unit-exponential, served at the assigned server/skill rate. Preemption resumes
remaining work, possibly on another eligible server. Arrival, skill, and work
random streams are separated. Waiting includes every interruption.

Surge/recovery reporting blocks default to 10,000 expected arrivals each.
Recovery is first confirmed from fine-bin queue means, separately per run;
non-detections are marked censored and never treated as zero recovery time.

All function results are DICTIONARIES, not the old four-element tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import multiprocessing as mp
import operator
import os
from typing import Any, Sequence

import numpy as np

PHASES = ("warmup", "baseline", "surge", "recovery")


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _divide(numerator: Any, denominator: Any, fill: float = np.nan) -> np.ndarray:
    a, b = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    return np.divide(a, b, out=np.full(a.shape, fill), where=(b != 0))


def make_surge_schedule(
    baseRate: float,
    expectedTotalNumberRequests: float = 100_000,
    numberWarmingUpRequests: float = 10_000,
    numberBaselineRequests: float = 20_000,
    numberSurgeRequests: float = 10_000,
    surgeMultiplier: float = 3.0,
    sampleEveryArrivals: float = 100.0,
    *,
    queueReportEveryArrivals: float = 10_000.0,
) -> dict[str, Any]:
    """Create comparable phase boundaries and bins for any positive base rate.

    sampleEveryArrivals is the MAXIMUM bin width in baseline-arrival units.
    Each report block is divided into bins no larger than this value.
    queueReportEveryArrivals sizes reporting blocks using each phase's ACTUAL
    intensity (still expected counts). Phase/report boundaries are bin edges;
    no bin crosses a phase or report boundary.
    """
    rate = _positive_float(baseRate, "baseRate")
    total = _positive_float(expectedTotalNumberRequests, "expectedTotalNumberRequests")
    warm = _positive_float(numberWarmingUpRequests, "numberWarmingUpRequests", allow_zero=True)
    baseline = _positive_float(numberBaselineRequests, "numberBaselineRequests")
    surge = _positive_float(numberSurgeRequests, "numberSurgeRequests")
    multiplier = _positive_float(surgeMultiplier, "surgeMultiplier")
    width = _positive_float(sampleEveryArrivals, "sampleEveryArrivals")
    report_width = _positive_float(queueReportEveryArrivals, "queueReportEveryArrivals")
    recovery = total - warm - baseline - surge
    if recovery <= 0:
        raise ValueError("Total expected arrivals must exceed warmup + baseline + surge.")

    counts = np.array([warm, baseline, surge, recovery], dtype=float)
    multipliers = np.array([1.0, 1.0, multiplier, 1.0])
    durations = counts / multipliers
    phase_edges = np.r_[0.0, np.cumsum(durations)]
    if not np.all(np.isfinite(phase_edges)) or np.any(np.diff(phase_edges[1:]) <= 0):
        raise ValueError("The phase times are not finite or numerically distinguishable.")
    if not np.all(np.isfinite(phase_edges / rate)):
        raise ValueError("The physical-time horizon is not finite.")

    edges = [phase_edges[1]]
    bin_phases = []
    report_windows = []
    for phase in (1, 2, 3):
        # Reporting blocks count EXPECTED arrivals at the rate IN THIS PHASE.
        # Insert each block boundary before subdividing it into fine bins;
        # reports can then sum exact integrated areas without interpolation.
        blocks = 1 if phase == 1 else int(np.ceil(counts[phase] / report_width))
        for block in range(blocks):
            count_start = 0.0 if phase == 1 else block * report_width
            count_end = counts[phase] if phase == 1 else min(
                counts[phase], (block + 1) * report_width
            )
            start_x = phase_edges[phase] + count_start / multipliers[phase]
            end_x = (phase_edges[phase + 1] if block == blocks - 1 else
                     phase_edges[phase] + count_end / multipliers[phase])
            fine_bins = max(1, int(np.ceil(
                (count_end - count_start) / (multipliers[phase] * width)
            )))
            first_bin = len(bin_phases)
            edges.extend(np.linspace(start_x, end_x, fine_bins + 1)[1:])
            bin_phases.extend([phase] * fine_bins)
            if phase in (2, 3):
                report_windows.append({
                    "phase": PHASES[phase],
                    "block_index": block,
                    "expected_arrivals_start": float(count_start),
                    "expected_arrivals_end": float(count_end),
                    "expected_arrivals": float(count_end - count_start),
                    "start_time": float(start_x / rate),
                    "end_time": float(end_x / rate),
                    "duration": float((end_x - start_x) / rate),
                    "normalized_start": float(start_x - phase_edges[2]),
                    "normalized_end": float(end_x - phase_edges[2]),
                    "normalized_duration": float(end_x - start_x),
                    "bin_start": first_bin,
                    "bin_stop": len(bin_phases),
                })
    edges = np.asarray(edges, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges are not numerically distinguishable; increase bin width.")
    tau_edges = edges - phase_edges[2]
    return {
        "phase_names": PHASES,
        "sample_every_arrivals": width,
        "queue_report_every_arrivals": report_width,
        "queue_report_windows": report_windows,
        "base_rate": rate,
        "expected_arrivals": counts,
        "rate_multipliers": multipliers,
        "phase_edges_x": phase_edges,
        "phase_edges_time": phase_edges / rate,
        "bin_edges_x": edges,
        "bin_phases": np.asarray(bin_phases, dtype=int),
        "normalized_time_edges": tau_edges,
        "normalized_time": (tau_edges[:-1] + tau_edges[1:]) / 2,
        "surge_start_time": phase_edges[2] / rate,
        "surge_end_time": phase_edges[3] / rate,
        "stop_time": phase_edges[4] / rate,
        "normalized_surge_end": durations[2],
        "expected_total_arrivals": total,
    }


def _next_arrival_x(
    now: float, surge_start: float, surge_end: float,
    multiplier: float, rng: np.random.Generator,
) -> float:
    """Invert the integrated intensity across both rate-change boundaries."""
    remaining = float(rng.exponential())
    if multiplier == 1.0:
        return now + remaining
    t = now
    for boundary, rate in ((surge_start, 1.0), (surge_end, multiplier)):
        if t >= boundary:
            continue
        available = rate * (boundary - t)
        if remaining < available:
            return t + remaining / rate
        remaining -= available
        t = boundary
    return t + remaining  # baseline intensity is one in x time


def _validate_model(rate, cdf, rates, costs, buckets):
    rate = _positive_float(rate, "requestArrivalRate")
    rates = np.array(rates, dtype=float, copy=True)
    if rates.ndim != 2 or min(rates.shape) == 0:
        raise ValueError("skillServerRates must have shape (servers, skills).")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Service rates must be finite and nonnegative.")
    cdf = np.array(cdf, dtype=float, copy=True)
    if (cdf.shape != (rates.shape[1],) or not np.all(np.isfinite(cdf))
            or np.any(cdf < 0) or np.any(np.diff(cdf) < 0)
            or not np.isclose(cdf[-1], 1.0, rtol=0, atol=1e-12)):
        raise ValueError("skillDistribution must be a nondecreasing skill CDF ending at 1.")
    cdf[-1] = 1.0
    if np.any(np.diff(cdf) < 0):
        raise ValueError("skillDistribution contains values greater than 1.")
    probabilities = np.diff(np.r_[0.0, cdf])
    if np.any((probabilities > 0) & ~np.any(rates > 0, axis=0)):
        raise ValueError("Every arriving skill needs at least one eligible server.")
    costs = rates.copy() if costs is None else np.asarray(costs, dtype=float)
    if costs.shape != rates.shape or not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("skillServerCosts must be finite, nonnegative, and match service rates.")
    scores = np.divide(costs, rates, out=np.full_like(rates, np.inf), where=rates > 0)
    scores = np.round(scores, 6)
    buckets = np.asarray(buckets, dtype=float)
    if (buckets.ndim != 1 or np.any(np.isnan(buckets)) or np.any(buckets < 0)
            or np.any(buckets[1:] < buckets[:-1])):
        raise ValueError("Histogram thresholds must be a sorted nonnegative vector.")
    return rate, cdf, rates, costs.copy(), scores, buckets


@dataclass(slots=True)
class _Job:
    sequence: int
    skill: int
    arrival: float
    arrival_phase: int
    remaining_work: float
    entered_queue: float
    waiting: float = 0.0
    service: float = 0.0


class _CompletedJobs:
    """Completion-observed statistics classified by the job's ARRIVAL phase."""
    def __init__(self, n_skills, thresholds, enabled):
        self.enabled = enabled
        self.thresholds = thresholds
        self.count = np.zeros((4, n_skills), dtype=np.int64)
        self.mean = np.zeros((4, n_skills, 2))  # waiting, sojourn
        self.m2 = np.zeros_like(self.mean)
        self.cdf_count = np.zeros((4, n_skills, len(thresholds)), dtype=np.int64)

    def record(self, job, now, base_rate):
        p, k = job.arrival_phase, job.skill
        self.count[p, k] += 1
        if not self.enabled:
            return
        observation = np.array([job.waiting, now - job.arrival]) / base_rate
        delta = observation - self.mean[p, k]
        self.mean[p, k] += delta / self.count[p, k]
        self.m2[p, k] += delta * (observation - self.mean[p, k])
        self.cdf_count[p, k] += observation[0] <= self.thresholds

    def summary(self, phases):
        if not self.enabled:
            return None
        ix = list(phases)
        count = self.count[ix].sum(axis=0)
        mean = _divide(
            (self.count[ix, :, None] * self.mean[ix]).sum(axis=0),
            count[:, None], fill=0.0,
        )
        m2 = (self.m2[ix] + self.count[ix, :, None]
              * (self.mean[ix] - mean[None, :, :]) ** 2).sum(axis=0)
        variance = _divide(m2, np.where(count > 1, count - 1, 0)[:, None])
        mean[count == 0] = np.nan
        return {
            "completed_count_by_skill": count,
            "mean_waiting_time": mean[:, 0],
            "variance_waiting_time": variance[:, 0],
            "mean_sojourn_time": mean[:, 1],
            "variance_sojourn_time": variance[:, 1],
            "waiting_cdf_thresholds": self.thresholds.copy(),
            "waiting_cdf": _divide(self.cdf_count[ix].sum(axis=0), count[:, None]),
        }


class _TimeMonitor:
    """Integrate piecewise-constant state exactly, including boundary overlaps."""
    def __init__(self, schedule, n_servers, n_skills, max_queue):
        self.schedule = schedule
        self.edges = schedule["bin_edges_x"]
        self.bin_phase = schedule["bin_phases"]
        n_bins = len(self.edges) - 1
        self.bin_queue = np.zeros((n_bins, n_skills))
        self.bin_system = np.zeros_like(self.bin_queue)
        self.bin_arrivals = np.zeros((n_bins, n_skills), dtype=np.int64)
        self.bin_departures = np.zeros_like(self.bin_arrivals)
        self.phase_queue = np.zeros((4, n_skills))
        self.phase_system = np.zeros_like(self.phase_queue)
        self.phase_busy = np.zeros((4, n_servers, n_skills))
        self.phase_peak = np.zeros(4, dtype=np.int64)
        self.phase_duration = np.zeros(4)
        self.system_histogram = np.zeros((n_skills, max_queue + 1))
        self.max_queue = max_queue
        self.skill_ids = np.arange(n_skills)
        self.p = 0 if self.edges[0] > 0 else 1
        self.b = 0

    def advance(self, t0, t1, queue, in_system, serving_skills):
        if t1 < t0:
            raise RuntimeError("Simulation time moved backwards.")
        if t1 == t0:
            return
        queue_total = int(queue.sum())
        active = np.flatnonzero(serving_skills >= 0)
        active_skills = serving_skills[active]
        capped = np.minimum(in_system, self.max_queue)
        while t0 < t1:
            boundary = self.edges[0] if self.p == 0 else self.edges[self.b + 1]
            end = min(t1, boundary)
            duration = end - t0
            p, b = self.p, self.b
            self.phase_queue[p] += queue * duration
            self.phase_system[p] += in_system * duration
            self.phase_busy[p, active, active_skills] += duration
            self.phase_duration[p] += duration
            self.phase_peak[p] = max(self.phase_peak[p], queue_total)
            if p > 0:
                self.bin_queue[b] += queue * duration
                self.bin_system[b] += in_system * duration
                self.system_histogram[self.skill_ids, capped] += duration
            t0 = end
            if end == boundary:
                if p == 0:
                    self.p = 1
                else:
                    self.b += 1
                    self.p = int(self.bin_phase[self.b]) if self.b < len(self.bin_phase) else 4



def _validate_recovery_options(absolute, relative, hold):
    return (
        _positive_float(absolute, "recoveryAbsTolerance", allow_zero=True),
        _positive_float(relative, "recoveryRelTolerance", allow_zero=True),
        _positive_float(hold, "recoveryHoldArrivals"),
    )


def _queue_report_metrics(schedule, bin_queue_area, bin_arrivals, baseline):
    """Exact time-weighted reports; every report boundary is a fine-bin edge."""
    windows = schedule["queue_report_windows"]
    queue_means = np.stack([
        bin_queue_area[w["bin_start"]:w["bin_stop"]].sum(axis=0)
        / w["normalized_duration"]
        for w in windows
    ])
    actual = np.stack([
        bin_arrivals[w["bin_start"]:w["bin_stop"]].sum(axis=0)
        for w in windows
    ])
    return {
        "windows": windows,
        "mean_queue_total": queue_means.sum(axis=1),
        "mean_queue_by_skill": queue_means,
        "mean_queue_excess_total": queue_means.sum(axis=1) - baseline.sum(),
        "mean_queue_excess_by_skill": queue_means - baseline,
        "actual_arrivals_total": actual.sum(axis=1),
        "actual_arrivals_by_skill": actual,
    }


def measure_recovery(
    schedule: dict[str, Any],
    queue_total: Sequence[float],
    baseline_queue_total: float,
    *,
    recoveryAbsTolerance: float = 1.0,
    recoveryRelTolerance: float = 0.10,
    recoveryHoldArrivals: float = 1_000.0,
) -> dict[str, Any]:
    """Detect first CONFIRMED recovery after the surge, separately per run.

    Threshold = this run's baseline mean + max(absTolerance,
                                              relTolerance * baseline mean).
    All fine-bin queue MEANS in a consecutive post-surge run must be at or
    below that threshold for at least recoveryHoldArrivals baseline-arrival
    time units. The confirmation time is the END of the first bin completing
    that run; its start is also returned. Bin averages do not constrain every
    instantaneous queue state. Later excursions do not undo first detection.

    This is a user-configurable operational definition, not a stationarity
    test or a guarantee of permanent recovery. Missing recovery is NaN plus an
    explicit censored flag. A censored run is NOT assigned zero recovery time.
    """
    absolute, relative, hold = _validate_recovery_options(
        recoveryAbsTolerance, recoveryRelTolerance, recoveryHoldArrivals
    )
    baseline = _positive_float(baseline_queue_total, "baseline_queue_total", allow_zero=True)
    queue = np.asarray(queue_total, dtype=float)
    bin_edges = np.asarray(schedule["bin_edges_x"], dtype=float)
    if queue.shape != (len(bin_edges) - 1,) or not np.all(np.isfinite(queue)) or np.any(queue < 0):
        raise ValueError("queue_total must contain one finite nonnegative mean per bin.")
    threshold = baseline + max(absolute, relative * baseline)
    if not np.isfinite(threshold):
        raise ValueError("The recovery threshold is not finite.")
    recovery_bins = np.flatnonzero(schedule["bin_phases"] == 3)
    surge_end = float(schedule["phase_edges_x"][3])
    horizon = float(schedule["phase_edges_x"][4] - surge_end)
    rate = float(schedule["base_rate"])
    run_start = None
    detected = np.nan
    onset = np.nan
    for b in recovery_bins:
        if queue[b] <= threshold:
            if run_start is None:
                run_start = float(bin_edges[b])
            duration = float(bin_edges[b + 1] - run_start)
            if duration >= hold or math.isclose(duration, hold, rel_tol=1e-12, abs_tol=1e-10):
                onset = run_start - surge_end
                detected = float(bin_edges[b + 1] - surge_end)
                break
        else:
            run_start = None
    recovered = bool(np.isfinite(detected))
    observed = detected if recovered else horizon
    return {
        "recovered": recovered,
        "censored": not recovered,
        "baseline_queue_total": baseline,
        "threshold_queue": threshold,
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "hold_arrivals": hold,
        "recovery_time_arrivals": detected,
        "recovery_time": detected / rate,
        "qualifying_run_start_arrivals": onset,
        "qualifying_run_start_time": onset / rate,
        "followup_arrivals": horizon,
        "followup_time": horizon / rate,
        "observed_time_arrivals": observed,
        "observed_time": observed / rate,
        "max_bin_width_arrivals": float(np.diff(bin_edges)[recovery_bins].max()),
    }


def _summarize_recovery(records):
    """Keep censoring explicit; do not pass NaNs to ordinary batch means."""
    recovered = np.array([r["recovered"] for r in records], dtype=bool)
    arrival_times = np.array([r["recovery_time_arrivals"] for r in records])
    physical_times = np.array([r["recovery_time"] for r in records])
    reference = records[0]
    return {
        "replication_results": records,
        "recovered": recovered,
        "censored": ~recovered,
        "recovered_count": int(recovered.sum()),
        "censored_count": int((~recovered).sum()),
        "recovered_fraction": float(recovered.mean()),
        "recovery_time_arrivals": arrival_times,
        "recovery_time": physical_times,
        "threshold_queue": _replication_summary([r["threshold_queue"] for r in records]),
        "absolute_tolerance": reference["absolute_tolerance"],
        "relative_tolerance": reference["relative_tolerance"],
        "hold_arrivals": reference["hold_arrivals"],
        "max_bin_width_arrivals": reference["max_bin_width_arrivals"],
        "followup_arrivals": reference["followup_arrivals"],
        "followup_time": reference["followup_time"],
        "conditional_on_recovery": {
            "time_arrivals": _replication_summary(arrival_times[recovered]),
            "time": _replication_summary(physical_times[recovered]),
        },
        # Mean min(confirmation time, follow-up horizon), NOT an estimate of
        # the uncensored/unrestricted mean when runs are censored.
        "horizon_capped_time_arrivals": _replication_summary([
            r["observed_time_arrivals"] for r in records
        ]),
        "horizon_capped_time": _replication_summary([
            r["observed_time"] for r in records
        ]),
    }


def print_surge_report(result: dict[str, Any], *, label: str = "Scenario", file=None) -> None:
    """Print baseline, surge/recovery blocks and per-replication recovery stats.

    Pass a BATCH result. Queue SD/SEM describe between-replication variability.
    Block labels are expected-arrival counts within their respective phases;
    mean actual arrivals are printed separately. A final partial block, when
    present, is labeled with its actual expected exposure.
    """
    if "trajectories" not in result or "queue_reports" not in result or "recovery" not in result:
        raise ValueError("Pass an updated simulation_batch_run result.")
    schedule = result["schedule"]
    reports = result["queue_reports"]
    recovery = result["recovery"]
    baseline = result["baseline"]["queue_total"]
    total = schedule["expected_total_arrivals"]
    print(f"\n{label}: {result['replications']} replications x {total:,.0f} expected arrivals", file=file)
    print("Queue = people WAITING, excluding customers in service.", file=file)
    print("All block sizes are EXPECTED arrivals; actual counts fluctuate.", file=file)
    counts = schedule["expected_arrivals"]
    print("Expected phase counts: " + ", ".join(
        f"{name}={count:,.0f}" for name, count in zip(PHASES, counts)
    ), file=file)
    print(f"Surge multiplier: {schedule['rate_multipliers'][2]:g}", file=file)
    print(f"Baseline mean Q: {float(baseline['mean']):.4f} "
          f"(SD={float(baseline['sd']):.4f}, SEM={float(baseline['sem']):.4f})", file=file)
    print("", file=file)
    print(f"{'Period':<12} {'Expected arrivals in phase':>27} {'Mean Q':>11} {'SD':>10} "
          f"{'SEM':>10} {'Mean excess Q':>15} {'Actual arrivals*':>17}", file=file)
    print("-" * 110, file=file)
    for j, window in enumerate(reports["windows"]):
        period = "Surge" if window["phase"] == "surge" else "Post-surge"
        exposure = f"{window['expected_arrivals_start']:,.0f}-{window['expected_arrivals_end']:,.0f}"
        q = reports["mean_queue_total"]
        excess = reports["mean_queue_excess_total"]["mean"][j]
        actual = reports["actual_arrivals_total"]["mean"][j]
        print(f"{period:<12} {exposure:>27} {q['mean'][j]:11.4f} {q['sd'][j]:10.4f} "
              f"{q['sem'][j]:10.4f} {excess:15.4f} {actual:17.2f}", file=file)
    print("* Actual arrivals averaged across replications. SD/SEM are across replication means.", file=file)
    print("\nRecovery rule (evaluated separately in each replication):", file=file)
    print(f"  Fine-bin mean Q <= baseline + max({recovery['absolute_tolerance']:g}, "
          f"{recovery['relative_tolerance']:g} * baseline)", file=file)
    print(f"  for at least {recovery['hold_arrivals']:,.0f} consecutive baseline-arrival time units.", file=file)
    print(f"  Fine bins are at most {recovery['max_bin_width_arrivals']:g} baseline-arrival units wide.", file=file)
    print("  Recovery time is the END of the qualifying interval, measured from surge end.", file=file)
    print(f"Recovered by horizon: {recovery['recovered_count']}/{result['replications']} "
          f"({recovery['recovered_fraction']:.1%}); censored: {recovery['censored_count']}.", file=file)
    if recovery["recovered_count"]:
        arr = recovery["conditional_on_recovery"]["time_arrivals"]
        physical = recovery["conditional_on_recovery"]["time"]
        print(f"Mean confirmed recovery, recovered runs only: {float(arr['mean']):,.2f} "
              f"baseline-arrival units (SD={float(arr['sd']):,.2f}, SEM={float(arr['sem']):,.2f}).", file=file)
        print(f"Equivalent physical time: {float(physical['mean']):,.2f} "
              f"(SD={float(physical['sd']):,.2f}, SEM={float(physical['sem']):,.2f}).", file=file)
    else:
        print("No confirmed recoveries; the recovered-only mean is undefined (NaN).", file=file)
    if recovery["censored_count"]:
        capped = recovery["horizon_capped_time_arrivals"]["mean"]
        print(f"Follow-up horizon: {recovery['followup_arrivals']:,.0f} baseline-arrival units.", file=file)
        print(f"Horizon-capped mean confirmation time: {float(capped):,.2f}; "
              "this is NOT the uncensored mean recovery time.", file=file)
    print("This is a bin-mean threshold rule, not proof of permanent or stationary recovery.", file=file)


def simulation(
    requestArrivalRate=5.0,
    skillDistribution=None,
    skillServerRates=None,
    skillServerCosts=None,
    histogramBucketsInput=(0.0,),
    expectedTotalNumberRequests=100_000,
    numberWarmingUpRequests=10_000,
    seed=10,
    collectWaitingTimeStatisticsInput=True,
    *,
    numberBaselineRequests=20_000,
    numberSurgeRequests=10_000,
    surgeMultiplier=3.0,
    sampleEveryArrivals=100.0,
    queueReportEveryArrivals=10_000.0,
    recoveryAbsTolerance=1.0,
    recoveryRelTolerance=0.10,
    recoveryHoldArrivals=1_000.0,
    maxQueue=500,
    preemptive=True,
) -> dict[str, Any]:
    """Run one replication; return trajectories, phase metrics, and diagnostics.

    The first nine positional arguments match the original simulation API.
    New options are keyword-only. Defaults allocate expected arrival counts:
    10k warmup + 20k baseline + 10k surge + 60k recovery = 100k total.
    A multiplier of 1 is a no-surge control with the same phase arrival counts.
    queueReportEveryArrivals gives exact time-weighted surge/recovery reports.
    Recovery is measured from fine bins using a configurable baseline threshold
    and consecutive-window rule; see measure_recovery and print_surge_report.

    Queues exclude jobs in service. The number-in-system PMF and utilization
    cover the whole POST-WARMUP observation period, not a stationary regime.
    PMF columns 0..maxQueue-1 have a separate P(N >= maxQueue) overflow value;
    maxQueue NEVER caps the simulated queue. Waiting statistics describe only
    observed completions from the stated arrival cohorts; unfinished jobs are
    reported as censored, not assigned zero waits. Time statistics are in the
    physical time units of the provided rates.
    """
    rate, cdf, rates, costs, scores, buckets = _validate_model(
        requestArrivalRate, skillDistribution, skillServerRates,
        skillServerCosts, histogramBucketsInput,
    )
    maxQueue = _integer(maxQueue, "maxQueue", 1)
    if not isinstance(preemptive, (bool, np.bool_)):
        raise ValueError("preemptive must be Boolean.")
    schedule = make_surge_schedule(
        rate, expectedTotalNumberRequests, numberWarmingUpRequests,
        numberBaselineRequests, numberSurgeRequests, surgeMultiplier, sampleEveryArrivals,
        queueReportEveryArrivals=queueReportEveryArrivals,
    )
    _validate_recovery_options(recoveryAbsTolerance, recoveryRelTolerance, recoveryHoldArrivals)
    edges = schedule["phase_edges_x"]
    stop = float(edges[-1])
    multiplier = float(surgeMultiplier)
    normalized_rates = rates / rate
    if not np.all(np.isfinite(normalized_rates)) or np.any((rates > 0) & (normalized_rates == 0)):
        raise ValueError("Service/arrival rate ratios exceed floating-point range.")
    n_servers, n_skills = rates.shape
    # Copy a passed SeedSequence, so reusing it does not mutate its spawn state.
    if isinstance(seed, np.random.SeedSequence):
        root = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size)
    else:
        root = np.random.SeedSequence(seed)
    arrival_rng, skill_rng, work_rng = [np.random.default_rng(s) for s in root.spawn(3)]

    queues = [[] for _ in range(n_skills)]  # heap by original arrival order
    queue_count = np.zeros(n_skills, dtype=np.int64)
    in_system = np.zeros(n_skills, dtype=np.int64)
    jobs = [None] * n_servers
    serving_skills = np.full(n_servers, -1, dtype=int)
    started = np.zeros(n_servers)
    versions = np.zeros(n_servers, dtype=np.int64)
    idle = set(range(n_servers))
    eligible_servers = [tuple(np.flatnonzero(rates[:, k] > 0)) for k in range(n_skills)]
    eligible_skills = [tuple(np.flatnonzero(rates[s] > 0)) for s in range(n_servers)]
    arrivals = np.zeros((4, n_skills), dtype=np.int64)
    departures = np.zeros_like(arrivals)
    server_completions = np.zeros((4, n_servers, n_skills), dtype=np.int64)
    completed = _CompletedJobs(n_skills, buckets, collectWaitingTimeStatisticsInput)
    monitor = _TimeMonitor(schedule, n_servers, n_skills, maxQueue)
    event_heap = []
    event_seq = 0
    job_seq = 0
    event_count = 0
    preemptions = 0
    now = 0.0

    def push_event(when, kind, server=-1, version=-1):
        nonlocal event_seq
        event_seq += 1
        # Complete service before an arrival in an exact timestamp tie.
        heapq.heappush(event_heap, (when, kind, event_seq, server, version))

    def schedule_arrival():
        when = _next_arrival_x(now, edges[2], edges[3], multiplier, arrival_rng)
        push_event(when, 1)

    def start_job(server, skill):
        _, job = heapq.heappop(queues[skill])
        queue_count[skill] -= 1
        job.waiting += now - job.entered_queue
        jobs[server] = job
        serving_skills[server] = skill
        started[server] = now
        idle.discard(server)
        versions[server] += 1
        finish = now + job.remaining_work / normalized_rates[server, skill]
        push_event(finish, 0, server, int(versions[server]))

    def pause_job(server):
        nonlocal preemptions
        job = jobs[server]
        elapsed = now - started[server]
        job.service += elapsed
        job.remaining_work = max(
            0.0, job.remaining_work - elapsed * normalized_rates[server, job.skill]
        )
        job.entered_queue = now
        heapq.heappush(queues[job.skill], (job.sequence, job))
        queue_count[job.skill] += 1
        jobs[server] = None
        serving_skills[server] = -1
        versions[server] += 1  # invalidate the canceled departure event
        idle.add(server)
        preemptions += 1

    def fill_idle_servers():
        # Also handles a preempted job that can migrate to another idle server.
        for server in sorted(idle):
            candidate = None
            for skill in eligible_skills[server]:
                if queues[skill]:
                    key = (scores[server, skill], queues[skill][0][0], skill)
                    if candidate is None or key < candidate:
                        candidate = key
            if candidate is not None:
                start_job(server, candidate[2])

    def select_preemption_server(skill):
        """Lowest-priority running job on a server eligible for this skill.

        Smaller scores mean HIGHER priority. Only strict improvements on the
        candidate server are permitted. A global interpretation of job priority
        requires scores that are comparable across servers, e.g. one per class.
        Equal victim scores retain the original incoming-score/server-ID ties.
        """
        candidates = (
            s for s in eligible_servers[skill]
            if jobs[s] is not None
            and scores[s, skill] < scores[s, serving_skills[s]]
        )
        return min(
            candidates,
            key=lambda s: (
                -scores[s, serving_skills[s]],
                scores[s, skill],
                s,
            ),
            default=None,
        )

    def rebalance_queues():
        """Fill idle servers and repeat feasible strict-priority swaps.

        This is local reassignment, NOT an optimizer over all server/job
        matchings. It never migrates a running job solely to free a server.
        For server-specific scores, a waiting class is ranked using its score
        on its selected victim server; ties use original arrival sequence.
        """
        fill_idle_servers()
        if not preemptive:
            return
        while True:
            choice = None
            for skill in range(n_skills):
                if not queues[skill]:
                    continue
                server = select_preemption_server(skill)
                if server is None:
                    continue
                key = (scores[server, skill], queues[skill][0][0], skill, server)
                if choice is None or key < choice:
                    choice = key
            if choice is None:
                return
            _, _, skill, server = choice
            pause_job(server)
            start_job(server, skill)
            fill_idle_servers()

    schedule_arrival()
    while event_heap:
        when, kind, _, server, version = heapq.heappop(event_heap)
        if kind == 0 and version != versions[server]:
            continue  # lazily discard canceled departures
        monitor.advance(now, min(when, stop), queue_count, in_system, serving_skills)
        if when >= stop:
            now = stop
            break
        now = when
        phase = monitor.p
        event_count += 1
        if kind == 1:
            skill = int(np.searchsorted(cdf, skill_rng.random(), side="right"))
            job = _Job(job_seq, skill, now, phase, float(work_rng.exponential()), now)
            job_seq += 1
            arrivals[phase, skill] += 1
            in_system[skill] += 1
            heapq.heappush(queues[skill], (job.sequence, job))
            queue_count[skill] += 1
            if phase > 0:
                monitor.bin_arrivals[monitor.b, skill] += 1
            server = next((s for s in eligible_servers[skill] if s in idle), None)
            if server is None and preemptive:
                server = select_preemption_server(skill)
                if server is not None:
                    pause_job(server)
            if server is not None:
                start_job(server, skill)
            rebalance_queues()
            schedule_arrival()
        else:
            job = jobs[server]
            skill = job.skill
            job.service += now - started[server]
            # An accounting invariant, including all preemptions and migrations.
            if not math.isclose(job.waiting + job.service, now - job.arrival, rel_tol=1e-9, abs_tol=1e-8):
                raise RuntimeError("Waiting/service/sojourn accounting is inconsistent.")
            completed.record(job, now, rate)
            departures[phase, skill] += 1
            server_completions[phase, server, skill] += 1
            in_system[skill] -= 1
            if phase > 0:
                monitor.bin_departures[monitor.b, skill] += 1
            jobs[server] = None
            serving_skills[server] = -1
            idle.add(server)
            rebalance_queues()
    else:
        # Arrival scheduling normally guarantees an event beyond the horizon.
        monitor.advance(now, stop, queue_count, in_system, serving_skills)
        now = stop

    phase_durations = np.diff(edges)
    if not np.allclose(monitor.phase_duration, phase_durations, rtol=1e-9, atol=1e-8):
        raise RuntimeError("Observation windows were not fully integrated.")
    if not np.array_equal(arrivals.sum(axis=0) - departures.sum(axis=0), in_system):
        raise RuntimeError("Arrival/departure conservation failed.")
    in_service = np.bincount(serving_skills[serving_skills >= 0], minlength=n_skills)
    if not np.array_equal(in_system, queue_count + in_service):
        raise RuntimeError("Queue/service conservation failed.")

    bin_durations = np.diff(schedule["bin_edges_x"])
    queue_curve = monitor.bin_queue / bin_durations[:, None]
    system_curve = monitor.bin_system / bin_durations[:, None]
    phase_queue = _divide(monitor.phase_queue, phase_durations[:, None])
    phase_system = _divide(monitor.phase_system, phase_durations[:, None])
    phase_util = _divide(monitor.phase_busy, phase_durations[:, None, None])
    observation_duration = stop - edges[1]
    histogram = monitor.system_histogram / observation_duration
    baseline = phase_queue[1].copy()
    post_warmup_completions = server_completions[1:].sum(axis=0)
    phases = {}
    for p, name in enumerate(PHASES):
        phases[name] = {
            "expected_arrivals": schedule["expected_arrivals"][p],
            "duration": phase_durations[p] / rate,
            "normalized_duration": phase_durations[p],
            "actual_arrivals_by_skill": arrivals[p].copy(),
            "departures_by_skill": departures[p].copy(),
            "arrival_cohort_completed_by_skill": completed.count[p].copy(),
            "unfinished_arrival_cohort_by_skill": arrivals[p] - completed.count[p],
            "mean_queue_by_skill": phase_queue[p].copy(),
            "mean_queue_total": float(phase_queue[p].sum()),
            "mean_number_in_system_by_skill": phase_system[p].copy(),
            "max_queue_total": int(monitor.phase_peak[p]),
            "utilization": phase_util[p].copy(),
            "waiting": completed.summary([p]),
        }
    return {
        "schedule": schedule,
        "model": {"base_rate": rate, "n_servers": n_servers, "n_skills": n_skills,
                  "service_rates": rates, "skill_cdf": cdf, "costs": costs,
                  "priority_scores": scores, "preemptive": bool(preemptive)},
        "seed": {"entropy": root.entropy, "spawn_key": root.spawn_key},
        "normalized_time": schedule["normalized_time"].copy(),
        "normalized_time_edges": schedule["normalized_time_edges"].copy(),
        "time": (schedule["bin_edges_x"][:-1] + bin_durations / 2) / rate,
        "queue_by_skill": queue_curve,
        "queue_total": queue_curve.sum(axis=1),
        "number_in_system_by_skill": system_curve,
        "queue_excess_by_skill": queue_curve - baseline,
        "queue_excess_total": queue_curve.sum(axis=1) - baseline.sum(),
        "baseline_queue_by_skill": baseline,
        "baseline_queue_total": float(baseline.sum()),
        "bin_arrivals_by_skill": monitor.bin_arrivals,
        "bin_departures_by_skill": monitor.bin_departures,
        "phases": phases,
        "queue_reports": _queue_report_metrics(
            schedule, monitor.bin_queue, monitor.bin_arrivals, baseline
        ),
        "recovery": measure_recovery(
            schedule, queue_curve.sum(axis=1), float(baseline.sum()),
            recoveryAbsTolerance=recoveryAbsTolerance,
            recoveryRelTolerance=recoveryRelTolerance,
            recoveryHoldArrivals=recoveryHoldArrivals,
        ),
        "number_in_system_pmf": histogram[:, :maxQueue],
        "number_in_system_overflow": histogram[:, maxQueue],
        "utilization": monitor.phase_busy[1:].sum(axis=0) / observation_duration,
        "assignment_fractions": _divide(post_warmup_completions, post_warmup_completions.sum(axis=0)),
        "waiting": completed.summary([1, 2, 3]),
        "diagnostics": {
            "actual_arrivals": int(arrivals.sum()),
            "actual_departures": int(departures.sum()),
            "unfinished_by_skill": in_system.copy(),
            "final_queue_by_skill": queue_count.copy(),
            "final_in_service_by_skill": in_service,
            "processed_events": event_count,
            "preemptions": preemptions,
        },
    }


def _replication_summary(values: Sequence[Any]) -> dict[str, np.ndarray]:
    """Pointwise replication spread; SEM is not a simultaneous confidence band."""
    if len(values) == 0:
        return {"replications": np.array([], dtype=float),
                **{key: float("nan") for key in ("mean", "sd", "sem", "p10", "p90")}}
    values = np.stack(values)
    n = values.shape[0]
    sd = values.std(axis=0, ddof=1) if n > 1 else np.full(values.shape[1:], np.nan)
    return {
        "replications": values,
        "mean": values.mean(axis=0),
        "sd": sd,
        "sem": sd / np.sqrt(n),
        "p10": np.quantile(values, 0.10, axis=0),
        "p90": np.quantile(values, 0.90, axis=0),
    }


def _simulation_worker(kwargs):
    return simulation(**kwargs)


def simulation_batch_run(
    failureRates,
    serviceRates,
    skillServerAssignment,
    priorities=None,
    histogramBucketsInput=(0.0,),
    expectedTotalNumberRequests=100_000,
    numberWarmingUpRequests=10_000,
    replications=35,
    maxQueue=500,
    nCores=None,
    startSeed=10,
    *,
    numberBaselineRequests=20_000,
    numberSurgeRequests=10_000,
    surgeMultiplier=3.0,
    sampleEveryArrivals=100.0,
    queueReportEveryArrivals=10_000.0,
    recoveryAbsTolerance=1.0,
    recoveryRelTolerance=0.10,
    recoveryHoldArrivals=1_000.0,
    preemptive=True,
    collectWaitingTimeStatisticsInput=True,
    caseId=0,
    mpStartMethod=None,
    keepReplicationResults=False,
) -> dict[str, Any]:
    """Run replications with reproducible independent child random streams.

    The first eleven positional arguments match the old batch API. serviceRates
    may be a per-skill vector or a (servers, skills) matrix. priorities may be
    scalar, per-skill, or a (servers, skills) matrix; LOWER is higher priority.

    Use a distinct caseId per test case for separate streams. Same seed/caseId
    gives the same replications regardless of worker count. nCores=1 avoids
    multiprocessing and is convenient for notebook debugging. For processes,
    import this module and call the batch inside an if __name__ == '__main__'
    guard. Worker exceptions propagate; incomplete batches are never silently
    averaged as though all replications succeeded.
    """
    failures = np.asarray(failureRates, dtype=float)
    assignment = np.asarray(skillServerAssignment, dtype=float)
    if failures.ndim != 1 or failures.size == 0 or not np.all(np.isfinite(failures)) or np.any(failures < 0):
        raise ValueError("failureRates must be a finite nonnegative vector.")
    base_rate = _positive_float(failures.sum(), "sum(failureRates)")
    if (assignment.ndim != 2 or assignment.shape[0] == 0
            or assignment.shape[1] != failures.size
            or not np.all((assignment == 0) | (assignment == 1))):
        raise ValueError("skillServerAssignment must be a 0/1 (servers, skills) matrix.")
    service = np.asarray(serviceRates, dtype=float)
    if service.shape not in ((failures.size,), assignment.shape):
        raise ValueError("serviceRates must be per-skill or match the assignment matrix.")
    if not np.all(np.isfinite(service)) or np.any(service < 0):
        raise ValueError("serviceRates must be finite and nonnegative.")
    rates = assignment * service
    try:
        priority = np.broadcast_to(1.0 if priorities is None else np.asarray(priorities, dtype=float), rates.shape)
    except ValueError as exc:
        raise ValueError("priorities must broadcast to (servers, skills).") from exc
    if not np.all(np.isfinite(priority)) or np.any(priority < 0):
        raise ValueError("priorities must be finite and nonnegative.")
    costs = rates * priority
    cdf = np.cumsum(failures / base_rate)
    cdf[-1] = 1.0
    reps = _integer(replications, "replications", 1)
    startSeed = _integer(startSeed, "startSeed")
    caseId = _integer(caseId, "caseId")
    workers = min(reps, os.cpu_count() or 1) if nCores is None else min(reps, _integer(nCores, "nCores", 1))
    maxQueue = _integer(maxQueue, "maxQueue", 1)
    _validate_model(base_rate, cdf, rates, costs, histogramBucketsInput)
    make_surge_schedule(base_rate, expectedTotalNumberRequests, numberWarmingUpRequests,
                        numberBaselineRequests, numberSurgeRequests, surgeMultiplier, sampleEveryArrivals,
                        queueReportEveryArrivals=queueReportEveryArrivals)
    _validate_recovery_options(recoveryAbsTolerance, recoveryRelTolerance, recoveryHoldArrivals)
    seeds = np.random.SeedSequence([startSeed, caseId]).spawn(reps)
    shared = dict(
        requestArrivalRate=base_rate, skillDistribution=cdf,
        skillServerRates=rates, skillServerCosts=costs,
        histogramBucketsInput=histogramBucketsInput,
        expectedTotalNumberRequests=expectedTotalNumberRequests,
        numberWarmingUpRequests=numberWarmingUpRequests,
        collectWaitingTimeStatisticsInput=collectWaitingTimeStatisticsInput,
        numberBaselineRequests=numberBaselineRequests,
        numberSurgeRequests=numberSurgeRequests, surgeMultiplier=surgeMultiplier,
        sampleEveryArrivals=sampleEveryArrivals, maxQueue=maxQueue, preemptive=preemptive,
        queueReportEveryArrivals=queueReportEveryArrivals,
        recoveryAbsTolerance=recoveryAbsTolerance,
        recoveryRelTolerance=recoveryRelTolerance,
        recoveryHoldArrivals=recoveryHoldArrivals,
    )
    inputs = [dict(shared, seed=seed) for seed in seeds]
    if workers == 1:
        runs = [_simulation_worker(arguments) for arguments in inputs]
    else:
        context = mp.get_context(mpStartMethod)
        with context.Pool(processes=workers) as pool:
            runs = pool.map(_simulation_worker, inputs, chunksize=1)

    trajectory_keys = (
        "queue_total", "queue_by_skill", "number_in_system_by_skill",
        "queue_excess_total", "queue_excess_by_skill",
        "bin_arrivals_by_skill", "bin_departures_by_skill",
    )
    phase_keys = (
        "actual_arrivals_by_skill", "departures_by_skill",
        "arrival_cohort_completed_by_skill", "unfinished_arrival_cohort_by_skill",
        "mean_queue_total", "mean_queue_by_skill", "mean_number_in_system_by_skill",
        "max_queue_total", "utilization",
    )
    result = {
        "schedule": runs[0]["schedule"],
        "model": runs[0]["model"],
        "replications": reps, "start_seed": startSeed, "case_id": caseId,
        "normalized_time": runs[0]["normalized_time"],
        "normalized_time_edges": runs[0]["normalized_time_edges"],
        "trajectories": {key: _replication_summary([r[key] for r in runs]) for key in trajectory_keys},
        "baseline": {
            "queue_total": _replication_summary([r["baseline_queue_total"] for r in runs]),
            "queue_by_skill": _replication_summary([r["baseline_queue_by_skill"] for r in runs]),
        },
        "phases": {
            name: {key: _replication_summary([r["phases"][name][key] for r in runs]) for key in phase_keys}
            for name in PHASES
        },
        "queue_reports": {
            "windows": runs[0]["queue_reports"]["windows"],
            **{key: _replication_summary([r["queue_reports"][key] for r in runs])
               for key in ("mean_queue_total", "mean_queue_by_skill",
                           "mean_queue_excess_total", "mean_queue_excess_by_skill",
                           "actual_arrivals_total", "actual_arrivals_by_skill")},
        },
        "recovery": _summarize_recovery([r["recovery"] for r in runs]),
        "replication_diagnostics": [r["diagnostics"] for r in runs],
        "replication_waiting_statistics": [r["waiting"] for r in runs],
        "phase_waiting_statistics": {name: [r["phases"][name]["waiting"] for r in runs] for name in PHASES},
    }
    for key in ("number_in_system_pmf", "number_in_system_overflow", "utilization", "assignment_fractions"):
        result[key] = _replication_summary([r[key] for r in runs])
    if keepReplicationResults:
        result["raw_replications"] = runs
    return result


def aggregate_cases(caseResults: Sequence[dict[str, Any]], *, perServer=False) -> dict[str, Any]:
    """Equal-weight mean of case means, after each replication's baseline removal.

    The 10th/90th percentiles describe variation BETWEEN CASE MEANS, not a
    confidence interval. Different replication counts do not change a case's
    weight. All cases must have the same normalized experimental schedule.
    """
    cases = list(caseResults)
    if not cases:
        raise ValueError("Provide at least one case result.")
    reference = cases[0]
    for case in cases[1:]:
        for key in ("expected_arrivals", "rate_multipliers", "normalized_time_edges"):
            a, b = reference["schedule"][key], case["schedule"][key]
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-10):
                raise ValueError("Cases must have matching phase counts, multipliers, and normalized bins.")
    curves = np.stack([
        case["trajectories"]["queue_excess_total"]["mean"]
        / (case["model"]["n_servers"] if perServer else 1.0)
        for case in cases
    ])
    return {
        "normalized_time": reference["normalized_time"].copy(),
        "normalized_time_edges": reference["normalized_time_edges"].copy(),
        "normalized_surge_end": reference["schedule"]["normalized_surge_end"],
        "case_mean_curves": curves,
        "mean": curves.mean(axis=0),
        "p10": np.quantile(curves, 0.10, axis=0),
        "p90": np.quantile(curves, 0.90, axis=0),
        "n_cases": len(cases),
        "ylabel": "Additional people waiting per server" if perServer else "Additional people waiting above baseline",
    }

