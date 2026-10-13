"""Run one assigned case and return a complete, strict-JSON-ready surge record.

Requires the accompanying surge_simulation.py and NumPy (Python >= 3.10).
The original ten positional arguments of Surge_phase_costs are retained.
No inventory optimizer is inferred from queue lengths. Inventory costs are
explicitly unavailable unless the caller supplies an inventory_evaluator.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np
from . import surge_simulation as _surge_engine
from . import surge_fast as _surge_fast
from . import priority_rules as _priority_rules
from .priority_rules import resolve_priorities

SCHEMA_VERSION = "1.0"


def to_json_ready(value: Any) -> Any:
    """Convert arrays/scalars; missing NaN -> null, +/-inf -> labeled strings.

    Infinity strings preserve unbounded CDF thresholds and ineligible routing
    scores, instead of conflating them with missing/censored measurements.
    Unknown objects and non-string dictionary keys raise rather than silently
    changing or dropping input data.
    """
    if isinstance(value, np.ndarray):
        return to_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return to_json_ready(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON record dictionaries must have string keys.")
        return {key: to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}.")


def save_case_json(
    record: Mapping[str, Any], path: str | Path, *,
    indent: int | None = None, overwrite: bool = False,
) -> Path:
    """Save a complete record. A .gz suffix selects gzip-compressed JSON.

    Write a temporary file first, so serialization errors do not leave a
    partially written destination. Existing files require overwrite=True.
    Do not have concurrent processes write to the same destination path.
    """
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ready = to_json_ready(record)
    fd, temporary = tempfile.mkstemp(prefix=".surge-", dir=destination.parent)
    os.close(fd)
    try:
        opener = gzip.open if destination.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            json.dump(ready, stream, allow_nan=False, ensure_ascii=False,
                      indent=indent, separators=(",", ":") if indent is None else None)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_case_json(path: str | Path) -> dict[str, Any]:
    """Load JSON/JSON.gz as plain Python types; does not recreate all arrays."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    def reject_constant(token):
        raise ValueError(f"Non-standard JSON numeric constant: {token}")

    with opener(path, "rt", encoding="utf-8") as stream:
        result = json.load(stream, parse_constant=reject_constant)
    if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Not a supported surge case record.")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _nonnegative_array(value: Any, name: str, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None:
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as exc:
            raise ValueError(f"{name} must broadcast to shape {shape}.") from exc
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite, nonnegative values.")
    return array.copy()


def _rep_summary(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    sd = values.std(axis=0, ddof=1) if n > 1 else np.full(values.shape[1:], np.nan)
    return {"replications": values, "mean": values.mean(axis=0), "sd": sd,
            "sem": sd / np.sqrt(n), "p10": np.quantile(values, 0.10, axis=0),
            "p90": np.quantile(values, 0.90, axis=0)}


def _small_summary(summary) -> dict[str, Any]:
    return {key: summary[key] for key in ("mean", "sd", "sem", "p10", "p90")}


def _prepare_inputs(assignment, m, n, failures, service, holding, penalty,
                    skill_cost, machine_cost, priorities):
    m = _integer(m, "m_skills", 1)
    n = _integer(n, "n_servers", 1)
    assignment = np.asarray(assignment)
    if assignment.shape != (n, m) or not np.all((assignment == 0) | (assignment == 1)):
        raise ValueError("optimized_assignment must be a 0/1 (n_servers, m_skills) matrix.")
    assignment = assignment.astype(np.int64)
    failures = _nonnegative_array(failures, "failure_rates")
    if failures.shape != (m,) or not np.isfinite(failures.sum()) or failures.sum() <= 0:
        raise ValueError("failure_rates must have length m_skills and a positive finite sum.")
    service = _nonnegative_array(service, "service_rates")
    if service.shape not in ((m,), (n, m)):
        raise ValueError("service_rates must be per-skill or a (servers, skills) matrix.")
    if np.any((failures > 0) & ~np.any(assignment * service > 0, axis=0)):
        raise ValueError("Every arriving skill must have an assigned positive-rate server.")
    holding = _nonnegative_array(holding, "holding_costs", (m,))
    priority_info = resolve_priorities(
        priorities, holding_costs=holding, service_rates=service,
        skill_server_assignment=assignment,
    )
    return {
        "optimized_assignment": assignment, "m_skills": m, "n_servers": n,
        "active_server_mask": assignment.any(axis=1),
        "active_server_count": int(assignment.any(axis=1).sum()),
        "failure_rates": failures, "service_rates": service,
        "base_arrival_rate": float(failures.sum()),
        "holding_costs": holding,
        "penalty_cost": _nonnegative_array(penalty, "penalty_cost", (m,)),
        "skill_costs": _nonnegative_array(skill_cost, "skill_costs", (n, m)),
        "machine_costs": _nonnegative_array(machine_cost, "machine_costs", (n,)),
        **priority_info,
    }


def _cost_record(case, batch, evaluator, cost_basis):
    installed = case["optimized_assignment"] * case["skill_costs"]
    servers = case["active_server_mask"] * case["machine_costs"]
    fixed = float(installed.sum() + servers.sum())
    if not np.isfinite(fixed):
        raise ValueError("Fixed costs overflow floating-point range.")
    result = {
        "cost_basis": cost_basis,
        "cost_of_skills": float(installed.sum()),
        "costs_of_servers": float(servers.sum()),
        "fixed_assignment_cost": fixed,
        "skill_cost_by_server_skill": installed,
        "server_cost_by_server": servers,
        "inventory_status": "not_computed",
        "inventory_evaluation_scope": None,
        "S": None, "EBO": None, "Ebo_holding": None,
        "holding_costs_array": None, "Holding_Costs_Range": None,
        "total_cost": None,
        "note": ("Queue lengths are not substituted for inventory backorders. "
                 "The original stock/inventory optimizer was not supplied. "
                 "holding_costs and penalty_cost are retained as inputs."),
    }
    if evaluator is None:
        return result
    inventory = evaluator(case_inputs=case, simulation_result=batch)
    if not isinstance(inventory, Mapping):
        raise TypeError("inventory_evaluator must return a mapping.")
    required = ("S", "EBO", "Ebo_holding", "evaluation_scope", "cost_basis")
    missing = [key for key in required if key not in inventory]
    if missing:
        raise ValueError(f"Inventory evaluator is missing keys: {missing}")
    if inventory["cost_basis"] != cost_basis:
        raise ValueError("Inventory and assignment costs must declare the same cost_basis.")
    if not isinstance(inventory["evaluation_scope"], str) or not inventory["evaluation_scope"].strip():
        raise ValueError("Inventory evaluation_scope must describe its observation/model scope.")
    m = case["m_skills"]
    stock = _nonnegative_array(inventory["S"], "S")
    ebo = _nonnegative_array(inventory["EBO"], "EBO")
    objective = _nonnegative_array(inventory["Ebo_holding"], "Ebo_holding")
    if stock.shape != (m,) or np.any(stock != np.floor(stock)) or ebo.shape != (m,):
        raise ValueError("S must be nonnegative integer stock levels and EBO a length-m vector.")
    if objective.ndim != 0:
        raise ValueError("Ebo_holding must be a finite, nonnegative scalar.")
    holding = stock * case["holding_costs"]
    positive = holding[holding > 0]
    total = fixed + float(objective)
    if not np.all(np.isfinite(holding)) or not np.isfinite(total):
        raise ValueError("Inventory costs overflow floating-point range.")
    result.update({
        "inventory_status": "computed_by_caller_evaluator",
        "inventory_evaluation_scope": inventory["evaluation_scope"],
        "S": stock, "EBO": ebo, "Ebo_holding": float(objective),
        "holding_costs_array": holding,
        "Holding_Costs_Range": float(np.ptp(positive)) if positive.size else None,
        "total_cost": total,
        "inventory_details": dict(inventory),
        "note": ("Inventory definitions and optimization are supplied by the evaluator. "
                 "total_cost = Ebo_holding + fixed_assignment_cost, on its declared cost basis. "
                 "No optimality claim is made for the supplied assignment."),
    })
    return result


def _utilization_record(batch, active_mask):
    def describe(summary):
        # Sum SKILL busy fractions before estimating SD/SEM across runs.
        by_server = np.asarray(summary["replications"]).sum(axis=2)
        ranges = np.ptp(by_server[:, active_mask], axis=1)
        return {
            "by_server": _rep_summary(by_server),
            "range_across_active_servers": _rep_summary(ranges),
            "range_of_server_means": float(np.ptp(by_server.mean(axis=0)[active_mask])),
        }
    return {
        "definition": ("Observed busy time / physical observation duration. "
                       "Includes busy work on unfinished and preempted jobs. "
                       "Ranges include assigned servers with zero observed utilization."),
        "overall": describe(batch["utilization"]),
        "by_phase": {name: describe(batch["phases"][name]["utilization"])
                     for name in _surge_engine.PHASES},
        "assignment_fraction_definition": ("simulation.assignment_fractions gives the "
            "server shares of post-warmup COMPLETIONS of each skill; these are not "
            "observed busy-time fractions or arrival-routing fractions."),
    }


def Surge_phase_costs(
    optimized_assignment, m_skills, n_servers, failure_rates, service_rates,
    holding_costs, penalty_cost, maxQueue, skill_costs, machine_costs,
    *,
    caseId=0, case_name=None,
    expectedTotalNumberRequests=100_000, replications=35,
    numberWarmingUpRequests=10_000, numberBaselineRequests=20_000,
    numberSurgeRequests=10_000, surgeMultiplier=3.0,
    sampleEveryArrivals=100.0, queueReportEveryArrivals=10_000.0,
    recoveryAbsTolerance=1.0, recoveryRelTolerance=0.10,
    recoveryHoldArrivals=1_000.0,
    priorities="fastest_service_first", preemptive=True,
    histogramBucketsInput=(0.0, 1.0, 5.0, 10.0, np.inf),
    collectWaitingTimeStatisticsInput=True,
    nCores=None, startSeed=12345, mpStartMethod=None,
    inventory_evaluator: Callable[..., Mapping[str, Any]] | None = None,
    cost_basis: str | None = None,
    output_path: str | Path | None = None,
    print_report=False,
    engine="python",
    stocks=None,
) -> dict[str, Any]:
    """Run a single CASE with many replications and return JSON-safe dictionaries.

    Default expected arrivals: 10k warmup + 20k baseline + 10k surge + 60k
    recovery. These are expected counts, not exact arrival-count stopping rules.
    All batch outputs and replication arrays are retained under `simulation`.

    Default priorities reproduce the original descending-service-rate ranks;
    ties receive distinct ranks in increasing skill-index order. Pass None for
    equal priority, or explicit values to override. The new named rules
    holding_cost_first and holding_cost_times_service_rate rank descending
    cost and cost-times-rate respectively, preserving exact ties. The latter
    supports per-server service rates and ranks eligible skills on each server.
    See resolve_priorities() for definitions and previewing resolved ranks.

    skill_costs may be scalar, per-skill, or server-by-skill; machine_costs
    scalar or per-server. Cost inputs must already use compatible units/bases.
    holding_costs and penalty_cost are NOT interpreted as queue cost rates.

    Optional inventory_evaluator(case_inputs=..., simulation_result=...) must
    not mutate its arguments and must return S, EBO, Ebo_holding,
    evaluation_scope, and cost_basis. The latter must match the nonempty
    cost_basis passed here. This hook is executed once, in the parent process.
    Without it, inventory outputs and total_cost are None, not invented zeros.

    Use a distinct caseId for each case and call under a __main__ guard when
    nCores > 1. nCores=1 is suitable for notebooks or an outer case-level pool.

    engine="fast" runs the compiled engine (surge_fast.py: the dispatch rules of simulation_fast, about 250 times
    faster, preemptive only, no waiting-time statistics; replication i uses seed startSeed + 20 i, so give each case
    its own startSeed). stocks (base-stock levels, fast engine only) adds backorder trajectories and recovery.
    """
    if engine not in ("python", "fast"):
        raise ValueError("engine must be 'python' or 'fast'.")
    if stocks is not None and engine != "fast":
        raise ValueError("stocks (backorders) need engine='fast'.")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(output_path)
    caseId = _integer(caseId, "caseId")
    if case_name is not None and not isinstance(case_name, str):
        raise ValueError("case_name must be a string or None.")
    if cost_basis is not None and (not isinstance(cost_basis, str) or not cost_basis.strip()):
        raise ValueError("cost_basis must be a nonempty description or None.")
    if inventory_evaluator is not None and (not callable(inventory_evaluator) or cost_basis is None):
        raise ValueError("An inventory_evaluator requires a callable and an explicit cost_basis.")
    case = _prepare_inputs(optimized_assignment, m_skills, n_servers, failure_rates,
        service_rates, holding_costs, penalty_cost, skill_costs, machine_costs, priorities)
    case["case_id"] = caseId
    case["case_name"] = case_name or f"case_{caseId}"
    case["cost_basis"] = cost_basis
    options = {
        "expectedTotalNumberRequests": expectedTotalNumberRequests,
        "numberWarmingUpRequests": numberWarmingUpRequests,
        "numberBaselineRequests": numberBaselineRequests,
        "numberSurgeRequests": numberSurgeRequests,
        "surgeMultiplier": surgeMultiplier, "replications": replications,
        "sampleEveryArrivals": sampleEveryArrivals,
        "queueReportEveryArrivals": queueReportEveryArrivals,
        "recoveryAbsTolerance": recoveryAbsTolerance,
        "recoveryRelTolerance": recoveryRelTolerance,
        "recoveryHoldArrivals": recoveryHoldArrivals,
        "histogramBucketsInput": np.asarray(histogramBucketsInput, dtype=float),
        "collectWaitingTimeStatisticsInput": collectWaitingTimeStatisticsInput,
        "preemptive": preemptive, "maxQueue": maxQueue,
        "nCores": nCores, "startSeed": startSeed, "caseId": caseId,
        "mpStartMethod": mpStartMethod,
    }
    started = time.perf_counter()
    if engine == "fast":
        options["collectWaitingTimeStatisticsInput"] = False
    batch = (_surge_fast if engine == "fast" else _surge_engine).simulation_batch_run(
        failureRates=case["failure_rates"], serviceRates=case["service_rates"],
        skillServerAssignment=case["optimized_assignment"],
        priorities=case["priorities"], **options,
        **({"stocks": stocks} if stocks is not None else {}),
    )
    elapsed = time.perf_counter() - started
    costs = _cost_record(case, batch, inventory_evaluator, cost_basis)
    utilization = _utilization_record(batch, case["active_server_mask"])
    recovery = batch["recovery"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "inputs": case,
        "configuration": {
            **options,
            "priorities": priorities,
            "priority_rule_name": case["priority_rule_name"],
            "priority_resolution_version": case["priority_resolution_version"],
            "engine": engine, "stocks": stocks,
        },
        "costs": costs,
        "utilization_analysis": utilization,
        "summary": {
            "baseline_queue": _small_summary(batch["baseline"]["queue_total"]),
            "surge_queue": _small_summary(batch["phases"]["surge"]["mean_queue_total"]),
            "post_surge_queue": _small_summary(batch["phases"]["recovery"]["mean_queue_total"]),
            "surge_peak_queue": _small_summary(batch["phases"]["surge"]["max_queue_total"]),
            "recovered_count": recovery["recovered_count"],
            "censored_count": recovery["censored_count"],
            "recovered_fraction": recovery["recovered_fraction"],
            "recovery_arrival_units_recovered_only": _small_summary(
                recovery["conditional_on_recovery"]["time_arrivals"]),
            "recovery_physical_time_recovered_only": _small_summary(
                recovery["conditional_on_recovery"]["time"]),
        },
        "simulation": batch,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "simulation_elapsed_seconds": elapsed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "simulator_sha256": hashlib.sha256(Path((_surge_fast if engine == "fast" else _surge_engine).__file__).read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "priority_rules_sha256": hashlib.sha256(
                Path(_priority_rules.__file__).read_bytes()
            ).hexdigest(),
            "array_axes": {
                "trajectories.*.replications": "replication, time_bin[, skill]",
                "queue_reports.*.replications": "replication, report_window[, skill]",
                "utilization.replications": "replication, server, skill",
                "number_in_system_pmf.replications": "replication, skill, state_below_maxQueue",
            },
            "json_encoding": {"NaN": "null (missing/undefined)",
                "positive_infinity": "Infinity string", "negative_infinity": "-Infinity string",
                "arrays": "nested lists", "replication_data": "all batch replication arrays retained"},
            "interpretation": [
                "Arrival phase budgets are expected counts; actual counts fluctuate.",
                "Queue excludes customers in service; number in system includes them.",
                "Bin/report means are time weighted, not averages of event observations.",
                "Normalized time = baseline arrival rate * (time - surge start).",
                "Undefined/censored recovery times are null; consult explicit censor flags.",
                "Recovery confirmation is a bin-mean threshold rule, not permanent recovery.",
                "Completed-job waiting statistics may be censored by the finite horizon.",
                "The occupation PMF is post-warmup/transient, not a stationary distribution.",
                "PMF overflow is explicit; maxQueue does not cap the simulated queue.",
                "simulation.model.costs contains routing scores, not financial costs.",
                "Named cost-based priorities use holding costs as dispatch indices, not inventory EBO estimates.",
                "Numeric priorities are lower-is-higher; recorded dispatch metric values are higher-is-higher.",
                "Fixed assignment cost is recorded once, not multiplied by replications or phases.",
            ],
        },
    }
    if print_report:
        print(f"Priority rule: {case['priority_rule']}")
        _surge_engine.print_surge_report(batch, label=case["case_name"])
        print(f"Fixed assignment cost: {costs['fixed_assignment_cost']:.6g}")
        print(f"Inventory cost status: {costs['inventory_status']}")
    ready = to_json_ready(record)
    if output_path is not None:
        save_case_json(ready, output_path)
    return ready


def _case_row_arguments(case, assignment_matrix, *, maxQueue=500,
                        priorities="fastest_service_first"):
    """Map the benchmark's singular cost names to the original ten inputs."""
    if not isinstance(case, Mapping):
        raise TypeError("case must be a mapping; for a pandas row use row.to_dict().")
    required = ("m_skills", "n_servers", "failure_rates", "service_rates",
                "holding_costs", "penalty_cost", "skill_cost", "machine_cost")
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"Input case is missing required fields: {missing}")

    def dimension(value, name):
        # DataFrame rows can promote integer-valued dimensions to floats.
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer.")
        if isinstance(value, (int, np.integer)):
            return _integer(value, name, 1)
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and float(value).is_integer():
                return _integer(int(value), name, 1)
        raise ValueError(f"{name} must be a positive integer, not {value!r}.")

    if isinstance(assignment_matrix, str):
        try:
            assignment_matrix = json.loads(assignment_matrix)
        except json.JSONDecodeError as exc:
            raise ValueError("assignment_matrix is not a valid JSON matrix.") from exc
    arguments = {
        "optimized_assignment": np.asarray(assignment_matrix),
        "m_skills": dimension(case["m_skills"], "m_skills"),
        "n_servers": dimension(case["n_servers"], "n_servers"),
        "failure_rates": np.asarray(case["failure_rates"], dtype=float),
        "service_rates": np.asarray(case["service_rates"], dtype=float),
        "holding_costs": np.asarray(case["holding_costs"], dtype=float),
        "penalty_cost": case["penalty_cost"],
        "maxQueue": _integer(maxQueue, "maxQueue", 1),
        "skill_costs": np.asarray(case["skill_cost"], dtype=float),
        "machine_costs": case["machine_cost"],
    }
    # Validate before launching workers; do NOT silently transpose or resize.
    _prepare_inputs(
        arguments["optimized_assignment"], arguments["m_skills"],
        arguments["n_servers"], arguments["failure_rates"],
        arguments["service_rates"], arguments["holding_costs"],
        arguments["penalty_cost"], arguments["skill_costs"],
        arguments["machine_costs"], priorities,
    )
    return arguments


def Surge_phase_costs_from_case(
    case: Mapping[str, Any], assignment_matrix, *,
    caseId=0, case_name=None, maxQueue=500,
    source: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    **simulation_options,
) -> dict[str, Any]:
    """Run a benchmark row with its separately stored PSO assignment.

    case uses the exact source keys: n_servers, m_skills, failure_rates,
    service_rates, holding_costs, penalty_cost, skill_cost, machine_cost.
    Optional total_cost/runtime and every other source field are preserved
    under source.input_case. They are reference optimization data, NOT newly
    estimated surge/inventory costs.

    Defaults (including 100,000 expected arrivals and 35 replications) are
    inherited from Surge_phase_costs. All its simulation keyword options are
    accepted. output_path writes a single JSON or JSON.gz object, not JSONL.
    """
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(output_path)
    if source is not None and not isinstance(source, Mapping):
        raise TypeError("source must be a mapping or None.")
    provenance = to_json_ready(dict(source or {}))
    provenance.update({
        "input_case": to_json_ready(dict(case)),
        "reference_total_cost": to_json_ready(case.get("total_cost")),
        "reference_runtime": to_json_ready(case.get("runtime")),
        "reference_note": (
            "total_cost and runtime belong to the original input optimization. "
            "They are retained for reference, not used as surge cost estimates."
        ),
    })
    arguments = _case_row_arguments(
        case, assignment_matrix, maxQueue=maxQueue,
        priorities=simulation_options.get("priorities", "fastest_service_first"),
    )
    overlap = set(arguments).intersection(simulation_options)
    if overlap:
        raise TypeError(f"Do not override case-derived inputs: {sorted(overlap)}")
    record = Surge_phase_costs(
        **arguments, caseId=caseId, case_name=case_name, **simulation_options,
    )
    record["source"] = provenance
    if output_path is not None:
        save_case_json(record, output_path)
    return record


def aggregate_case_records(
    records: Sequence[Mapping[str, Any]], *, per_active_server: bool = False,
) -> dict[str, Any]:
    """Aggregate returned or JSON-loaded records without rebuilding all arrays.

    Equal weight per case after within-case replication averaging and baseline
    subtraction. The band describes variation across CASE MEANS, not a CI.
    """
    if not records:
        raise ValueError("Provide at least one case record.")
    reference = records[0]["simulation"]
    curves = []
    for record in records:
        batch = record["simulation"]
        for key in ("expected_arrivals", "rate_multipliers", "normalized_time_edges"):
            a = np.asarray(reference["schedule"][key], dtype=float)
            b = np.asarray(batch["schedule"][key], dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-10):
                raise ValueError("Case schedules must match in arrival-normalized units.")
        scale = record["inputs"]["active_server_count"] if per_active_server else 1
        if scale <= 0:
            raise ValueError("Cannot normalize by zero active servers.")
        curves.append(np.asarray(batch["trajectories"]["queue_excess_total"]["mean"], float) / scale)
    curves = np.stack(curves)
    return to_json_ready({
        "case_ids": [r["inputs"]["case_id"] for r in records],
        "n_cases": len(records), "per_active_server": per_active_server,
        "normalized_time": reference["normalized_time"],
        "normalized_time_edges": reference["normalized_time_edges"],
        "normalized_surge_end": reference["schedule"]["normalized_surge_end"],
        "case_mean_curves": curves, "mean": curves.mean(axis=0),
        "p10": np.quantile(curves, 0.10, axis=0),
        "p90": np.quantile(curves, 0.90, axis=0),
        "ylabel": "Additional people waiting per active server" if per_active_server
                  else "Additional people waiting above baseline",
        "band_definition": "10th-90th percentiles across case means, NOT a confidence interval",
    })
