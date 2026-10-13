"""
run_surge_cases.py -- R3 / E2 of the revision plan: simultaneous, finite-duration surge of all SKUs (Section 6.2), on the
re-optimized designs, with the compiled engine (surge_tools.surge_fast).

Per design (one result file of PSO_results_rerun/ or GA_results_rerun/), rule and surge multiplier gamma:
  35 replications of 100,000 expected failures: warm-up 10,000, baseline 20,000, surge 10,000 at gamma x the nominal
  failure rates of all SKUs, recovery 60,000; base-stock levels fixed at the design's (no re-optimization during the
  surge); bins of at most 100 expected failures.
Rules (lower value = higher priority; the definitions of surge_tools.priority_rules):
  FCFS  -- no priorities;
  mu    -- "fastest_service_first": faster repairs first (ranks of mu, ties by SKU index);
  hmu   -- "holding_cost_times_service_rate": larger h*mu first (dense ranks, exact ties share a rank).
Seeds: replication i of design c uses 2,000,000 + 1,000 c + 20 i, the same for all rules and multipliers (common
random numbers across rules); no stream is shared between designs.

Measured per replication (all in expected-failure units of time, i.e. Lambda * t, and in physical time):
  mean and peak backorders (sum over SKUs of max(N_k - S_k, 0)) and mean and peak waiting parts, per phase;
  cumulative backorder loss = integral over surge + recovery of (backorders - this replication's baseline mean), and
  the same against the design's steady-state EBO (sum of "EBO" in the design file; less noisy than a 20,000-failure
  baseline);
  recovery time = end of the first 1,000-failure stretch after the surge in which every bin mean is at most
  baseline + max(1, 10% of baseline), for backorders and for waiting parts; NaN and censored = 1 if not reached.

Usage (from this folder):
    python3 run_surge_cases.py ../PSO_results_rerun --out ../surge_results --gamma 1.5 --workers 2
    python3 run_surge_cases.py ../PSO_results_rerun --out /tmp/x --cases 1 2 --reps 5 --test     (quick check)
    python3 run_surge_cases.py --collect ../surge_results                                      (-> R3_transient.csv)
Finished (case, rule, gamma) records are skipped, so an interrupted run resumes.
"""
import argparse, glob, gzip, json, os, re, sys, time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surge_tools import surge_fast as fs
from surge_tools import surge_simulation as ss
from surge_tools.priority_rules import resolve_priorities

RULES = {"FCFS": None, "mu": "fastest_service_first", "hmu": "holding_cost_times_service_rate"}
PROTOCOL = dict(expectedTotalNumberRequests=100_000, numberWarmingUpRequests=10_000, numberBaselineRequests=20_000,
                numberSurgeRequests=10_000, sampleEveryArrivals=100.0, queueReportEveryArrivals=10_000.0,
                recoveryAbsTolerance=1.0, recoveryRelTolerance=0.10, recoveryHoldArrivals=1_000.0, maxQueue=2000)
REPS = 35
SEED0 = 2_000_000


def case_number(path):
    return int(re.search(r"case_(\d+)", os.path.basename(path)).group(1))


def opts_tag(opts):
    """File-name tag of the operating options: _np (non-preemptive base rule), _sw<rule>[np][D] (rule switched at the
    disruption start; D = back at the disruption end, otherwise until the end of the run)."""
    if not opts: return ""
    t = "_np" if opts.get("nonpreemptive") else ""
    if opts.get("switch_to"):
        t += f"_sw{opts['switch_to']}" + ("np" if opts.get("switch_np") else "") + ("D" if opts.get("switch_end") == "disruption" else "")
    return t


def out_name(out, c, rule, gamma, duration=10_000, absence=None, opts=None):
    tag = ("" if not absence else f"_abs{absence}") + opts_tag(opts)
    return os.path.join(out, f"surge_case_{c:04d}_{rule}_g{gamma:g}_d{duration}{tag}.json.gz")


def run_one(task):
    path, out, rule, gamma, reps, protocol, absence, opts = task
    c = case_number(path)
    target = out_name(out, c, rule, gamma, protocol["numberSurgeRequests"], absence, opts)
    if os.path.exists(target):
        return c, rule, gamma, "exists", 0.0
    d = json.load(open(path))
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    X = np.asarray(d["best_assignment"], int)
    S = np.asarray(d["stocks"], int)
    pr = resolve_priorities(RULES[rule], holding_costs=h, service_rates=mu, skill_server_assignment=X)
    abs_spec = None
    if absence:                                                       # B5: a technician absent during the "surge" phase (gamma = 1)
        util = np.asarray(d["utilizationRatesAverage"], float).sum(axis=1); emp = np.flatnonzero(X.sum(axis=1) > 0)
        j = int(np.argmax(util)) if absence == "busiest" else int(np.random.default_rng(c).choice(emp))
        abs_spec = (j, float(protocol["numberWarmingUpRequests"] + protocol["numberBaselineRequests"]),
                    float(protocol["numberWarmingUpRequests"] + protocol["numberBaselineRequests"] + protocol["numberSurgeRequests"]))
    t0d = float(protocol["numberWarmingUpRequests"] + protocol["numberBaselineRequests"]); t1d = t0d + float(protocol["numberSurgeRequests"])
    switch = sw_pr = None
    if opts.get("switch_to"):                                         # operational response: switch the rule at the disruption
        sw_pr = resolve_priorities(RULES[opts["switch_to"]], holding_costs=h, service_rates=mu, skill_server_assignment=X)["priorities"]
        switch = {"preemptive": not opts.get("switch_np"), "start": t0d, "end": t1d if opts.get("switch_end") == "disruption" else np.inf, "to": opts["switch_to"]}
    t = time.time()
    b = fs.simulation_batch_run(lam, mu, X, priorities=pr["priorities"], replications=reps, preemptive=not opts.get("nonpreemptive"),
                                startSeed=SEED0 + 1000 * c, surgeMultiplier=gamma, stocks=S, caseId=c, absence=abs_spec,
                                switch=switch, switch_priorities=sw_pr, **protocol)
    elapsed = time.time() - t
    sch = b["schedule"]; Lam = float(lam.sum())
    w = np.diff(sch["bin_edges_x"]); ph = sch["bin_phases"]; post = ph >= 2
    bo = b["trajectories"]["backorders_total"]["replications"]          # rep x bin
    bo_base = b["baseline"]["backorders_total"]["replications"]
    loss = ((bo[:, post] - bo_base[:, None]) * w[post]).sum(1)          # failure units x backorders
    ebo_nominal = float(np.sum(d["EBO"]))                               # steady state, 35 x 100,000 independent runs
    loss_nom = ((bo[:, post] - ebo_nominal) * w[post]).sum(1)
    reps_rows = []
    for i in range(reps):
        row = {"replication": i, "seed": b["seeds"][i], "cumulative_backorder_loss": float(loss[i]),
               "cumulative_backorder_loss_time": float(loss[i] / Lam),
               "cumulative_backorder_loss_vs_nominal": float(loss_nom[i]), "nominal_EBO": ebo_nominal}
        for p in ("baseline", "surge", "recovery"):
            P = b["phases"][p]
            row[f"{p}_mean_backorders"] = float(P["mean_backorders_total"]["replications"][i])
            row[f"{p}_peak_backorders"] = int(P["max_backorders_total"]["replications"][i])
            row[f"{p}_mean_queue"] = float(P["mean_queue_total"]["replications"][i])
            row[f"{p}_peak_queue"] = int(P["max_queue_total"]["replications"][i])
        for tag, rec in (("backorders", b["recovery_backorders"]), ("queue", b["recovery"])):
            r = rec["replication_results"][i]
            row[f"recovery_{tag}_arrivals"] = r["recovery_time_arrivals"]
            row[f"recovery_{tag}_time"] = r["recovery_time"]
            row[f"recovery_{tag}_censored"] = int(r["censored"])
        reps_rows.append(row)
    util = b["utilization"]["mean"].sum(1)
    record = {
        "case": c, "rule": rule, "priority_rule": pr["priority_rule"], "priorities": pr["priorities"],
        "gamma": gamma, "replications": reps, "seeds": b["seeds"], "preemptive": not opts.get("nonpreemptive"),
        "switch": None if switch is None else {k: (None if v == np.inf else v) for k, v in switch.items()},
        "absence": None if abs_spec is None else {"kind": absence, "technician": abs_spec[0], "start": abs_spec[1], "end": abs_spec[2],
                                                   "skills": int(X[abs_spec[0]].sum()), "utilization": float(np.asarray(d["utilizationRatesAverage"], float).sum(axis=1)[abs_spec[0]])},
        "factors": {k: d.get(k) for k in ("m_skills", "n_servers", "utilization_rate", "holding_cost_min",
                                          "holding_costs_variant", "machine_cost_factor", "skill_cost_factor")},
        "design_source": os.path.abspath(path), "design_set": os.path.basename(os.path.dirname(os.path.abspath(path))), "stocks": S, "assignment": X,
        "protocol": {**protocol, "surgeMultiplier": gamma, "engine": b["model"]["engine"]},
        "per_replication": reps_rows,
        "mean_trajectories": {"normalized_time": b["normalized_time"],
                              "backorders_total": b["trajectories"]["backorders_total"]["mean"],
                              "queue_total": b["trajectories"]["queue_total"]["mean"],
                              "backorders_total_sem": b["trajectories"]["backorders_total"]["sem"]},
        "utilization_by_technician_post_warmup": util,
        "overflow_share_max": float(b["number_in_system_overflow"]["mean"].max()),
        "elapsed_seconds": elapsed,
    }
    os.makedirs(out, exist_ok=True)
    tmp = target + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(ss_json(record), f)
    os.replace(tmp, target)
    return c, rule, gamma, "done", elapsed


def ss_json(x):
    if isinstance(x, dict):
        return {k: ss_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ss_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return ss_json(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def collect(folder):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "surge_case_*.json.gz"))):
        r = json.load(gzip.open(f, "rt"))
        for rep in r["per_replication"]:
            ab = r.get("absence") or {}
            rows.append({"instance": r["case"], "rule": r["rule"], "gamma": r["gamma"], "duration": r["protocol"]["numberSurgeRequests"], "design": r.get("design_set", ""),
                         "absence": ab.get("kind", ""), "absent_technician": ab.get("technician", -1), "absent_utilization": ab.get("utilization", np.nan), **r["factors"], **rep})
    df = pd.DataFrame(rows)
    out = os.path.join(folder, "R3_transient.csv")
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} rows, {df.instance.nunique() if len(df) else 0} instances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "surge_results"))
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.5])
    ap.add_argument("--durations", type=int, nargs="+", default=[10_000], help="expected failures during the surge (the recovery phase gets the rest of the 100,000)")
    ap.add_argument("--absence", choices=["busiest", "random"], default=None, help="B5: instead of a surge (gamma 1), the named technician is absent during the surge phase")
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--cases", type=int, nargs="*")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--test", action="store_true", help="short runs (20,000 failures) for a quick check")
    ap.add_argument("--collect", metavar="FOLDER")
    ap.add_argument("--nonpreemptive", action="store_true", help="non-preemptive priorities (base rule)")
    ap.add_argument("--switch-to", choices=["mu", "hmu"], default=None, help="switch to this rule at the start of the disruption (operational response)")
    ap.add_argument("--switch-np", action="store_true", help="the switched-to rule is non-preemptive")
    ap.add_argument("--switch-end", choices=["disruption", "run"], default="run", help="switch back at the end of the disruption, or keep the rule to the end of the run")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect); sys.exit()
    protocol = dict(PROTOCOL)
    if a.test:
        protocol.update(expectedTotalNumberRequests=20_000, numberWarmingUpRequests=2_000,
                        numberBaselineRequests=4_000, numberSurgeRequests=2_000, queueReportEveryArrivals=2_000.0)
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases:
        files = [f for f in files if case_number(f) in a.cases]
    if a.absence:
        a.gamma = [1.0]
    opts = {"nonpreemptive": a.nonpreemptive, "switch_to": a.switch_to, "switch_np": a.switch_np, "switch_end": a.switch_end}
    tasks = [(f, a.out, rule, g, a.reps, dict(protocol, numberSurgeRequests=D), a.absence, opts) for f in files for g in a.gamma for D in a.durations for rule in a.rules]
    print(f"{len(files)} designs x {len(a.gamma)} multiplier(s) x {len(a.durations)} duration(s) x {len(a.rules)} rules = {len(tasks)} tasks, "
          f"{a.reps} replications each, {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time(); done = 0
    with Pool(a.workers) as pool:
        for c, rule, g, status, el in pool.imap_unordered(run_one, tasks):
            done += 1
            if done % 50 == 0 or done == len(tasks): print(f"[{done}/{len(tasks)}] case {c:4d} {rule:4s} gamma {g:g}: {status} ({el:.1f} s)", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min")
