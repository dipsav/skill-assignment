"""
sku_surges.py -- B1 / Section 6.1: permanent surge of the failure rate of ONE SKU at a time, on the final designs.

Per design set (a result folder; the rule is the one stored in each file unless --rule overrides it), instance, SKU k
and surge level delta: lambda_k -> lambda_k (1 + delta), all other rates nominal; the skill matrix and the base-stock
levels are kept as designed (decision D3: stock fixed; the cost with stock re-optimized is reported as well).
  - LP stability: constraints (2)-(7) with the increased workload (checkSkillAssignment, max load 0.999);
  - simulated stability: a scenario is "unstable in simulation" if a queue-length distribution puts more than 0.1% of
    its mass beyond 2,000 parts (the interface's guard), which under preemptive priorities can happen although the LP is
    feasible (the lowest-priority SKU starves);
  - cost: 35 replications x 100,000 failures, independent seeds 3,000,000 + 1,000 case + 20 i, the same for every rule
    and design set (common random numbers across the arms); per replication the total cost with S fixed
    (holding + b EBO(S) + technicians + skills) and, from the averaged distribution, the cost with S re-optimized.
Output: one CSV row per (design set, instance, SKU, delta) with means over replications and the stability flags, plus
a per-replication CSV for the paired tests.

    python3 sku_surges.py ../PSO_results_tuned_polished --out tables/B1_fcfs_designs.csv --deltas 0.10 0.25 0.40 0.60 --workers 20
    python3 sku_surges.py ../PSO_results_tuned_polished --rule mu --out tables/B1_fcfs_designs_under_mu.csv ...
Resumable: scenarios already in the output CSV are skipped.
"""
import argparse, glob, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from simulation_codes import simulation_fast as sf
from simulation_codes import SimulationInterface_fast as SIf

SEED0, REPS, N_REQ, MAXQ, MASS = 3_000_000, 35, 100_000, 2000, 1e-3
PREEMPTIVE = True                    # --nonpreemptive sets False (module global, inherited by the forked workers)


def case_number(path):
    return int(re.search(r"case_(\d+)", os.path.basename(path)).group(1))


def priorities(rule, mu, h):
    return {"fcfs": None, "mu": 1.0 / mu, "hmu": 1.0 / (h * mu)}[rule]


def ebo_at(dist, S):
    n = np.arange(dist.shape[1]); return np.array([np.sum(dist[k] * np.maximum(n - S[k], 0)) for k in range(len(S))])


def run_scenario(task):
    path, rule, k, delta = task
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[key], float) for key in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    X = np.asarray(d["best_assignment"], int); S = np.asarray(d["stocks"], int)
    lam2 = lam.copy(); lam2[k] *= 1 + delta
    fixed = f * (X.sum(axis=1) > 0).sum() + cs * X.sum()
    lp = bool(SIf.checkSkillAssignment(X, lam2, mu)[0])
    base = {"design_set": os.path.basename(os.path.dirname(os.path.abspath(path))), "instance": c, "rule": rule, "preemptive": PREEMPTIVE, "sku": k, "delta": delta,
            "lp_feasible": lp, "N": d["m_skills"], "M": d["n_servers"], "rho": d.get("utilization_rate"), "f": d.get("machine_cost_factor"),
            "c": d.get("skill_cost_factor"), "nominal_total": d["total_cost"], "surge_workload_share": float(lam2[k] / mu[k] / (lam2 / mu).sum())}
    if not lp:
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
    pr = priorities(rule, mu, h); R = X * mu[None, :]
    score = None if pr is None else np.where(R > 0, np.round(np.broadcast_to(pr, R.shape), 6), np.inf)
    reps, acc = [], None
    try:
        for i in range(REPS):
            o = sf.simulate(lam2, R, score, N_REQ, 0, SEED0 + 1000 * c + 20 * i, MAXQ, preemptive=PREEMPTIVE)
            dist = o["dist"]; mass = dist.sum(axis=1)
            if np.any(mass < 1 - MASS):                                   # queue past 2,000 parts: unstable in simulation
                return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
            dist = dist / mass[:, None]; e = ebo_at(dist, S)
            reps.append({"instance": c, "rule": rule, "sku": k, "delta": delta, "replication": i, "total_fixed": float((h * S).sum() + b * e.sum() + fixed), "ebo_fixed": float(e.sum())})
            acc = dist if acc is None else acc + dist
    except RuntimeError:                                                  # queue capacity exceeded
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
    ehb, S2, E2 = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, acc / REPS)
    tf = np.array([r["total_fixed"] for r in reps])
    return dict(base, sim_stable=True, total_fixed=float(tf.mean()), total_fixed_sem=float(tf.std(ddof=1) / np.sqrt(REPS)),
                total_reopt=float(ehb + fixed), backorders_fixed=float(np.mean([r["ebo_fixed"] for r in reps])), units_reopt=int(S2.sum()), units_fixed=int(S.sum())), reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", required=True); ap.add_argument("--rule", default=None, help="override the rule stored in the files (fcfs|mu|hmu)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.10, 0.25, 0.40, 0.60]); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--nonpreemptive", action="store_true")
    ap.add_argument("--cases", type=int, nargs="*"); ap.add_argument("--skus", type=int, nargs="*", help="restrict to these SKU indices (tests)")
    a = ap.parse_args()
    PREEMPTIVE = not a.nonpreemptive
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases: files = [p for p in files if case_number(p) in a.cases]
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); done = set(zip(old.instance, old.rule, old.sku, old.delta)); rows = old.to_dict("records")
    else:
        rows = []
    rep_out = a.out.replace(".csv", "_replications.csv"); rep_rows = pd.read_csv(rep_out).to_dict("records") if os.path.exists(rep_out) else []
    tasks = []
    for p in files:
        d = json.load(open(p)); rule = a.rule or d.get("rule", "fcfs"); N = d["m_skills"]
        for k in (a.skus if a.skus else range(N)):
            for delta in a.deltas:
                if (case_number(p), rule, k, delta) not in done: tasks.append((p, rule, k, delta))
    print(f"{len(files)} designs, {len(tasks)} scenarios to run ({len(done)} done), {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (row, reps) in enumerate(pool.imap_unordered(run_scenario, tasks, chunksize=4), 1):
            rows.append(row); rep_rows += reps
            if i % 200 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(a.out, index=False); pd.DataFrame(rep_rows).to_csv(rep_out, index=False)
                print(f"[{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    print(f"done: {len(df)} scenarios; LP-feasible {df.lp_feasible.mean():.1%}; stable in simulation {df.sim_stable.mean():.1%}")
