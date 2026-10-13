"""
distribution_sensitivity.py -- E3: sensitivity of the final designs to the repair-time and interarrival-time
distributions (evaluation only; the designs and their base-stock levels are kept). For each design file (the rule
stored in it), each grid cell (arrival spec, service spec): 35 replications x 100,000 failures, seeds
4,000,000 + 1,000 case + 20 i (common across cells and design sets); per replication the total cost with S fixed, and
from the averaged distribution the cost with S re-optimized. Output: one row per (design set, instance, cell), plus a
per-replication file. Resumable.
    python3 distribution_sensitivity.py ../PSO_results_tuned_polished --out tables/E3_fcfs_designs.csv --workers 20
    python3 distribution_sensitivity.py ../PSO_results_tuned_rule_mu_polished --out tables/E3_mu_designs.csv --workers 20
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from simulation_codes import simulation_fast as sf
from simulation_codes import SimulationInterface_fast as SIf
from sku_surges import case_number, priorities, ebo_at

SEED0, REPS, N_REQ, MAXQ, MASS = 4_000_000, 35, 100_000, 2000, 1e-3
# (label, arrival spec, service spec); CV 1 exponential = the published model
GRID = [("base", None, None),
        ("svc_erlang4", None, ("erlang", 0.5)), ("svc_erlang2", None, ("erlang", 1 / np.sqrt(2))), ("svc_logn1.5", None, ("lognormal", 1.5)), ("svc_logn2", None, ("lognormal", 2.0)),
        ("arr_erlang4", ("erlang", 0.5), None), ("arr_h2", ("h2", 2.0), None),
        ("both_low", ("erlang", 0.5), ("erlang", 0.5)), ("both_high", ("h2", 2.0), ("lognormal", 2.0))]


def run_cell(task):
    path, rule, label, arrival, service = task
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[key], float) for key in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    X = np.asarray(d["best_assignment"], int); S = np.asarray(d["stocks"], int)
    fixed = f * (X.sum(axis=1) > 0).sum() + cs * X.sum()
    pr = priorities(rule, mu, h); R = X * mu[None, :]
    score = None if pr is None else np.where(R > 0, np.round(np.broadcast_to(pr, R.shape), 6), np.inf)
    base = {"design_set": os.path.basename(os.path.dirname(os.path.abspath(path))), "instance": c, "rule": rule, "cell": label,
            "arr": arrival[0] if arrival else "exponential", "arr_cv": arrival[1] if arrival else 1.0, "svc": service[0] if service else "exponential", "svc_cv": service[1] if service else 1.0,
            "N": d["m_skills"], "M": d["n_servers"], "rho": d.get("utilization_rate"), "f": d.get("machine_cost_factor"), "c": d.get("skill_cost_factor"), "nominal_total": d["total_cost"]}
    reps, acc, util = [], None, np.zeros(R.shape[0])
    try:
        for i in range(REPS):
            o = sf.simulate(lam, R, score, N_REQ, 0, SEED0 + 1000 * c + 20 * i, MAXQ, arrival=arrival, service=service)
            dist = o["dist"]; mass = dist.sum(axis=1)
            if np.any(mass < 1 - MASS):
                return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
            dist = dist / mass[:, None]; e = ebo_at(dist, S); util += o["util"].sum(axis=1) / REPS
            reps.append({"instance": c, "rule": rule, "cell": label, "replication": i, "total_fixed": float((h * S).sum() + b * e.sum() + fixed), "ebo_fixed": float(e.sum())})
            acc = dist if acc is None else acc + dist
    except RuntimeError:
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
    ehb, S2, E2 = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, acc / REPS)
    tf = np.array([r["total_fixed"] for r in reps])
    return dict(base, sim_stable=True, total_fixed=float(tf.mean()), total_fixed_sem=float(tf.std(ddof=1) / np.sqrt(REPS)), total_reopt=float(ehb + fixed),
                backorders_fixed=float(np.mean([r["ebo_fixed"] for r in reps])), units_reopt=int(S2.sum()), units_fixed=int(S.sum()), max_util=float(util.max())), reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", required=True); ap.add_argument("--rule", default=None)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--cases", type=int, nargs="*"); ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases: files = [p for p in files if case_number(p) in a.cases]
    done, rows = set(), []
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); done = set(zip(old.instance, old.rule, old.cell)); rows = old.to_dict("records")
    rep_out = a.out.replace(".csv", "_replications.csv"); rep_rows = pd.read_csv(rep_out).to_dict("records") if os.path.exists(rep_out) else []
    tasks = []
    for p in files:
        rule = a.rule or json.load(open(p)).get("rule", "fcfs")
        for label, arr, svc in GRID:
            if a.cells and label not in a.cells: continue
            if (case_number(p), rule, label) not in done: tasks.append((p, rule, label, arr, svc))
    print(f"{len(files)} designs, {len(tasks)} cells to run ({len(done)} done), {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (row, reps) in enumerate(pool.imap_unordered(run_cell, tasks, chunksize=1), 1):
            rows.append(row); rep_rows += reps
            if i % 50 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(a.out, index=False); pd.DataFrame(rep_rows).to_csv(rep_out, index=False)
                print(f"[{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows); print(f"done: {len(df)} cells; stable in simulation {df.sim_stable.mean():.1%}")
