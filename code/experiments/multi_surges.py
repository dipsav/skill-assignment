"""
multi_surges.py -- B2 / B3: permanent surges of SEVERAL SKUs at once, on the final designs (same protocol and seeds as
sku_surges.py, so the one-SKU, all-SKU and correlated surges are comparable).

Scenarios per instance:
  all      -- every SKU's failure rate x (1 + delta), delta in {5, 10, 20}%  (R7.10 (i));
  busiest  -- the SKUs in the skill set of the busiest technician (under the design's own rule, nominal rates)
              x (1 + delta), delta in {10, 25}%: a correlated surge that hits one technician's whole workload;
  random   -- the SKUs in the skill set of a technician drawn at random (seed = case), delta in {10, 25}%.
Per scenario: LP stability, simulated stability (mass beyond 2,000 parts), cost with stock fixed (per replication) and
with stock re-optimized. Output as sku_surges.py (a row per scenario, a per-replication file).

    python3 multi_surges.py ../PSO_results_tuned_polished --out tables/B23_fcfs_designs.csv --workers 20
    python3 multi_surges.py ../PSO_results_tuned_polished --rule mu --out tables/B23_fcfs_designs_under_mu.csv --workers 20
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from simulation_codes import simulation_fast as sf
from simulation_codes import SimulationInterface_fast as SIf
from sku_surges import case_number, priorities, ebo_at, SEED0, REPS, N_REQ, MAXQ, MASS
PREEMPTIVE = True

SCENARIOS = {"all": (0.05, 0.10, 0.20), "busiest": (0.10, 0.25), "random": (0.10, 0.25)}


def surged_skus(d, rule, kind):
    """The SKUs whose failure rates rise: all; the skill set of the busiest technician; of a random technician."""
    X = np.asarray(d["best_assignment"], int); N = X.shape[1]
    if kind == "all":
        return list(range(N)), -1
    util = np.asarray(d["utilizationRatesAverage"], float).sum(axis=1)          # stored: nominal rates, the design's own rule
    if kind == "busiest":
        j = int(np.argmax(util))
    else:
        emp = np.flatnonzero(X.sum(axis=1) > 0); j = int(np.random.default_rng(case_number(d["_path"])).choice(emp))
    return [int(k) for k in np.flatnonzero(X[j])], j


def run_scenario(task):
    path, rule, kind, delta = task
    d = json.load(open(path)); d["_path"] = path; c = case_number(path)
    lam, mu, h = (np.asarray(d[key], float) for key in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    X = np.asarray(d["best_assignment"], int); S = np.asarray(d["stocks"], int)
    skus, tech = surged_skus(d, rule, kind)
    lam2 = lam.copy(); lam2[skus] *= 1 + delta
    fixed = f * (X.sum(axis=1) > 0).sum() + cs * X.sum()
    lp = bool(SIf.checkSkillAssignment(X, lam2, mu)[0])
    base = {"design_set": os.path.basename(os.path.dirname(os.path.abspath(path))), "instance": c, "rule": rule, "preemptive": PREEMPTIVE, "scenario": kind, "delta": delta,
            "surged_skus": len(skus), "technician": tech, "lp_feasible": lp, "N": d["m_skills"], "M": d["n_servers"], "rho": d.get("utilization_rate"),
            "f": d.get("machine_cost_factor"), "c": d.get("skill_cost_factor"), "nominal_total": d["total_cost"],
            "workload_increase": float((lam2 / mu).sum() / (lam / mu).sum() - 1)}
    if not lp:
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
    pr = priorities(rule, mu, h); R = X * mu[None, :]
    score = None if pr is None else np.where(R > 0, np.round(np.broadcast_to(pr, R.shape), 6), np.inf)
    reps, acc = [], None
    try:
        for i in range(REPS):
            o = sf.simulate(lam2, R, score, N_REQ, 0, SEED0 + 1000 * c + 20 * i, MAXQ, preemptive=PREEMPTIVE)
            dist = o["dist"]; mass = dist.sum(axis=1)
            if np.any(mass < 1 - MASS):
                return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
            dist = dist / mass[:, None]; e = ebo_at(dist, S)
            reps.append({"instance": c, "rule": rule, "scenario": kind, "delta": delta, "replication": i, "total_fixed": float((h * S).sum() + b * e.sum() + fixed), "ebo_fixed": float(e.sum())})
            acc = dist if acc is None else acc + dist
    except RuntimeError:
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf, backorders_fixed=np.inf), []
    ehb, S2, E2 = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, acc / REPS)
    tf = np.array([r["total_fixed"] for r in reps])
    return dict(base, sim_stable=True, total_fixed=float(tf.mean()), total_fixed_sem=float(tf.std(ddof=1) / np.sqrt(REPS)),
                total_reopt=float(ehb + fixed), backorders_fixed=float(np.mean([r["ebo_fixed"] for r in reps])), units_reopt=int(S2.sum()), units_fixed=int(S.sum())), reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", required=True); ap.add_argument("--rule", default=None)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--nonpreemptive", action="store_true"); ap.add_argument("--cases", type=int, nargs="*"); ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    a = ap.parse_args()
    PREEMPTIVE = not a.nonpreemptive
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases: files = [p for p in files if case_number(p) in a.cases]
    done = set(); rows = []
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); done = set(zip(old.instance, old.rule, old.scenario, old.delta)); rows = old.to_dict("records")
    rep_out = a.out.replace(".csv", "_replications.csv"); rep_rows = pd.read_csv(rep_out).to_dict("records") if os.path.exists(rep_out) else []
    tasks = []
    for p in files:
        rule = a.rule or json.load(open(p)).get("rule", "fcfs")
        for kind in a.scenarios:
            for delta in SCENARIOS[kind]:
                if (case_number(p), rule, kind, delta) not in done: tasks.append((p, rule, kind, delta))
    print(f"{len(files)} designs, {len(tasks)} scenarios to run ({len(done)} done), {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (row, reps) in enumerate(pool.imap_unordered(run_scenario, tasks, chunksize=2), 1):
            rows.append(row); rep_rows += reps
            if i % 100 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(a.out, index=False); pd.DataFrame(rep_rows).to_csv(rep_out, index=False)
                print(f"[{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows); print(f"done: {len(df)} scenarios; LP-feasible {df.lp_feasible.mean():.1%}; stable in simulation {df.sim_stable.mean():.1%}")
