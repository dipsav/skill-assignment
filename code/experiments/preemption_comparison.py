"""
preemption_comparison.py -- R7.2: preemptive vs non-preemptive priority rules (evaluation of the final designs).
Arms per instance: the FCFS design under FCFS, mu and h x mu (sequential approach); the mu design under mu; the h x mu
design under h x mu; full cross-training (X = 1, its FCFS stock) under FCFS, mu and h x mu -- each priority rule both
preemptive (resume) and non-preemptive. Cells: exponential repairs (the model), Erlang-4 (CV 0.5) and lognormal (CV 2)
repairs. 35 replications x 100,000 failures, seeds 4,000,000 + 1,000 case + 20 i (as E3; common across arms and cells),
general kernel for every arm. Per arm and cell: total cost with S fixed and re-optimized, units, interruptions per failure.
    python3 preemption_comparison.py --out tables/P1_preemption.csv --workers 20
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from simulation_codes import simulation_fast as sf
from simulation_codes import SimulationInterface_fast as SIf
from sku_surges import case_number, priorities, ebo_at

SEED0, REPS, N_REQ, MAXQ, MASS = 4_000_000, 35, 100_000, 2000, 1e-3
ROOT = os.path.join(HERE, "..")
DESIGNS = {"fcfs": "PSO_results_tuned_polished", "mu": "PSO_results_tuned_rule_mu_polished", "hmu": "PSO_results_tuned_rule_hmu_polished"}
CELLS = {"exp": None, "svc_erlang4": ("erlang", 0.5), "svc_logn2": ("lognormal", 2.0)}
# (design, rule, preemptive)
ARMS = [("fcfs", "fcfs", True), ("fcfs", "mu", True), ("fcfs", "mu", False), ("fcfs", "hmu", True), ("fcfs", "hmu", False),
        ("mu", "mu", True), ("mu", "mu", False), ("hmu", "hmu", True), ("hmu", "hmu", False),
        ("full", "fcfs", True), ("full", "mu", True), ("full", "mu", False), ("full", "hmu", True), ("full", "hmu", False)]


def path_of(design, c):
    folder = DESIGNS["fcfs" if design == "full" else design]
    return glob.glob(os.path.join(ROOT, folder, f"*case_{c:04d}.json"))[0]


def run_arm(task):
    c, design, rule, pre, cell = task
    d = json.load(open(path_of(design, c)))
    lam, mu, h = (np.asarray(d[key], float) for key in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    if design == "full":
        X = np.ones((d["n_servers"], d["m_skills"]), int); S = np.asarray(d["stocks_ff"], int); fixed = float(d["Skills_Costs_ff"] + d["Machine_Costs_ff"])
    else:
        X = np.asarray(d["best_assignment"], int); S = np.asarray(d["stocks"], int); fixed = f * (X.sum(axis=1) > 0).sum() + cs * X.sum()
    pr = priorities(rule, mu, h); R = X * mu[None, :]
    score = None if pr is None else np.where(R > 0, np.round(np.broadcast_to(pr, R.shape), 6), np.inf)
    base = {"instance": c, "design": design, "rule": rule, "preemptive": pre, "cell": cell, "N": d["m_skills"], "M": d["n_servers"], "rho": d.get("utilization_rate"),
            "hv": d.get("holding_costs_variant"), "hmin": d.get("holding_cost_min"), "f": d.get("machine_cost_factor"), "c": d.get("skill_cost_factor")}
    reps, acc, intr, arr = [], None, 0.0, 0.0
    try:
        for i in range(REPS):
            o = sf.simulate(lam, R, score, N_REQ, 0, SEED0 + 1000 * c + 20 * i, MAXQ, service=CELLS[cell], preemptive=pre, arrival=("exponential", 1.0))
            dist = o["dist"]; mass = dist.sum(axis=1)
            if np.any(mass < 1 - MASS):
                return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf), []
            dist = dist / mass[:, None]; e = ebo_at(dist, S); intr += o["events"]; arr += o["arrivals"].sum()
            reps.append({"instance": c, "design": design, "rule": rule, "preemptive": pre, "cell": cell, "replication": i, "total_fixed": float((h * S).sum() + b * e.sum() + fixed)})
            acc = dist if acc is None else acc + dist
    except RuntimeError:
        return dict(base, sim_stable=False, total_fixed=np.inf, total_reopt=np.inf), []
    ehb, S2, E2 = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, acc / REPS)
    tf = np.array([r["total_fixed"] for r in reps])
    return dict(base, sim_stable=True, total_fixed=float(tf.mean()), total_fixed_sem=float(tf.std(ddof=1) / np.sqrt(REPS)), total_reopt=float(ehb + fixed),
                units_reopt=int(S2.sum()), units_fixed=int(S.sum()), interruptions_per_failure=intr / max(arr, 1)), reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cases", type=int, nargs="*"); ap.add_argument("--cells", nargs="*", default=list(CELLS)); a = ap.parse_args()
    cases = a.cases or sorted(case_number(p) for p in glob.glob(os.path.join(ROOT, DESIGNS["fcfs"], "*case_*.json")))
    done, rows = set(), []
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); done = set(zip(old.instance, old.design, old.rule, old.preemptive, old.cell)); rows = old.to_dict("records")
    rep_out = a.out.replace(".csv", "_replications.csv"); rep_rows = pd.read_csv(rep_out).to_dict("records") if os.path.exists(rep_out) else []
    tasks = [(c, dg, r, p, cell) for c in sorted(cases, reverse=True) for cell in a.cells for dg, r, p in ARMS if (c, dg, r, p, cell) not in done]
    print(f"{len(cases)} instances, {len(tasks)} arm-cells to run ({len(done)} done), {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (row, reps) in enumerate(pool.imap_unordered(run_arm, tasks, chunksize=1), 1):
            rows.append(row); rep_rows += reps
            if i % 100 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(a.out, index=False); pd.DataFrame(rep_rows).to_csv(rep_out, index=False)
                print(f"[{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows); print(f"done: {len(df)} arm-cells; stable {df.sim_stable.mean():.1%}")
