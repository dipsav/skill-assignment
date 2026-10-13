"""
polish_designs.py -- local-search polish of final designs ("BPSO + local search"): the VNS local search of
VNS_optimization_v2 (best of first improvements over N_remove and N_add, the paper's orders) applied to every design of
a result folder, with the full search evaluation (35 x 100,000 failures, common seeds 10 + 20 i, the repair rule stored
in the file) and an independent re-estimation of the polished design (seeds 1,000,000 + 20 i).

Output: a folder with the same files and keys as the input (so every analysis reads it unchanged); the design, stocks,
costs and utilization are the polished ones; "polish" holds the counts (skills removed/added, designs simulated,
search cost before/after, seconds) and "unpolished_assignment" the input design. The search budget is not limited: a
polish is a descent, it ends when no single removal or addition improves the cost.

    python3 polish_designs.py ../PSO_results_rerun --out ../PSO_results_rerun_polished --workers 20
Finished files are skipped (resumable).
"""
import os, re, sys, glob, json, time, argparse
from multiprocessing import Pool
import numpy as np

FINAL_SEED = 1_000_000


def case_number(path):
    return int(re.search(r"case_(\d+)", os.path.basename(path)).group(1))


def polish_one(task):
    path, out_dir, order = task
    from simulation_codes import SimulationInterface_fast as SIf
    import VNS_optimization_v2 as vns
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    rule = d.get("rule", "fcfs")
    SIf.priorities = None if d.get("priorities") is None else np.asarray(d["priorities"], float)
    SIf.preemptive = bool(d.get("preemptive", True))
    SIf.min_coverage = int(d.get("min_coverage", 1))
    X0 = np.asarray(d["best_assignment"], np.int64)
    t0 = time.time()
    ev = vns.Evaluator(lam, mu, h, b, f, cs, 10**9, 35, 100_000)
    c0, u0 = ev(X0)
    X, c1, _ = vns.local_search(X0, c0, u0, ev, order, h, lam / mu)
    ehb, S, EBO, _, util, assign = SIf.simulation_optimization_run(lam, mu, h, b, X, useDatabase=False, startSeed=FINAL_SEED)
    emp = int((X.sum(axis=1) > 0).sum()); total = ehb + f * emp + cs * X.sum()
    out = dict(d)
    out.update({"best_assignment": X.tolist(), "stocks": [int(s) for s in S], "EBO": [float(e) for e in EBO],
                "Stock_Costs": float((h * S).sum()), "EBO_Costs": float(b * np.sum(EBO)), "Skills_Costs": float(cs * X.sum()),
                "Machine_Costs": float(f * emp), "total_cost": float(total),
                "utilizationRatesAverage": np.asarray(util).tolist(), "skillServerAssignmentStatisitcsAverage": np.asarray(assign).tolist(),
                "search_total_cost": float(c1), "final_design_feasible": bool(SIf.checkSkillAssignment(X, lam, mu)[0]),
                "algorithm": d.get("algorithm", "PSO") + "+LS", "unpolished_assignment": X0.tolist(),
                "unpolished_total_cost": d["total_cost"],
                "polish": {"order": order, "rule": rule, "skills_removed": int(((X0 == 1) & (X == 0)).sum()),
                           "skills_added": int(((X0 == 0) & (X == 1)).sum()), "designs_simulated": ev.simulated,
                           "requests": ev.requested, "search_cost_before": float(c0), "search_cost_after": float(c1),
                           "seconds": time.time() - t0,
                           "evaluation": "35 x 100,000 failures, search seeds 10 + 20 i; polished design re-estimated with seeds 1,000,000 + 20 i"}})
    json.dump(out, open(os.path.join(out_dir, os.path.basename(path)), "w"))
    return c, d["total_cost"], total, out["polish"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--order", choices=["paper", "index"], default="paper"); ap.add_argument("--cases", type=int, nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases:
        files = [p for p in files if case_number(p) in a.cases]
    todo = [p for p in files if not os.path.exists(os.path.join(a.out, os.path.basename(p)))]
    todo.sort(key=lambda p: -os.path.getsize(p))
    print(f"{len(files)} designs, {len(todo)} to polish -> {a.out} | {a.workers} workers | order {a.order}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (c, before, after, pl) in enumerate(pool.imap_unordered(polish_one, [(p, a.out, a.order) for p in todo]), 1):
            print(f"[{i}/{len(todo)}] case {c:4d}: {before:,.0f} -> {after:,.0f} ({after / before - 1:+.2%}, independent seeds) | "
                  f"-{pl['skills_removed']} +{pl['skills_added']} skills | {pl['designs_simulated']} designs | {pl['seconds'] / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
