"""
prune_check.py -- do the final designs carry redundant skills?  (diagnostic for Section 5.1; not part of either algorithm)

For a final design X (BPSO or GA result file): repeatedly try removing one skill; keep the removal if the design stays
feasible and its cost (stock re-optimized, technicians and skills included) falls; stop after a full pass without a
gain (first improvement, skills tried in a fixed order: technician by technician, SKU by SKU).
  - screening: 10 replications x 100,000 failures with common seeds 3,000,000 + 20 i (not used for any reported cost);
  - the pruned design is then re-estimated with the reported independent seeds (35 x 100,000, 1,000,000 + 20 i), so
    its cost compares directly with the stored total_cost of the unpruned design.

    python3 prune_check.py ../PSO_results_rerun ../GA_results_rerun --out tables/prune_check.csv --workers 2
"""
import os, re, sys, glob, json, time, argparse
from multiprocessing import Pool
import numpy as np

SCREEN_SEED, SCREEN_REPS, FINAL_SEED, FINAL_REPS = 3_000_000, 10, 1_000_000, 35


def cost(SIf, X, lam, mu, h, b, f, cs, reps, seed):
    ehb = SIf.simulation_optimization_run(lam, mu, h, b, X, useDatabase=False, replications=reps, startSeed=seed)[0]
    return ehb + f * (X.sum(axis=1) > 0).sum() + cs * X.sum()


def prune(task):
    path, label = task
    from simulation_codes import SimulationInterface_fast as SIf
    d = json.load(open(path)); c = int(re.search(r"case_(\d+)", path).group(1))
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    X = np.asarray(d["best_assignment"], int).copy()
    t0 = time.time(); evals = 0
    best = cost(SIf, X, lam, mu, h, b, f, cs, SCREEN_REPS, SCREEN_SEED); start = best
    removed = []
    improved = True
    while improved:
        improved = False
        for j, i in np.argwhere(X == 1):
            Y = X.copy(); Y[j, i] = 0
            if not SIf.checkSkillAssignment(Y, lam, mu)[0]:
                continue
            evals += 1
            cy = cost(SIf, Y, lam, mu, h, b, f, cs, SCREEN_REPS, SCREEN_SEED)
            if cy < best:
                X, best, improved = Y, cy, True
                removed.append((int(j), int(i)))
    final = cost(SIf, X, lam, mu, h, b, f, cs, FINAL_REPS, FINAL_SEED) if removed else float(d["total_cost"])
    return {"case": c, "algorithm": label, "c_factor": d.get("skill_cost_factor"), "N": d["m_skills"], "M": d["n_servers"],
            "skills_before": int(np.sum(d["best_assignment"])), "skills_removed": len(removed),
            "technicians_freed": int((np.asarray(d["best_assignment"]).sum(1) > 0).sum() - (X.sum(1) > 0).sum()),
            "total_reported": float(d["total_cost"]), "total_pruned": float(final),
            "gain": 1 - final / float(d["total_cost"]), "screen_gain": 1 - best / start,
            "evaluations": evals, "seconds": time.time() - t0, "pruned_assignment": X.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pso"); ap.add_argument("ga"); ap.add_argument("--out", default="tables/prune_check.csv")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--cases", type=int, nargs="*")
    a = ap.parse_args()
    import pandas as pd
    tasks = [(p, lab) for folder, lab in ((a.pso, "BPSO"), (a.ga, "GA"))
             for p in sorted(glob.glob(os.path.join(folder, "*case_*.json")))
             if not a.cases or int(re.search(r"case_(\d+)", p).group(1)) in a.cases]
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); done = set(zip(old.case, old.algorithm)); rows = old.to_dict("records")
    else:
        rows = []
    tasks = [t for t in tasks if (int(re.search(r"case_(\d+)", t[0]).group(1)), t[1]) not in done]
    print(f"{len(tasks)} designs to check, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(prune, tasks):
            rows.append(r)
            print(f"case {r['case']:4d} {r['algorithm']:4s}: {r['skills_removed']} of {r['skills_before']} skills removed, "
                  f"cost {r['gain']:+.2%} (independent seeds), {r['evaluations']} evaluations, {r['seconds']:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)        # written after every design: resumable
