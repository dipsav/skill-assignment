"""
rule_comparison.py -- run A1: every design of a result folder evaluated under FCFS, the mu rule and the h*mu rule
(Section 5.3; and the "sequential" reference for the value of choosing the rule jointly, Section 5.x).

Per design and rule: 35 replications x 100,000 failures, independent seeds 1,000,000 + 20 i (the seeds of the stored
evaluations, so a design under its own rule reproduces its stored total exactly, and the rules are compared with common
random numbers). Two stock policies from the same simulation:
  reoptimized -- base-stock levels re-optimized for the rule (EHB(X, pi) as defined in the paper);
  fixed       -- the design's stored stock levels kept (D3-style: the rule changes, the stock does not).
Also per technician: utilization and its share of the repairs started (sum over SKUs; above 1 with preemptions).

    python3 rule_comparison.py ../PSO_results_rerun --out tables/R1_costs.csv --workers 2
Output rows: instance, factors, rule, stock policy, total, stock cost, backorder cost, skills cost, technicians cost,
busiest technician load, utilization spread; plus tables/R1_util.csv (instance, rule, technician, utilization).
"""
import os, re, sys, glob, json, time, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

FINAL_SEED, REPS, N_REQ, MAXQ = 1_000_000, 35, 100_000, 2000
RULES = ("fcfs", "mu", "hmu")


def case_number(path):
    return int(re.search(r"case_(\d+)", os.path.basename(path)).group(1))


def priorities(rule, mu, h):
    return {"fcfs": None, "mu": 1.0 / mu, "hmu": 1.0 / (h * mu)}[rule]


def ebo_at(dist, S):
    n = np.arange(dist.shape[1])
    return np.array([np.sum(dist[k] * np.maximum(n - S[k], 0)) for k in range(len(S))])


def evaluate(task):
    path, rules = task
    from simulation_codes import simulation_fast as sf
    from simulation_codes import SimulationInterface_fast as SIf
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"])
    X = np.asarray(d["best_assignment"], float); S0 = np.asarray(d["stocks"], int)
    fixed = f * (X.sum(axis=1) > 0).sum() + cs * X.sum()
    factors = {k: d.get(k) for k in ("m_skills", "n_servers", "utilization_rate", "holding_cost_min", "holding_costs_variant",
                                     "machine_cost_factor", "skill_cost_factor")}
    rows, urows = [], []
    for rule in rules:
        pr = priorities(rule, mu, h)
        dist, _, util, starts, _ = sf.simulation_batch_run(lam, mu, X, priorities=pr, expectedTotalNumberRequests=N_REQ,
                                                          numberWarmingUpRequests=0, replications=REPS, maxQueue=MAXQ, startSeed=FINAL_SEED)
        ehb, S, EBO = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, dist)
        for policy, SS, E in (("reoptimized", S, EBO), ("fixed", S0, ebo_at(dist, S0))):
            rows.append({"instance": c, **factors, "design_rule": d.get("rule", "fcfs"), "rule": rule, "stock": policy,
                         "total": float((h * SS).sum() + b * E.sum() + fixed), "stock_cost": float((h * SS).sum()),
                         "backorder_cost": float(b * E.sum()), "skills_cost": float(cs * X.sum()),
                         "technicians_cost": float(f * (X.sum(axis=1) > 0).sum()), "units_in_stock": int(SS.sum()),
                         "max_load": float(util.sum(axis=1).max()),
                         "load_spread": float(np.ptp(util.sum(axis=1)[X.sum(axis=1) > 0])),
                         "stored_total": d["total_cost"] if (rule == d.get("rule", "fcfs") and policy == "reoptimized") else None})
        for j in range(X.shape[0]):
            if X[j].sum() > 0:
                urows.append({"instance": c, "rule": rule, "technician": j, "utilization": float(util[j].sum()),
                              "assignment_share": float(starts[j].sum())})
    return rows, urows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", default="tables/R1_costs.csv"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rules", nargs="+", default=list(RULES)); ap.add_argument("--cases", type=int, nargs="*")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, "*case_*.json")), key=case_number)
    if a.cases:
        files = [p for p in files if case_number(p) in a.cases]
    print(f"{len(files)} designs x {len(a.rules)} rules, {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time(); allrows, allu = [], []
    with Pool(a.workers) as pool:
        for i, (rows, urows) in enumerate(pool.imap_unordered(evaluate, [(p, a.rules) for p in files]), 1):
            allrows += rows; allu += urows
            r = {x["rule"]: x["total"] for x in rows if x["stock"] == "reoptimized"}
            base = r.get(rows[0]["design_rule"], r["fcfs"])
            print(f"[{i}/{len(files)}] case {rows[0]['instance']:4d}: " + " | ".join(f"{k} {v / base - 1:+.1%}" for k, v in r.items()), flush=True)
    df = pd.DataFrame(allrows); df.to_csv(a.out, index=False)
    pd.DataFrame(allu).to_csv(a.out.replace("costs", "util"), index=False)
    chk = df.dropna(subset=["stored_total"])
    print(f"stored totals reproduced: max |diff| {np.abs(chk.total / chk.stored_total - 1).max():.1e} over {len(chk)} designs | {(time.time() - t0) / 60:.1f} min")
