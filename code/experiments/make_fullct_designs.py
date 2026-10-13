"""make_fullct_designs.py -- R1: the fully cross-trained shop as design folders, so that the disruption drivers
(sku_surges, multi_surges, run_surge_cases) run on it unchanged.
  ../PSO_results_fullCT/           X = 1, FCFS, stock and costs of the stored final evaluation (stocks_ff, total_cost_ff)
  ../PSO_results_fullCT_rule_mu/   X = 1, mu rule, base-stock levels re-optimized for the rule with the final-evaluation
                                   seeds (1,000,000 + 20 i, 35 x 100,000), as every reported design
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); ROOT = os.path.join(HERE, "..")
from simulation_codes import SimulationInterface_fast as SIf
FINAL_SEED = 1_000_000


def make(p):
    d = json.load(open(p)); M, N = d["n_servers"], d["m_skills"]; X = np.ones((M, N), int)
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs")); b = float(d["penalty_cost"])
    base = {k: d[k] for k in ("caseID", "n_servers", "m_skills", "utilization_rate", "holding_cost_min", "holding_costs_variant", "machine_cost_factor", "skill_cost_factor",
                              "penalty_cost", "machine_cost", "skill_cost", "failure_rates", "service_rates", "holding_costs") if k in d}
    fixed = float(d["Skills_Costs_ff"] + d["Machine_Costs_ff"])
    f = dict(base, best_assignment=X.tolist(), stocks=d["stocks_ff"], total_cost=d["total_cost_ff"], EBO=d["EBO_ff"], utilizationRatesAverage=d["utilizationRatesAverage_ff"],
             rule="fcfs", priorities=None, preemptive=True, design="full cross-training", total_cost_ff=d["total_cost_ff"])
    json.dump(f, open(os.path.join(ROOT, "PSO_results_fullCT", os.path.basename(p)), "w"))
    SIf.priorities = 1.0 / mu; SIf.preemptive = True
    ehb, S, EBO, _, util, _ = SIf.simulation_optimization_run(lam, mu, h, b, X, useDatabase=False, startSeed=FINAL_SEED)
    m = dict(base, best_assignment=X.tolist(), stocks=np.asarray(S, int).tolist(), total_cost=float(ehb + fixed), EBO=np.asarray(EBO, float).tolist(), utilizationRatesAverage=np.asarray(util).tolist(),
             rule="mu", priorities=(1.0 / mu).tolist(), preemptive=True, design="full cross-training", total_cost_ff=d["total_cost_ff"])
    json.dump(m, open(os.path.join(ROOT, "PSO_results_fullCT_rule_mu", os.path.basename(p)), "w"))
    return d["caseID"] if "caseID" in d else p, d["total_cost_ff"], m["total_cost"]


if __name__ == "__main__":
    for o in ("PSO_results_fullCT", "PSO_results_fullCT_rule_mu"): os.makedirs(os.path.join(ROOT, o), exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "PSO_results_tuned_polished", "*case_*.json")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 20) as pool:
        res = pool.map(make, files, chunksize=1)
    r = np.array([(a, b) for _, a, b in res]); print(f"{len(res)} designs; full CT under mu vs FCFS: mean {np.mean(1 - r[:, 1] / r[:, 0]):.2%}, cheaper in {(r[:, 1] < r[:, 0]).sum()}")
