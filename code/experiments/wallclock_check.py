"""wallclock_check.py -- wall-clock time of one BPSO run with parallel particle evaluation (20 workers) against one VNS
run (sequential by construction), same instance, same budget of 2,500 distinct designs, same seeds; costs re-estimated
independently. Output: tables/wallclock.csv. Run on an otherwise idle machine."""
import json, sys, time, numpy as np, pandas as pd
from simulation_codes import SimulationInterface_fast as SIf
import PSO_optimization_v2 as pso, VNS_optimization_v2 as vns
case, workers = int(sys.argv[1]), int(sys.argv[2])
d = json.load(open(f"../PSO_results_rerun/fullRangeResultsPSO_new_2019_case_{case:04d}.json"))
lam, mu, h = (np.array(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs")); b, f, cs, M = d["penalty_cost"], d["machine_cost"], d["skill_cost"], d["n_servers"]
rows = []
for method, w in (("bpso", workers), ("bpso", 1), ("vns", 1)):
    SIf.database = {}; SIf.simulated = 0; SIf.budget = 2500 if method == "bpso" else None; pso.PARALLEL_WORKERS = w
    t0 = time.time()
    if method == "bpso":
        params = {"generation": 250, "population": 100, "c1": 2.0, "c2": 4.0, "inertia_min": 0.1, "inertia_max": 2.0, "speedRange": 0.0}
        X, total, *_ = pso.optimizeSkillAssignmentPSO(M, lam, mu, h, b, f, cs, pso_params=params, seed=100001 + case); n = SIf.simulated
    else:
        X, total, S, E, log, *_ = vns.optimizeSkillAssignmentVNS(M, lam, mu, h, b, f, cs, vns_params={"budget": 2500}, seed=100001 + case); n = log.simulated
    SIf.budget = None
    rows.append({"case": case, "method": method, "workers": w, "designs": n, "minutes": (time.time() - t0) / 60, "total_cost": float(total)})
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"tables/wallclock_case{case:04d}.csv", index=False)
