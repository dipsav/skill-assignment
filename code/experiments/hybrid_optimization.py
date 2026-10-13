"""
hybrid_optimization.py -- "BPSO + local search": the tuned BPSO (PSO_optimization_v2) on a share of the evaluation budget,
then the local search of VNS_optimization_v2 (first improvements over single-skill removal, single-skill addition and
removal of a whole technician) on the rest, the whole run capped at one budget of distinct simulated designs.

Budget accounting: both stages go through SimulationInterface_fast with its in-memory cache and SIf.budget, so a design
simulated by the swarm is not simulated again by the local search, and the total never exceeds `budget`. The swarm's
inertia schedule is driven by its share of the budget (completes when its share is spent); the local search starts
from the swarm's best design and stops when no move improves or the budget is spent (then the cheapest simulated
design is returned). The final design is re-estimated with independent seeds, as for the other methods.
"""
import time
import numpy as np

from simulation_codes import SimulationInterface_fast as SIf
import PSO_optimization_v2 as pso
import VNS_optimization_v2 as vns

HYBRID_PARAMS = {"budget": 2500, "split": 0.7, "moves": ("remove", "add", "technician"), "order": "paper",
                 "pso": {"population": 100, "c1": 2.0, "c2": 4.0, "inertia_min": 0.1, "inertia_max": 2.0, "speedRange": 0.0}}


class _Log(list):
    search_total_cost = float("inf")


def optimizeSkillAssignmentHybrid(n_servers, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost,
                                  skill_cost, hybrid_params=None, seed=None, final_seed=1_000_000):
    p = dict(HYBRID_PARAMS, **(hybrid_params or {}))
    lam, mu, h = (np.asarray(a, float) for a in (failure_rates, service_rates, holding_costs))
    b, f, cs = float(penalty_cost), float(machine_cost), float(skill_cost)
    budget = int(p["budget"]); pso_budget = max(1, int(round(p["split"] * budget)))
    t0 = time.time()
    SIf.database = {}; SIf.simulated = 0; SIf.budget = pso_budget
    gen = 10 * max(1, budget // 100)
    params = dict(p["pso"], generation=gen)
    X, _, _, _, plog, util, _ = pso.optimizeSkillAssignmentPSO(int(n_servers), lam, mu, h, b, f, cs, pso_params=params,
                                                            seed=seed, final_seed=final_seed)
    pso_simulated = SIf.simulated; pso_cost = float(plog.search_total_cost)
    SIf.budget = budget
    ev = vns.Evaluator(lam, mu, h, b, f, cs, None, 35, 100_000, use_db=True)
    X = np.ascontiguousarray(X, dtype=np.int64)
    best_X, best_cost = X, pso_cost
    try:
        c0, u0 = ev(X)                                   # cache hit: the swarm already simulated it
        best_X, best_cost, _ = vns.local_search(X, c0, u0, ev, p["order"], h, lam / mu, tuple(p["moves"]))
    except SIf.BudgetSpent:
        pass
    for key, (c, u) in ev.cache.items():                 # cheapest design the local search simulated
        if c < best_cost:
            best_X, best_cost = np.frombuffer(key, dtype=np.int64).reshape(X.shape).copy(), c
    SIf.budget = None
    log = _Log([{"gen": 0, "nevals": pso_simulated, "simulated": pso_simulated, "total": {"min": pso_cost}, "stage": "pso"},
                {"gen": 1, "nevals": SIf.simulated - pso_simulated, "simulated": SIf.simulated, "total": {"min": float(best_cost)}, "stage": "local search"}])
    log.search_total_cost = float(best_cost); log.simulated = SIf.simulated; log.pso_simulated = pso_simulated
    log.pso_cost = pso_cost; log.budget_stopped = SIf.simulated >= budget; log.runtime = time.time() - t0
    ehb, S, EBO, _, util, assign = SIf.simulation_optimization_run(lam, mu, h, b, best_X, useDatabase=False, startSeed=final_seed)
    total = ehb + f * (best_X.sum(axis=1) > 0).sum() + cs * best_X.sum()
    return best_X, total, S, EBO, log, util, assign
