"""
test_vns_v2.py -- checks of VNS_optimization_v2.

  1. Consistency: the returned search cost is the cost of the returned design (re-evaluated on the search seeds).
  2. Budget: never more distinct simulated designs than the budget.
  3. Reproducibility: same seed, same design.
  4. Optimality on small instances (3 SKUs x 3 technicians, 512 designs): brute-force enumeration on the SAME seeds
     (so the objective is the same deterministic function), 12 random instances in the test-bed style, both orders.
Short runs (5 x 20,000 failures) keep it to a few minutes on one core.
"""
import itertools, sys, time
import numpy as np
import VNS_optimization_v2 as vns
from simulation_codes import SimulationInterface_fast as SIf

SHORT = {"replications": 5, "requests": 20_000}
ok = []


def report(name, passed, detail):
    ok.append(passed); print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


def instance(rng, N, M, rho, c_factor=0.1):
    lam = rng.dirichlet(np.ones(N)); delta = rng.dirichlet(np.ones(N))
    mu = lam / (delta * rho * M)
    h = rng.uniform(1, 100, N); b = 50 * np.sum(lam * h) / lam.sum(); f = 10 * 100.0
    return lam, mu, h, b, f, c_factor * f


def cost(X, lam, mu, h, b, f, cs):
    ev = vns.Evaluator(lam, mu, h, b, f, cs, 10**9, SHORT["replications"], SHORT["requests"])
    return ev(np.asarray(X, int))[0]


# 1-3 on a test-bed case
import json
d = json.load(open("../largeCases/fullRangeResultsPSO_new_2019_case_0001.json"))
args = (d["n_servers"], d["failure_rates"], d["service_rates"], d["holding_costs"], d["penalty_cost"], d["machine_cost"], d["skill_cost"])
lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
t = time.time()
X1, tot1, S1, E1, log1, *_ = vns.optimizeSkillAssignmentVNS(*args, vns_params=dict(SHORT, budget=300), seed=5)
dt = time.time() - t
c_re = cost(X1, lam, mu, h, d["penalty_cost"], d["machine_cost"], d["skill_cost"])
report("1. returned cost = cost of the returned design", abs(c_re - log1.search_total_cost) < 1e-6 * c_re,
       f"search cost {log1.search_total_cost:,.0f}, re-evaluated {c_re:,.0f}, independent {tot1:,.0f}")
report("2. budget respected", log1.simulated <= 300, f"{log1.simulated} designs simulated, {log1.requested} requests, budget 300, {dt:.0f} s")
X2, *_ = vns.optimizeSkillAssignmentVNS(*args, vns_params=dict(SHORT, budget=300), seed=5)
report("3. reproducible", np.array_equal(X1, X2), "same seed, same design")

# 4. brute force on small instances
rng = np.random.default_rng(2026)
found = {"paper": 0, "index": 0}; gaps = {"paper": [], "index": []}; n_inst = 12
for r in range(n_inst):
    lam, mu, h, b, f, cs = instance(rng, 3, 3, rho=rng.choice([0.6, 0.8]), c_factor=rng.choice([0.01, 0.1]))
    best = np.inf
    for bits in itertools.product((0, 1), repeat=9):
        X = np.array(bits).reshape(3, 3)
        if X.sum(0).min() == 0:
            continue
        best = min(best, cost(X, lam, mu, h, b, f, cs))
    for order in ("paper", "index"):
        Xv, _, _, _, lg, *_ = vns.optimizeSkillAssignmentVNS(3, lam, mu, h, b, f, cs, vns_params=dict(SHORT, budget=200, order=order), seed=r)
        g = lg.search_total_cost / best - 1
        gaps[order].append(g); found[order] += g < 1e-9
for order in ("paper", "index"):
    report(f"4. optimum found on 3 x 3 instances, order {order}", found[order] >= n_inst - 1,
           f"{found[order]} of {n_inst}; largest gap {max(gaps[order]):.2%}")

print(f"\n{sum(ok)}/{len(ok)} checks passed"); sys.exit(0 if all(ok) else 1)
