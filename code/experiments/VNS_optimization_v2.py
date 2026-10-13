"""
VNS_optimization_v2.py -- the basic VNS of Abdelwanis, Gabor, Mladenovic and Sleptchenko (Computers & Operations Research
164, 2024, 106546), function Basic_VNS_based_optimization_intial_solution_ADD_REMOVE of vns/localsearch_1.py, on the
compiled simulator and the guarded interface (SimulationInterface_fast), with the evaluation and reporting of the
revised BPSO and GA (independent re-estimation of the final design).

Algorithm (as in the paper, Algorithms 1 and 4):
  1. initial solution: two MIPs on the capacity constraints -- fewest technicians, fewest skills (max load 0.999);
     the cheaper of the two (simulated) is the start;
  2. local search ("best of first improvements"): the first improving removal of one skill (N_remove) and the first
     improving addition of one skill (N_add) are found; the better of the two is applied; repeat until neither improves;
  3. shake with k = 2, ..., K_max = N x M: flip k distinct random positions of the incumbent, then local search; a better
     result becomes the incumbent and k returns to 2, otherwise k + 1; stop when k > K_max or the budget is spent.
Neighborhood order (order="paper", default, Algorithms 2-3 of the paper): N_remove takes technicians in decreasing
order of utilization and, within a technician, skills in increasing order of the SKU's workload lambda/mu; N_add takes
technicians in increasing order of utilization and, for each, SKUs in decreasing order of holding cost. order="index" scans technician by technician, SKU by SKU, as the received code does.

Changes relative to the received code (each fixes a defect found on 5 Oct 2026, see CLAUDE.md):
  A. the incumbent is never modified in place (the received code's shake and local searches flipped bits of the
     caller's array, so a failed shake corrupted the incumbent and the returned design was not the one whose cost was
     returned: 746,674 vs 661,757 on case 1);
  B. the budget counts DISTINCT SIMULATED DESIGNS and is checked at every evaluation (the received code counted only
     local-search rounds that improved, not shakes or the initial MIPs, so a budget of 300 gave 514 designs);
  C. a shake flips exactly k distinct positions; an infeasible shaken design is redrawn from the incumbent (the received
     code kept flipping further bits of the same design until one was feasible); feasibility costs no simulation;
  D. evaluations go through SimulationInterface_fast: feasibility checked first, distributions up to 2,000 with an
     infinite cost if cut off, common random numbers (seeds 10 + 20 i) for all candidates, the repair rule set by the
     driver (SIf.priorities); the final design is re-estimated from independent seeds;
  E. random numbers are seeded (numpy Generator); the MIPs are solved with scipy/HiGHS (pulp 4 removed the API used).
"""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from simulation_codes import SimulationInterface_fast as SIf

VNS_PARAMS = {"budget": 2500, "kmax": None, "order": "paper", "replications": 35, "requests": 100_000}


class BudgetSpent(Exception):
    pass


class _Log(list):
    """Logbook in the layout the analysis expects: one entry per improvement of the incumbent."""
    search_total_cost = float("inf")


def initial_mip(lam, mu, M, objective):
    """MIP of the received code: alpha (shares), x (skills), y (technicians); sum_j alpha_ij = 1; load of j <= maxRho
    <= 0.999; alpha <= x <= y. objective "servers" (fewest technicians) or "skills" (fewest skills)."""
    N = len(lam); load = lam / mu; na = N * M
    ix = lambda i, j: i * M + j
    nv = 2 * na + M + 1; rows, lo, hi = [], [], []
    for i in range(N):
        r = np.zeros(nv); r[[ix(i, j) for j in range(M)]] = 1; rows.append(r); lo.append(1); hi.append(1)
    for j in range(M):
        r = np.zeros(nv); r[[ix(i, j) for i in range(N)]] = load; r[-1] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
    for i in range(N):
        for j in range(M):
            r = np.zeros(nv); r[ix(i, j)] = 1; r[na + ix(i, j)] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
            r = np.zeros(nv); r[na + ix(i, j)] = 1; r[2 * na + j] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
    c = np.zeros(nv)
    if objective == "servers":
        c[2 * na:2 * na + M] = 1
    else:
        c[na:2 * na] = 1
    integ = np.zeros(nv); integ[na:2 * na + M] = 1
    ub = np.ones(nv); ub[-1] = 0.999
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), ub), options={"time_limit": 60})
    if res.x is None:
        return np.ones((M, N), int)
    return np.round(res.x[na:2 * na]).astype(int).reshape(N, M).T         # technicians x SKUs


class Evaluator:
    """Cost of a design with the search seeds; own cache (cost, stock, utilization), budget in distinct simulations."""
    def __init__(self, lam, mu, h, b, f, cs, budget, reps, requests, use_db=False):
        self.lam, self.mu, self.h, self.b, self.f, self.cs = lam, mu, h, b, f, cs
        self.budget, self.reps, self.requests = budget, reps, requests
        self.use_db = use_db          # True: go through the interface's cache (and its budget, SIf.budget) -- hybrid
        self.cache = {}; self.simulated = 0; self.requested = 0

    def feasible(self, X):
        return bool(SIf.checkSkillAssignment(X, self.lam, self.mu)[0])

    def __call__(self, X):
        X = np.ascontiguousarray(X, dtype=np.int64)
        self.requested += 1
        key = X.tobytes()
        if key in self.cache:
            return self.cache[key]
        if not self.feasible(X):
            self.cache[key] = (np.inf, None); return self.cache[key]
        if self.budget is not None and self.simulated >= self.budget:
            raise BudgetSpent
        ehb, S, EBO, _, util, _ = SIf.simulation_optimization_run(self.lam, self.mu, self.h, self.b, X, useDatabase=self.use_db,
                                                                  stopNrRequests=self.requests, replications=self.reps)
        self.simulated += 1
        total = ehb + self.f * (X.sum(axis=1) > 0).sum() + self.cs * X.sum()
        self.cache[key] = (float(total), None if util is None else np.asarray(util))
        return self.cache[key]


def _remove_order(X, util, load, order):
    """Algorithm 2 of the paper: technicians in decreasing order of utilization; within a technician, skills in
    increasing order of the SKU's workload lambda/mu. Utilization is the simulated one (the paper uses the LP shares)."""
    pos = [tuple(p) for p in np.argwhere(X == 1)]
    if order == "paper" and util is not None:
        u = util.sum(axis=1)
        pos.sort(key=lambda p: (-u[p[0]], load[p[1]]))
    return pos


def _add_order(X, util, h, order):
    """Algorithm 3: technicians in ascending order of utilization; skills in descending order of holding cost (the
    paper's text; its Algorithm 3 says increasing -- the two contradict each other, descending is the sensible one)."""
    if order == "paper" and util is not None:
        techs = np.argsort(util.sum(axis=1), kind="stable")      # least utilized technician first
        skus = np.argsort(-h, kind="stable")                     # highest holding cost first
        return [(j, i) for j in techs for i in skus if X[j, i] == 0]
    return [tuple(p) for p in np.argwhere(X == 0)]


def _technician_order(X, util, order):
    """Employed technicians, least utilized first (the paper has no such move; used by the hybrid/polish)."""
    emp = np.flatnonzero(X.sum(axis=1) > 0)
    if order == "paper" and util is not None:
        emp = emp[np.argsort(util.sum(axis=1)[emp], kind="stable")]
    return [int(j) for j in emp]


def local_search(X, cost, util, ev, order, h, load, moves=("remove", "add")):
    """Best of first improvements over the neighborhoods in `moves`, repeated until none improves. Returns a NEW array.
    moves: "remove" (one skill), "add" (one skill), "technician" (all skills of one technician -- crosses the plateau
    where removing skills one by one raises the cost before the technician's cost disappears)."""
    X = X.copy()
    while True:
        best = None
        for nb in moves:
            cand = (_remove_order(X, util, load, order) if nb == "remove" else _add_order(X, util, h, order) if nb == "add"
                    else _technician_order(X, util, order))
            for p in cand:
                Y = X.copy()
                if nb == "technician":
                    Y[p, :] = 0
                else:
                    Y[p] = 1 - Y[p]
                cy, uy = ev(Y)
                if cy < cost:
                    if best is None or cy < best[1]:
                        best = (Y, cy, uy)
                    break                                        # first improvement within this neighborhood
        if best is None:
            return X, cost, util
        X, cost, util = best


def optimizeSkillAssignmentVNS(n_servers, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost,
                               skill_cost, vns_params=None, seed=None, final_seed=1_000_000, verbose=False):
    """Same interface and outputs as PSO_optimization_v2.optimizeSkillAssignmentPSO:
    X, total (independent seeds), S, EBO, logbook (with .search_total_cost), utilization, assignment statistics."""
    p = dict(VNS_PARAMS, **(vns_params or {}))
    lam, mu, h = (np.asarray(a, float) for a in (failure_rates, service_rates, holding_costs))
    M, N = int(n_servers), len(lam)
    kmax = p["kmax"] or N * M
    rng = np.random.default_rng(seed)
    ev = Evaluator(lam, mu, h, float(penalty_cost), float(machine_cost), float(skill_cost), p["budget"], p["replications"], p["requests"])
    log = _Log(); t0 = time.time(); last = 0

    def record(cost):
        nonlocal last
        log.append({"gen": len(log), "nevals": ev.simulated - last, "simulated": ev.simulated, "requested": ev.requested,
                    "total": {"min": float(cost)}, "time": time.time() - t0})
        last = ev.simulated

    best_X, best_cost, best_util = None, np.inf, None
    try:
        for obj in ("servers", "skills"):
            X0 = initial_mip(lam, mu, M, obj)
            c0, u0 = ev(X0)
            if c0 < best_cost:
                best_X, best_cost, best_util = X0, c0, u0
        if best_X is None:                                       # both MIP designs unusable: start from full cross-training
            best_X = np.ones((M, N), np.int64); best_cost, best_util = ev(best_X)
        best_X = np.ascontiguousarray(best_X, dtype=np.int64)
        record(best_cost)
        best_X, best_cost, best_util = local_search(best_X, best_cost, best_util, ev, p["order"], h, lam / mu)
        record(best_cost)
        k = 2
        while k <= kmax:
            for _ in range(100):                                 # shake: k distinct flips of the incumbent, feasible
                Y = best_X.copy().ravel()
                idx = rng.choice(Y.size, size=min(k, Y.size), replace=False)
                Y[idx] = 1 - Y[idx]; Y = Y.reshape(best_X.shape)
                if ev.feasible(Y):
                    break
            else:
                k += 1; continue
            cy, uy = ev(Y)
            Z, cz, uz = local_search(Y, cy, uy, ev, p["order"], h, lam / mu)
            if cz < best_cost:
                best_X, best_cost, best_util = Z, cz, uz; k = 2; record(best_cost)
            else:
                k += 1
    except BudgetSpent:
        pass
    # The final design is the cheapest design simulated (a local search cut off by the budget may have passed the
    # incumbent; the incumbent itself is in the cache, so this never makes the result worse).
    for key, (c, u) in ev.cache.items():
        if c < best_cost:
            best_X, best_cost, best_util = np.frombuffer(key, dtype=best_X.dtype).reshape(best_X.shape).copy(), c, u
    record(best_cost)
    log.search_total_cost = float(best_cost)
    log.simulated, log.requested, log.runtime = ev.simulated, ev.requested, time.time() - t0
    # independent re-estimation of the final design (as the revised BPSO and GA)
    ehb, S, EBO, _, util, assign = SIf.simulation_optimization_run(lam, mu, h, float(penalty_cost), best_X, useDatabase=False,
                                                                   stopNrRequests=p["requests"], replications=p["replications"],
                                                                   startSeed=final_seed)
    total = ehb + float(machine_cost) * (best_X.sum(axis=1) > 0).sum() + float(skill_cost) * best_X.sum()
    if verbose:
        print(f"VNS: {ev.simulated} designs simulated ({ev.requested} requests), search cost {best_cost:,.0f}, "
              f"independent {total:,.0f}, {time.time() - t0:.0f} s")
    return best_X, total, S, EBO, log, util, assign
