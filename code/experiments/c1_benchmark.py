"""
c1_benchmark.py -- benchmark C1 (Section 5.1, Appendix C; R4.2, R7.9, R9.7): four searches at the same budget of DISTINCT
simulated designs, repeated with several seeds, on a 32-instance fraction of the test bed.

Algorithms
  pso      proposed BPSO (Section 4.1), tuned (T1/T2): increasing inertia 0.1 -> 2.0, c1 = 2, c2 = 4, 100 particles;
           (the published setting 0.01 -> 1.5, c1 = 1, c2 = 3 can be run as a label with --settings)
  pso_orig original BPSO of El-Maleh et al. (2013) as used in Sleptchenko et al. (ICACI 2017): decreasing inertia
           0.9 -> 0.01, c1 = c2 = 2, 100 particles (best setting of the tuning study T1; ICACI 2017 reports no single set);
  ga       GA of Sleptchenko et al. (2019) as published (population 100, crossover 0.4, mutation 0.1, bit flip 0.1,
           tournament 10, full cross-training in the initial population);
  vns      basic VNS of Abdelwanis et al. (2024), VNS_optimization_v2;
  hybrid   tuned BPSO on 70% of the budget, then the local search (skill removal/addition, technician removal) on the
           rest, capped at the same total (hybrid_optimization.py).
Budget: B distinct simulated designs (SIf.budget for pso/pso_orig/ga, the VNS's own counter). Under a budget the PSO's
inertia schedule is driven by the share of the budget used (not by generations), so it completes exactly when the budget
is spent; PSO and GA get enough generations for the budget to bind. All candidates share the search seeds 10 + 20 i (common random numbers); every final design is re-estimated
with the independent seeds 1,000,000 + 20 i, as in the main runs.
Instances: the 2^(7-2) fraction with generators F = ABCD, G = ABDE (A = N, B = M, C = rho, D = h_min,
E = holding-cost variant, F = technician cost, G = skill cost; levels coded -1/+1): all main effects estimable.
Seeds: run s of instance c uses seed 100,000 s + c.

    python3 c1_benchmark.py ../PSO_results_rerun --out ../C1_results --budget 1000 --seeds 1 2 3 4 5 --workers 20
    python3 c1_benchmark.py ../PSO_results_rerun --list          (print the 32 instances)
One JSON per (algorithm, instance, seed); finished runs are skipped (resumable).
"""
import os, re, sys, glob, json, time, argparse
from multiprocessing import Pool
import numpy as np

FINAL_SEED = 1_000_000
ALGS = ("pso", "pso_orig", "ga", "vns", "hybrid")


def case_number(path):
    return int(re.search(r"case_(\d+)", os.path.basename(path)).group(1))


def fraction(folder):
    """The 32 instances with F = ABCD and G = ABDE (coded levels)."""
    keys = ("m_skills", "n_servers", "utilization_rate", "holding_cost_min", "holding_costs_variant", "machine_cost_factor", "skill_cost_factor")
    files = sorted(glob.glob(os.path.join(folder, "*case_*.json")), key=case_number)
    rows = [(p, [json.load(open(p))[k] for k in keys]) for p in files]
    lows = [min(r[1][i] for r in rows) for i in range(7)]
    out = []
    for p, v in rows:
        A, B, C, D, E, F, G = (1 if v[i] != lows[i] else -1 for i in range(7))
        if F == A * B * C * D and G == A * B * D * E:
            out.append(p)
    return out


# Settings per algorithm label: a label is "pso", "pso_orig", "ga", "vns", optionally followed by ":<name>" with its own
# parameter overrides (T2 transfer check), e.g. {"pso:c2_4": {"c1": 2.0, "c2": 4.0, "inertia_min": 0.1, "inertia_max": 1.0}}.
OVERRIDES = {}


def base_alg(label):
    return label.split(":")[0]


def run(task):
    path, label, seed, budget, out_dir = task
    alg = base_alg(label)
    target = os.path.join(out_dir, label.replace(":", "_"), f"c1_{label.replace(':', '_')}_case_{case_number(path):04d}_seed_{seed}.json")
    if os.path.exists(target):
        return None
    from simulation_codes import SimulationInterface_fast as SIf
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    b, f, cs, M = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"]), int(d["n_servers"])
    SIf.database = {}; SIf.priorities = None; SIf.simulated = 0
    SIf.budget = None if alg == "vns" else budget
    s = 100_000 * seed + c
    t0 = time.time()
    if alg in ("pso", "pso_orig"):
        import PSO_optimization_v2 as pso
        gen = 10 * max(1, budget // 100)        # enough generations for the budget to bind; the inertia schedule follows the budget
        params = ({"generation": gen, "population": 100, "c1": 2.0, "c2": 4.0, "inertia_min": 0.1, "inertia_max": 2.0, "speedRange": 0.0}
                  if alg == "pso" else
                  {"generation": gen, "population": 100, "c1": 2.0, "c2": 2.0, "inertia_min": 0.01, "inertia_max": 0.9, "speedRange": 0.0,
                   "variant": "original"})
        params.update(OVERRIDES.get(label, {}))
        X, total, S, EBO, log, util, assign = pso.optimizeSkillAssignmentPSO(M, lam, mu, h, b, f, cs, pso_params=params, seed=s, final_seed=FINAL_SEED)
        trace, simulated, stopped = pso.TRACK.get("trace", []), SIf.simulated, log.budget_stopped
        extra = {"last_inertia": log.last_inertia, "generations_run": len(log)}
    elif alg == "ga":
        import GA_Optimization_v2 as ga
        params = {"population": 100, "generation": 10 * max(1, budget // 100), "crossover": 0.4, "mutation": 0.1, "indpb": 0.1, "tournsize": 10}
        params.update(OVERRIDES.get(label, {}))
        X, total, S, EBO, log, util, assign = ga.optimizeSkillAssignment(M, lam, mu, h, b, f, cs, ga_params=params, seed=s, final_seed=FINAL_SEED)
        trace, simulated, stopped = ga.TRACK.get("trace", []), SIf.simulated, log.budget_stopped
        extra = {}
    elif alg == "hybrid":
        import hybrid_optimization as hy
        params = dict({"budget": budget, "split": 0.7}, **OVERRIDES.get(label, {}))
        X, total, S, EBO, log, util, assign = hy.optimizeSkillAssignmentHybrid(M, lam, mu, h, b, f, cs, hybrid_params=params, seed=s, final_seed=FINAL_SEED)
        trace = [(e["simulated"], e["total"]["min"]) for e in log]; simulated, stopped = log.simulated, log.budget_stopped
        extra = {"pso_designs": log.pso_simulated, "pso_search_cost": log.pso_cost}
    else:
        import VNS_optimization_v2 as vns
        params = dict({"budget": budget}, **OVERRIDES.get(label, {}))
        X, total, S, EBO, log, util, assign = vns.optimizeSkillAssignmentVNS(M, lam, mu, h, b, f, cs, vns_params=params, seed=s, final_seed=FINAL_SEED)
        trace = [(e["simulated"], e["total"]["min"]) for e in log]; simulated, stopped = log.simulated, log.simulated >= budget
        extra = {}
    SIf.budget = None
    X = np.asarray(X, int)
    out = {"algorithm": alg, "label": label, "case": c, "seed": seed, "optimizer_seed": s, "budget": budget, "params": params,
           "factors": {k: d.get(k) for k in ("m_skills", "n_servers", "utilization_rate", "holding_cost_min", "holding_costs_variant",
                                             "machine_cost_factor", "skill_cost_factor")},
           "best_assignment": X.tolist(), "stocks": [int(v) for v in S], "total_cost": float(total),
           "search_total_cost": float(log.search_total_cost), "designs_simulated": int(simulated), "budget_stopped": bool(stopped),
           "trace": [[int(a), float(v)] for a, v in trace], "skills": int(X.sum()), "technicians": int((X.sum(axis=1) > 0).sum()),
           "feasible": bool(SIf.checkSkillAssignment(X, lam, mu)[0]), "runtime": time.time() - t0, **extra,
           "evaluation": "search: 35 x 100,000, seeds 10 + 20 i; total_cost: independent seeds 1,000,000 + 20 i"}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, target)
    return alg, c, seed, out["total_cost"], simulated, out["runtime"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", default="../C1_results"); ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5]); ap.add_argument("--algs", nargs="+", default=list(ALGS))
    ap.add_argument("--cases", type=int, nargs="*"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--list", action="store_true")
    ap.add_argument("--settings", help="JSON file {label: parameter overrides}; its labels are added to --algs (T2)")
    a = ap.parse_args()
    if a.settings:
        OVERRIDES.update(json.load(open(a.settings)))
        a.algs = list(dict.fromkeys(list(a.algs) + list(OVERRIDES)))
    files = fraction(a.folder)
    if a.list:
        print(len(files), "instances:", [case_number(p) for p in files]); sys.exit()
    if a.cases:
        files = [p for p in files if case_number(p) in a.cases]
    tasks = [(p, lab, s, a.budget, a.out) for s in a.seeds for p in files for lab in a.algs]
    tasks.sort(key=lambda t: -json.load(open(t[0]))["m_skills"] * json.load(open(t[0]))["n_servers"])   # large first
    print(f"{len(files)} instances x {len(a.algs)} algorithms x {len(a.seeds)} seeds = {len(tasks)} runs, budget {a.budget}, "
          f"{a.workers} workers -> {a.out}", flush=True)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, tasks), 1):
            if r:
                alg, c, s, tot, sim, rt = r
                print(f"[{i}/{len(tasks)}] {alg:8s} case {c:4d} seed {s}: {tot:,.0f} | {sim} designs | {rt / 60:.1f} min", flush=True)
