"""
rerun_pso_cases.py -- re-optimize selected cases with the revised PSO (PSO_optimization_v2.py) or, with --algorithm ga,
the revised GA (GA_Optimization_v2.py); both run on the compiled simulator and the guarded interface and report the final
design's cost from independent seeds. The instances are read from the original PSO result files.

Usage (from the folder that contains PSO_optimization_v2.py and simulation_codes/):
    python rerun_pso_cases.py <folder with the original result files> --cases 1,4,10 [--out PSO_results_rerun]
    python rerun_pso_cases.py <folder> --from-csv compare_pso.csv            # every case marked infeasible there
    python rerun_pso_cases.py <folder> --from-csv compare_pso.csv --include-truncated   # ... and feasible designs whose
                                                                             # distribution reached the 500 cut-off
    python rerun_pso_cases.py <folder> --all                                 # every case in the folder
Options: --seed-base 1000 (case k uses seed 1000 + k), --workers 4 (cases in parallel), --test (tiny swarm, quick check)

What it fixes relative to the 2019 runs:
  - evaluations go through SimulationInterface_fast: feasibility checked BEFORE the cache lookup, distributions up to
    2000 with an infinite cost if one is cut off, compiled simulator (about 30 times faster);
  - a fresh database per case (db_rerun_case_XXXX.dict): no records from other code or settings, no clashes between
    parallel runs;
  - the optimizer's random numbers are seeded (case k: seed_base + k);
  - the reported cost of the final design (and of full cross-training) comes from an INDEPENDENT evaluation
    (35 replications, seeds 1,000,000 + 20 i), not from the seeds that selected it; the search cost is kept as well;
  - the stored utilization belongs to the final design (full cross-training's is stored separately).
The output files have the same keys as the originals plus pso_params, seed, simulator and the search cost.
"""
import os, re, csv, sys, glob, json, time, random, argparse
import numpy as np
from multiprocessing import Pool


# PSO parameters. PUBLISHED (Appendix A of the submitted version; used for PSO_results_rerun and PSO_results_rerun_rule_mu):
#   c1 1.0, c2 3.0, inertia 0.01 -> 1.5.
# TUNED (study T1/T2, 6-7 Oct 2026: brute-force instances, then the transfer check on 8 test-bed instances at budget
# 2,500 -- 4.4% cheaper designs than the published setting in 16 of 16 runs): c1 2.0, c2 4.0, inertia 0.1 -> 2.0, linear.
PSO_PARAMS_PUBLISHED = {"generation": 25, "population": 100, "c1": 1.0, "c2": 3.0, "inertia_min": 0.01, "inertia_max": 1.5, "speedRange": 0.0}
PSO_PARAMS = {"generation": 25, "population": 100, "c1": 2.0, "c2": 4.0, "inertia_min": 0.1, "inertia_max": 2.0, "speedRange": 0.0}
TEST_PARAMS = dict(PSO_PARAMS, generation=2, population=6)
# GA parameters (the defaults of GA_Optimization.py) -- confirm they are the ones used for the paper's GA results.
GA_PARAMS = {"population": 100, "generation": 50, "crossover": 0.4, "mutation": 0.1, "indpb": 0.1, "tournsize": 10}
GA_TEST_PARAMS = dict(GA_PARAMS, population=8, generation=2)
# VNS (Abdelwanis et al. 2024): budget = distinct simulated designs (the paper's 2,500 iterations), K_max = N x M.
VNS_PARAMS = {"budget": 2500, "kmax": None, "order": "paper", "replications": 35, "requests": 100_000}
VNS_TEST_PARAMS = dict(VNS_PARAMS, budget=40)
FINAL_SEED, REPS, N_REQ, MAXQ = 1_000_000, 35, 100_000, 2000
# The cache of simulated designs lives in memory (one case per worker); on disk it took up to about 0.5 GB per case and
# filled the disk. --disk-cache restores the SQLite files (db_rerun_case_XXXX.dict in the output folder).
DISK_CACHE = False


def log_records(logbook):
    """The search log with its statistics: DEAP keeps MultiStatistics (total, stock_cost, ...) in logbook.chapters,
    so dict(entry) alone loses them. Same layout as the 2019 result files: each entry carries its chapters."""
    out = []
    chapters = getattr(logbook, "chapters", {}) or {}
    for i, rec in enumerate(logbook):
        r = dict(rec)
        for name, book in chapters.items():
            if i < len(book): r[name] = dict(book[i])
        out.append(r)
    return out


def case_number(path):
    m = re.search(r"case_(\d+)", os.path.basename(path)); return int(m.group(1)) if m else 10**9


def rule_priorities(rule, mu, h):
    """Per-SKU priorities, lower = higher priority: FCFS none; mu rule 1/mu (faster repairs first); h*mu rule 1/(h mu)."""
    return {"fcfs": None, "mu": 1.0 / mu, "hmu": 1.0 / (h * mu)}[rule]


def _rerun_case(args):
    path, out_dir, seed, params, algorithm, rule, preemptive, coverage = args
    from simulation_codes import SimulationInterface_fast as SIf
    d = json.load(open(path)); c = case_number(path)
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    SIf.priorities = rule_priorities(rule, mu, h)       # the rule applies to the search AND to both final evaluations
    SIf.preemptive = preemptive                          # A3: non-preemptive priorities inside the search and the evaluations
    SIf.min_coverage = coverage                          # B6b: every SKU repaired by at least this many technicians
    b, f, cs, M = float(d["penalty_cost"]), float(d["machine_cost"]), float(d["skill_cost"]), int(d["n_servers"])
    if DISK_CACHE:
        SIf.initDatabase(os.path.join(out_dir, f"db_rerun_case_{c:04d}"), createNewDatabase=True)
    else:
        SIf.database = {}          # in-memory cache for this case only: no disk use, freed when the next case starts
    t0 = time.time()
    if algorithm == "vns":
        import VNS_optimization_v2 as vns
        X, total, S, EBO, logbook, util, assign = vns.optimizeSkillAssignmentVNS(M, lam, mu, h, b, f, cs, vns_params=params,
                                                                                seed=seed, final_seed=FINAL_SEED)
    elif algorithm == "ga":
        import GA_Optimization_v2 as ga
        X, total, S, EBO, logbook, util, assign = ga.optimizeSkillAssignment(M, lam, mu, h, b, f, cs, ga_params=params,
                                                                             seed=seed, final_seed=FINAL_SEED)
    else:
        import PSO_optimization_v2 as pso
        X, total, S, EBO, logbook, util, assign = pso.optimizeSkillAssignmentPSO(M, lam, mu, h, b, f, cs, caseID=d.get("caseID", ""),
                                                                                pso_params=params, seed=seed, final_seed=FINAL_SEED)
    runtime = time.time() - t0
    X = np.asarray(X, int); feasible = SIf.checkSkillAssignment(X, lam, mu)[0]
    cost = total - f * (X.sum(axis=1) > 0).sum() - cs * X.sum()                    # holding + backorders (independent seeds)
    Xff = np.ones_like(X)
    cost_ff, S_ff, EBO_ff, _, util_ff, _ = SIf.simulation_optimization_run(lam, mu, h, b, Xff, useDatabase=False, startSeed=FINAL_SEED)
    emp, emp_ff = int((X.sum(axis=1) > 0).sum()), M
    out = {k: d[k] for k in ("caseID", "n_servers", "m_skills", "utilization_rate", "holding_cost_min", "holding_costs_variant",
                             "machine_cost_factor", "skill_cost_factor", "failure_rates", "service_rates", "holding_costs",
                             "penalty_cost", "machine_cost", "skill_cost") if k in d}
    out.update({"best_assignment": X.tolist(), "stocks": [int(s) for s in S], "EBO": [float(e) for e in EBO],
                "Stock_Costs": float((h * S).sum()), "EBO_Costs": float(b * np.sum(EBO)), "Skills_Costs": float(cs * X.sum()),
                "Machine_Costs": float(f * emp), "total_cost": float(cost + f * emp + cs * X.sum()),
                "stocks_ff": [int(s) for s in S_ff], "EBO_ff": [float(e) for e in EBO_ff], "Stock_Costs_ff": float((h * S_ff).sum()),
                "EBO_Costs_ff": float(b * np.sum(EBO_ff)), "Skills_Costs_ff": float(cs * Xff.sum()), "Machine_Costs_ff": float(f * emp_ff),
                "total_cost_ff": float(cost_ff + f * emp_ff + cs * Xff.sum()),
                "utilizationRatesAverage": None if util is None else np.asarray(util).tolist(),
                "skillServerAssignmentStatisitcsAverage": None if assign is None else np.asarray(assign).tolist(),
                "utilizationRatesAverage_ff": np.asarray(util_ff).tolist(),
                "search_total_cost": float(logbook.search_total_cost), "final_design_feasible": bool(feasible),
                "log": log_records(logbook) if algorithm != "vns" else list(logbook),
                "nr_iterations": params.get("generation", params.get("budget")),
                "designs_simulated": getattr(logbook, "simulated", None if DISK_CACHE else len(SIf.database)),
                "evaluations_requested": getattr(logbook, "requested", None), "runtime": runtime, "execTime": runtime,
                "algorithm": algorithm.upper(), {"ga": "ga_params", "vns": "vns_params"}.get(algorithm, "pso_params"): params, "seed": seed,
                "rule": rule, "preemptive": bool(preemptive), "min_coverage": int(coverage), "priorities": None if SIf.priorities is None else np.asarray(SIf.priorities).tolist(),
                "simulator": "simulation_fast (compiled), SimulationInterface_fast, " + {"ga": "GA_Optimization_v2", "vns": "VNS_optimization_v2"}.get(algorithm, "PSO_optimization_v2"),
                "final_evaluation": f"{REPS} x {N_REQ} requests, seeds {FINAL_SEED} + 20 i, no warm-up, maxQueue {MAXQ}"})
    json.dump(out, open(os.path.join(out_dir, os.path.basename(path)), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return c, feasible, out["total_cost"], out["total_cost_ff"], d["total_cost"], runtime


def rerun_case(args):
    """Run one case; an error is reported and the run goes on with the other cases (a crash in one case used to stop
    the whole run). The failed case has no result file, so a later run retries it."""
    try:
        return _rerun_case(args)
    except Exception as e:
        import traceback
        return case_number(args[0]), f"ERROR {type(e).__name__}: {e}", float("nan"), float("nan"), float("nan"), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--cases", default=None); ap.add_argument("--from-csv", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--test", action="store_true")
    ap.add_argument("--include-truncated", action="store_true",
                    help="with --from-csv: also feasible designs with more than 0.1%% of a distribution above 500")
    ap.add_argument("--all", action="store_true", help="every case in the folder")
    ap.add_argument("--algorithm", choices=["pso", "ga", "vns"], default="pso", help="pso (PSO_optimization_v2), ga (GA_Optimization_v2) or vns (VNS_optimization_v2)")
    ap.add_argument("--disk-cache", action="store_true", help="keep the cache of simulated designs in SQLite files instead of memory")
    ap.add_argument("--published", action="store_true", help="PSO with the published parameters instead of the tuned ones")
    ap.add_argument("--rule", choices=["fcfs", "mu", "hmu"], default="fcfs",
                    help="repair-priority rule inside the search and the final evaluations (joint optimization, A2)")
    ap.add_argument("--nonpreemptive", action="store_true", help="non-preemptive priorities (A3); output folder gets the suffix _np")
    ap.add_argument("--min-coverage", type=int, default=1, help="B6b: every SKU repaired by at least this many technicians; suffix _cov<k>")
    a = ap.parse_args()
    global DISK_CACHE; DISK_CACHE = a.disk_cache
    files = {case_number(p): p for p in glob.glob(os.path.join(a.folder, "*case_*.json"))}
    if a.all:
        wanted = sorted(files)
    elif a.from_csv:
        rows = list(csv.DictReader(open(a.from_csv)))
        wanted = [int(r["case"]) for r in rows if r["feasible"] == "False"]
        if a.include_truncated:
            wanted += [int(r["case"]) for r in rows if r["feasible"] == "True" and float(r["compiled share above 500"] or 0) > 1e-3]
    elif a.cases:
        wanted = [int(x) for x in a.cases.split(",")]
    else:
        sys.exit("give --cases, --from-csv or --all")
    if a.out is None:
        a.out = {"ga": "GA_results_rerun", "vns": "VNS_results_rerun"}.get(a.algorithm, "PSO_results_tuned") + ("" if a.rule == "fcfs" else f"_rule_{a.rule}") + ("_np" if a.nonpreemptive else "") + (f"_cov{a.min_coverage}" if a.min_coverage > 1 else "")
    if a.disk_cache and a.rule != "fcfs":
        sys.exit("--disk-cache with a priority rule: the cache key does not include the rule; use the in-memory cache")
    os.makedirs(a.out, exist_ok=True)
    def finished(c):
        f = os.path.join(a.out, os.path.basename(files[c]))
        if not os.path.exists(f): return False
        try:
            json.load(open(f)); return True
        except Exception:
            print(f"case {c}: unreadable result file (interrupted write) -- removed, will be rerun"); os.remove(f); return False
    todo = [c for c in wanted if c in files and not finished(c)]
    params = {"ga": (GA_TEST_PARAMS, GA_PARAMS), "vns": (VNS_TEST_PARAMS, VNS_PARAMS)}.get(a.algorithm, (TEST_PARAMS, PSO_PARAMS))[0 if a.test else 1]
    if a.algorithm == "pso" and a.published:
        params = dict(PSO_PARAMS_PUBLISHED, generation=2, population=6) if a.test else PSO_PARAMS_PUBLISHED
    print(f"{a.algorithm.upper()} rule {a.rule}: {len(wanted)} cases requested, {len(todo)} to run -> {a.out} | parameters {params} | {a.workers} in parallel", flush=True)
    size = {c: (lambda d: int(d["m_skills"]) * int(d["n_servers"]))(json.load(open(files[c]))) for c in todo}
    todo.sort(key=lambda c: -size[c])                       # largest instances first: shorter idle tail at the end
    jobs = [(files[c], a.out, a.seed_base + c, params, a.algorithm, a.rule, not a.nonpreemptive, a.min_coverage) for c in todo]
    runner = Pool(a.workers).imap_unordered(rerun_case, jobs) if a.workers > 1 else map(rerun_case, jobs)
    for c, feas, tot, tot_ff, old, rt in runner:
        if isinstance(feas, str):
            print(f"case {c:4d}: {feas} -- skipped, retried on the next run", flush=True); continue
        print(f"case {c:4d}: final design feasible {feas} | total {tot:,.0f} (independent seeds) | full cross-training {tot_ff:,.0f} "
              f"| saving {1 - tot / tot_ff:.1%} | old recorded {old:,.0f} | {rt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
