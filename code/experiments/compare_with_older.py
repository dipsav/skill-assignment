"""
compare_with_older.py -- rerun the final design of every PSO result file with the initial (older) simulator and with
the compiled one, and compare.

Usage (from the folder that contains simulation_codes/, with your simulation.py in it):
    python compare_with_older.py <folder with fullRangeResultsPSO_*_case_*.json> [--reps 35] [--workers 8]
                                 [--out compare_pso.csv] [--limit K]

Per file:
  1. feasibility of best_assignment (checkSkillAssignment from simulation_fast.py). Infeasible designs are listed, not rerun.
  2. OLDER simulator, exactly as the PSO ran it: 35 replications x 100,000 requests, no warm-up, seeds 10 + 20 i,
     distributions cut at 500, newsvendor on the averaged distribution. Its total cost should REPRODUCE the recorded
     total_cost (a check that the setup is the PSO's; needs --reps 35).
  3. COMPILED simulator, same design, 35 replications (distributions up to 2000; the share above 500 is reported).
  4. Comparison at the recorded base-stock levels: holding + backorder cost per replication under both simulators
     -> relative difference with a 95% interval; per-SKU mean numbers in system inside their intervals.
  5. with --fresh-older: the older simulator again with INDEPENDENT seeds. The PSO chose each design using seeds 10 + 20 i,
     so the recorded cost of the chosen design is optimistic (selection bias). Compiled vs older-fresh tests the
     simulators; recorded vs older-fresh measures the selection bias.
Results are appended to the CSV as each case finishes; Ctrl-C stops cleanly, and the same command resumes.
"""
import os, re, csv, glob, json, time, argparse, signal
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as student_t

MAXQ_PSO, MAXQ_FAST, N_REQ = 500, 2000, 100_000
FIELDS = ["case", "N", "M", "rho", "feasible", "recorded total", "older rerun total", "reproduced",
          "older parts in system", "compiled parts in system", "SKUs agreeing", "SKUs",
          "cost difference (compiled vs older)", "cost difference 95% half-width", "within interval",
          "compiled share above 500", "older fresh parts in system", "cost: recorded seeds vs fresh seeds (older)",
          "cost: compiled vs older fresh", "cost: compiled vs older fresh 95% half-width", "minutes"]


def case_number(path):
    m = re.search(r"case_(\d+)", os.path.basename(path)); return int(m.group(1)) if m else 10**9


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # workers ignore Ctrl-C; the main process handles it


def older_rep(args):
    """One replication of the older simulator, as the PSO ran it; returns the distribution cut at 500 (SKUs x 500)."""
    lam, mu, X, seed = args
    from simulation_codes import simulation as older
    R = X * mu[None, :]
    d, *_ = older.simulation(lam.sum(), np.cumsum(lam / lam.sum()), R, R.copy(), [0.0], N_REQ, 0, seed)
    p = np.zeros((len(lam), MAXQ_PSO))
    for k in d:
        for q, v in d[k].items():
            if q < MAXQ_PSO: p[int(k), int(q)] = v
    return p


def interface_cost(dist, h, b):
    """The PSO interface's newsvendor (OptimizeStockLevelsAndCostsSimBased), on the averaged, truncated distribution."""
    maxq = dist.shape[1]; n = np.arange(maxq); N = len(h)
    S = np.zeros(N, dtype=int); EBO = (dist * n).sum(axis=1); hb = h / b
    for k in range(N):
        while S[k] < maxq and dist[k, S[k] + 1:].sum() > hb[k]:
            S[k] += 1
            EBO[k] = (dist[k, S[k]:] * n[:-S[k]]).sum()
    return float((S * h).sum() + b * EBO.sum()), S


def ehb_at(dist, S, h, b):
    """Holding + backorder cost of one replication's distribution at fixed stock levels S."""
    q = np.arange(dist.shape[1])
    return float((h * S).sum() + b * (np.maximum(q[None, :] - S[:, None], 0) * dist).sum())


def compare_case(path, pool, a, tcrit, sf, checkSkillAssignment):
    c = case_number(path); t0 = time.time(); d = json.load(open(path))
    lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
    b, X = float(d["penalty_cost"]), np.asarray(d["best_assignment"], int); M, N = X.shape
    fixed = d["machine_cost"] * (X.sum(axis=1) > 0).sum() + d["skill_cost"] * X.sum()
    row = {"case": c, "N": N, "M": M, "rho": d.get("utilization_rate"), "recorded total": d["total_cost"], "SKUs": N}
    if not checkSkillAssignment(X, lam, mu)[0]:
        row["feasible"] = False
        print(f"case {c:4d}: INFEASIBLE final design -- not rerun"); return row
    tasks = [(lam, mu, X, 10 + 20 * i) for i in range(a.reps)]
    po = np.array(pool.map_async(older_rep, tasks).get(10**7))                 # .get(timeout) lets Ctrl-C through
    cost_old, _ = interface_cost(po.mean(axis=0), h, b)
    pf = np.array([sf.simulate(lam, X * mu[None, :], None, N_REQ, 0, 100_000 + 20 * i, MAXQ_FAST)["dist"] for i in range(a.reps)])
    above = float(pf[:, :, MAXQ_PSO:].sum(axis=2).max())
    if a.fresh_older:                                                         # older simulator, independent seeds
        pn = np.array(pool.map_async(older_rep, [(lam, mu, X, 200_000 + 20 * i) for i in range(a.reps)]).get(10**7))
    S = np.asarray(d["stocks"], int)
    co = np.array([ehb_at(p, S, h, b) for p in po]); cf = np.array([ehb_at(p, S, h, b) for p in pf])
    diff = cf.mean() / co.mean() - 1; hw = tcrit * np.sqrt(co.var(ddof=1) / a.reps + cf.var(ddof=1) / a.reps) / co.mean()
    Lo, Lf = (po * np.arange(MAXQ_PSO)).sum(axis=2), (pf * np.arange(MAXQ_FAST)).sum(axis=2)
    hwL = tcrit * np.sqrt(Lo.var(axis=0, ddof=1) / a.reps + Lf.var(axis=0, ddof=1) / a.reps)
    agree = int((np.abs(Lo.mean(axis=0) - Lf.mean(axis=0)) <= hwL).sum())
    total_old = cost_old + fixed
    if a.fresh_older:
        cn = np.array([ehb_at(p, S, h, b) for p in pn]); Ln = (pn * np.arange(MAXQ_PSO)).sum(axis=2)
        dfn = cf.mean() / cn.mean() - 1; hwn = tcrit * np.sqrt(cn.var(ddof=1) / a.reps + cf.var(ddof=1) / a.reps) / cn.mean()
        row.update({"older fresh parts in system": round(float(Ln.sum(axis=1).mean()), 4),
                    "cost: recorded seeds vs fresh seeds (older)": round(co.mean() / cn.mean() - 1, 5),
                    "cost: compiled vs older fresh": round(dfn, 5), "cost: compiled vs older fresh 95% half-width": round(hwn, 5)})
    row.update({"feasible": True, "older rerun total": round(total_old, 4),
                "reproduced": bool(np.isclose(total_old, d["total_cost"], rtol=1e-9)) if a.reps == 35 else "n/a (needs --reps 35)",
                "older parts in system": round(float(Lo.sum(axis=1).mean()), 4),
                "compiled parts in system": round(float(Lf.sum(axis=1).mean()), 4),
                "SKUs agreeing": agree, "cost difference (compiled vs older)": round(diff, 5),
                "cost difference 95% half-width": round(hw, 5), "within interval": bool(abs(diff) <= hw),
                "compiled share above 500": round(above, 6), "minutes": round((time.time() - t0) / 60, 2)})
    print(f"case {c:4d}: reproduced {row['reproduced']} | parts {row['older parts in system']:.3f} vs {row['compiled parts in system']:.3f} "
          f"| SKUs agreeing {agree}/{N} | cost {diff:+.2%} ± {hw:.2%} | {row['minutes']:.1f} min")
    if a.fresh_older:
        print(f"           older with fresh seeds: parts {row['older fresh parts in system']:.3f} | PSO seeds vs fresh seeds "
              f"{row['cost: recorded seeds vs fresh seeds (older)']:+.2%} | compiled vs older fresh {dfn:+.2%} ± {hwn:.2%}")
    return row


def summary(path, reps):
    rows = list(csv.DictReader(open(path))); feas = [r for r in rows if r["feasible"] == "True"]
    print(f"\n{len(rows)} cases: {len(rows) - len(feas)} infeasible final designs (not rerun), {len(feas)} rerun")
    if not feas: return
    rep = sum(r["reproduced"] == "True" for r in feas); win = sum(r["within interval"] == "True" for r in feas)
    sk = sum(int(r["SKUs agreeing"]) for r in feas); tot = sum(int(r["SKUs"]) for r in feas)
    trunc = [r["case"] for r in feas if float(r["compiled share above 500"]) > 1e-3]
    print(f"older simulator reproduces the recorded cost: {rep} of {len(feas)}" + ("" if reps == 35 else "  (reproduction needs --reps 35)"))
    print(f"cost difference inside its 95% interval: {win} of {len(feas)} (about 95% expected by chance alone)")
    print(f"per-SKU means inside their intervals: {sk} of {tot} ({sk / tot:.1%})")
    if trunc: print(f"designs with more than 0.1% of a distribution above 500 (the PSO's cut-off understated their cost): {trunc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--reps", type=int, default=35)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="compare_pso.csv"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pattern", default="*case_*.json")
    ap.add_argument("--cases", default=None, help="only these case numbers, e.g. 3,5")
    ap.add_argument("--fresh-older", action="store_true",
                    help="also rerun the older simulator with independent seeds (separates selection bias from simulator differences)")
    a = ap.parse_args()
    from simulation_codes import simulation_fast as sf
    from simulation_codes.simulation_fast import checkSkillAssignment

    files = sorted(glob.glob(os.path.join(a.folder, a.pattern)), key=case_number)
    if a.cases: files = [f for f in files if case_number(f) in {int(x) for x in a.cases.split(",")}]
    files = files[: a.limit]
    done = {int(r["case"]) for r in csv.DictReader(open(a.out))} if os.path.exists(a.out) else set()
    new_file = not os.path.exists(a.out)
    fh = open(a.out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file: w.writeheader(); fh.flush()
    tcrit = student_t.ppf(0.975, a.reps - 1)
    print(f"{len(files)} files; {len(done)} already in {a.out}; {a.reps} replications per simulator; {a.workers} workers")
    pool = Pool(a.workers, initializer=_ignore_sigint)
    try:
        for path in files:
            if case_number(path) in done: continue
            row = compare_case(path, pool, a, tcrit, sf, checkSkillAssignment)
            w.writerow(row); fh.flush()                                     # a case is saved only when complete
    except KeyboardInterrupt:
        pool.terminate(); pool.join(); fh.close()
        print("\nstopped -- finished cases are saved; run the same command again to resume")
        return
    pool.close(); pool.join(); fh.close()
    summary(a.out, a.reps)


if __name__ == "__main__":
    main()
