"""
t1_tuning.py -- tuning study T1: the proposed and the original BPSO on the small brute-force instances, with the exact
cost of every design looked up in tuning/tables.pkl (no simulation; the optimum is known).

The BPSO code is PSO_optimization_v2 unchanged; only its evaluation call (simulation_optimization_run) is replaced by the
table lookup, so the tuned code is exactly the code of the main runs. A matrix not in the table is infeasible (the
table holds every capacity-feasible design up to technician order).

Swarm: 20 particles, 50 generations (the instances have at most ~1,200 distinct feasible designs, so the 100 x 25 of the
paper nearly enumerates them). Measured per run: the gap of the best design to the optimum at the end and when the
search has evaluated 10%, 25% and 50% of the instance's feasible designs (distinct designs), and whether the optimum
was found.

Grids
  proposed (increasing inertia): c1 {0.5, 1, 2} x c2 {1, 2, 3, 4} x w_min {0.01, 0.1} x w_max {1.0, 1.5, 2.0}  (72)
  original (El-Maleh, decreasing inertia): c1 {0.5, 1, 2} x c2 {0.5, 1, 2} x w_max {0.3, 0.5, 0.9} x w_min {0.01, 0.1}  (54)

    python3 tuning/t1_tuning.py --min-designs 50 --seeds 1 2 --workers 4 --out tuning/t1_results.csv
"""
import argparse, itertools, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

GRIDS = {
    "proposed": [dict(c1=a, c2=b, inertia_min=lo, inertia_max=hi) for a, b, lo, hi in
                 itertools.product((0.5, 1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (0.01, 0.1), (1.0, 1.5, 2.0))],
    "original": [dict(c1=a, c2=b, inertia_min=lo, inertia_max=hi, variant="original") for a, b, hi, lo in
                 itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.3, 0.5, 0.9), (0.01, 0.1))],
    # extended schedule study (6 Oct): smaller start, higher end, slower rise (power p), c2 fixed (no effect in V4)
    "proposed_ext": [dict(c1=a, c2=3.0, inertia_min=lo, inertia_max=hi, inertia_power=pw) for a, lo, hi, pw in
                     itertools.product((1.0, 2.0), (0.001, 0.01, 0.1), (1.0, 2.0, 4.0, 8.0), (1.0, 2.0, 4.0))],
    # the paper's Eq. 12 (contested bits reset to c2 (r2 - 1/2)); here c2 matters
    "reset": [dict(c1=a, c2=b, inertia_min=lo, inertia_max=hi, inertia_power=pw, variant="reset") for a, b, lo, hi, pw in
              itertools.product((1.0, 2.0), (1.0, 3.0), (0.001, 0.01, 0.1), (1.0, 2.0, 4.0, 8.0), (1.0, 2.0, 4.0))],
}
POP, GEN = 20, 50
CHECKPOINTS = (0.10, 0.25, 0.50)
_T = None


def _init(path):
    global _T
    _T = pickle.load(open(path, "rb"))


def canon(X):
    return np.array(sorted(map(tuple, np.asarray(X, np.int8))), np.int8).tobytes()


class Lookup:
    """Replaces simulation_optimization_run: EHB from the table, infinite for designs not in it; records the distinct
    feasible designs in the order they are first evaluated with the best total cost so far."""
    def __init__(self, t):
        self.t = t; self.seen = set(); self.best = np.inf; self.trace = []    # (distinct designs, best total)

    def __call__(self, failureRates, serviceRates, holdingCosts, penalty, X, *args, **kw):
        X = np.asarray(X); N = len(failureRates)
        key = canon(X); e = self.t["ehb"].get(key, np.inf)
        if np.isfinite(e) and key not in self.seen:
            self.seen.add(key)
            tot = e + self.t["f"] * (X.sum(axis=1) > 0).sum() + self.t["c"] * X.sum()
            if tot < self.best:
                self.best = tot
            self.trace.append((len(self.seen), self.best))
        return e, np.zeros(N, int), np.zeros(N), False, None, None


def run(task):
    ti, variant, pi, seed = task
    import PSO_optimization_v2 as pso
    t = _T[ti]; lk = Lookup(t)
    pso.simulation_optimization_run = lk
    params = dict(GRIDS[variant][pi], generation=GEN, population=POP, speedRange=0.0)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X, total, *_ = pso.optimizeSkillAssignmentPSO(t["M"], t["lam"], t["mu"], t["h"], t["b"], t["f"], t["c"],
                                                      pso_params=params, seed=1_000_000 * seed + ti)
    opt = t["optimum"]; n = t["n_designs"]
    tr = np.array(lk.trace) if lk.trace else np.array([[0, np.inf]])
    row = {"instance": t["instance"], "setting": t["setting"], "N": t["N"], "M": t["M"], "n_designs": n, "variant": variant,
           "param_id": pi, "inertia_power": 1.0, **{k: v for k, v in GRIDS[variant][pi].items() if k != "variant"}, "seed": seed,
           "gap": lk.best / opt - 1, "optimal": bool(lk.best <= opt * (1 + 1e-9)), "distinct": len(lk.seen),
           "returned_gap": total / opt - 1}
    for q in CHECKPOINTS:
        k = max(1, int(np.ceil(q * n)))
        m = tr[tr[:, 0] <= k]
        row[f"gap_at_{int(q * 100)}"] = (m[-1, 1] if len(m) else np.inf) / opt - 1
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.path.join(HERE, "tables.pkl")); ap.add_argument("--out", default=os.path.join(HERE, "t1_results.csv"))
    ap.add_argument("--min-designs", type=int, default=50); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--variants", nargs="+", default=["proposed", "original"]); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first n eligible instances only (tests)")
    a = ap.parse_args()
    T = pickle.load(open(a.tables, "rb"))
    idx = [i for i, t in enumerate(T) if t["n_designs"] >= a.min_designs]
    if a.limit: idx = idx[:a.limit]
    tasks = [(i, v, p, s) for v in a.variants for p in range(len(GRIDS[v])) for i in idx for s in a.seeds]
    print(f"{len(idx)} instances (>= {a.min_designs} feasible designs) x settings {[len(GRIDS[v]) for v in a.variants]} x "
          f"{len(a.seeds)} seeds = {len(tasks)} runs, {a.workers} workers", flush=True)
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.tables,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run, tasks, chunksize=20), 1):
            rows.append(r)
            if k % 2000 == 0 or k == len(tasks):
                pd.DataFrame(rows).to_csv(a.out, index=False)
                el = time.time() - t0
                print(f"{k}/{len(tasks)} runs, {el / 60:.1f} min, about {el / k * (len(tasks) - k) / 60:.0f} min left", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
