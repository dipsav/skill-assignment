"""
Shared code for the R1 (rule comparison) and R2 (SKU-specific surges) notebooks.

Inputs: the optimized designs, one JSON object per line in designs.jsonl (see load_designs for the fields).
Simulator: simulation_pso_paper_v2.py (fixed priority simulator), FCFS = priorities None, rules = priority vectors
(lower value = higher priority).
"""
import json, os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.stats import t as student_t, ttest_rel, f as f_dist
import simulation_pso_paper_v2 as simv2

RULES = {"FCFS": None, "mu": "mu", "h": "h", "hmu": "hmu"}       # priority = 1/mu, 1/h, 1/(h mu)

# ----------------------------------------------------------------------------------------------- designs
REQUIRED = ["instance", "lambda", "mu", "h", "b", "f", "c", "X", "S"]

def load_designs(path):
    """designs.jsonl: one object per line with
       instance (id), factors (dict, optional: N, M, rho, hmin, relation, f, c),
       lambda [N], mu [N], h [N], b (scalar), f (scalar), c [N] (skill cost per SKU),
       X [M][N] skill matrix (rows technicians), S [N] base-stock levels, eps (optional, default 0.01)."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip(): continue
            d = json.loads(line)
            missing = [k for k in REQUIRED if k not in d]
            if missing: raise ValueError(f"instance {d.get('instance')}: missing {missing}")
            for k in ("lambda", "mu", "h", "c", "S"): d[k] = np.asarray(d[k], float)
            d["X"] = np.asarray(d["X"], int); d["S"] = d["S"].astype(int)
            M, N = d["X"].shape
            assert len(d["lambda"]) == N == len(d["mu"]) == len(d["h"]) == len(d["c"]) == len(d["S"]), f"instance {d['instance']}: inconsistent sizes"
            d.setdefault("factors", {}); d.setdefault("eps", 0.01)
            out.append(d)
    return out

def priorities_for(rule, mu, h):
    if rule == "FCFS": return None
    if rule == "mu": return 1.0 / mu
    if rule == "h": return 1.0 / h
    if rule == "hmu": return 1.0 / (h * mu)
    raise ValueError(rule)

# ----------------------------------------------------------------------------------------------- stability (constraints 2-8)
def stable(X, lam, mu, eps=0.01):
    """Can the skill matrix carry the workload with every employed technician below 1 - eps? (LP feasibility)"""
    M, N = X.shape; y = (X.sum(axis=1) > 0).astype(float)
    idx = [(i, j) for i in range(N) for j in range(M) if X[j, i] == 1]
    if not idx: return False
    A_eq = np.zeros((N, len(idx))); b_eq = np.ones(N)
    A_ub = np.zeros((M, len(idx))); b_ub = (1 - eps) * y
    for k, (i, j) in enumerate(idx):
        A_eq[i, k] = 1.0; A_ub[j, k] = lam[i] / mu[i]
    res = linprog(np.zeros(len(idx)), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return bool(res.status == 0)

# ----------------------------------------------------------------------------------------------- one replication
def _one_rep(args):
    lam, mu, X, prio, seed, n_failures, maxq = args
    lam, mu = np.asarray(lam, float), np.asarray(mu, float); X = np.asarray(X, float)
    rates = X * mu[None, :]; costs = rates.copy()
    if prio is not None: costs = costs * np.asarray(prio, float)[None, :]
    d, _, util, _ = simv2.simulation(lam.sum(), np.cumsum(lam / lam.sum()), rates, costs, [0.0], n_failures, 0, seed)
    N = len(lam); dist = np.zeros((N, maxq))
    for k in d:
        for q, p in d[k].items():
            if q < maxq: dist[k, q] += p
    return dist, util

def ebo(p, S):
    q = np.arange(len(p)); return float((np.maximum(q - S, 0) * p).sum())

def evaluate(design, rule, lam=None, reps=35, n_failures=100_000, maxq=2000, seed0=0, pool=None):
    """Simulate a design under a rule (and optionally changed failure rates) with the base-stock levels FIXED.
    Returns per-replication cost components and pooled distributions/utilization."""
    lam = design["lambda"] if lam is None else np.asarray(lam, float)
    mu, h, b, f, c, X, S = design["mu"], design["h"], design["b"], design["f"], design["c"], design["X"], design["S"]
    prio = priorities_for(rule, mu, h)
    jobs = [(lam, mu, X, prio, seed0 + r, n_failures, maxq) for r in range(reps)]
    results = pool.map(_one_rep, jobs) if pool is not None else [_one_rep(j) for j in jobs]
    y = (X.sum(axis=1) > 0); tech_cost = f * y.sum(); skill_cost = float((c[None, :] * X).sum()); hold = float((h * S).sum())
    rows = []
    for r, (dist, util) in enumerate(results):
        dist = dist / dist.sum(axis=1, keepdims=True)
        back = b * sum(ebo(dist[i], S[i]) for i in range(len(lam)))
        rows.append({"replication": r, "technicians": tech_cost, "skills": skill_cost, "holding": hold, "backorders": back,
                     "total": tech_cost + skill_cost + hold + back,
                     "parts_in_system": [float((dist[i] * np.arange(maxq)).sum()) for i in range(len(lam))]})
    pooled = np.mean([d / d.sum(axis=1, keepdims=True) for d, _ in results], axis=0)
    util = np.mean([u for _, u in results], axis=0)
    return rows, pooled, util

# ----------------------------------------------------------------------------------------------- bookkeeping (resumable)
def done_keys(path, keys):
    if not os.path.exists(path): return set()
    return {tuple(json.loads(l)[k] for k in keys) for l in open(path) if l.strip()}

def append_jsonl(path, obj):
    with open(path, "a") as fh: fh.write(json.dumps(obj) + "\n")

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []

# ----------------------------------------------------------------------------------------------- statistics
def rm_anova(table):
    """One-way repeated-measures ANOVA on a (subjects x treatments) array. Returns F, df, p."""
    Y = np.asarray(table, float); n, k = Y.shape
    gm = Y.mean(); ss_treat = n * ((Y.mean(axis=0) - gm) ** 2).sum(); ss_subj = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    ss_tot = ((Y - gm) ** 2).sum(); ss_err = ss_tot - ss_treat - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1); F = (ss_treat / df1) / (ss_err / df2)
    return F, (df1, df2), float(f_dist.sf(F, df1, df2))

def bonferroni_pairs(df, value, group, subject):
    """Paired t-tests between every pair of groups (paired by subject), Bonferroni-corrected."""
    groups = sorted(df[group].unique()); pairs = list(itertools.combinations(groups, 2)); rows = []
    wide = df.pivot(index=subject, columns=group, values=value)
    for a, b_ in pairs:
        t_stat, p = ttest_rel(wide[a], wide[b_])
        rows.append({"A": a, "B": b_, "mean A - B": wide[a].mean() - wide[b_].mean(), "p": p, "p (Bonferroni)": min(1.0, p * len(pairs))})
    return pd.DataFrame(rows)

def ci95(x):
    x = np.asarray(x, float); return student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

# ----------------------------------------------------------------------------------------------- synthetic designs for a dry run
def synthetic_designs(path, n_inst=4, seed=5):
    """Small invented designs (N = 4, M = 3, partial skills) with newsvendor base stock from a short FCFS run,
    in the designs.jsonl format -- only to exercise the notebooks."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        for k in range(n_inst):
            N, M, rho = 4, 3, (0.65, 0.80)[k % 2]
            mu = np.exp(rng.uniform(-1, 1, N)); lam = M * rho * rng.dirichlet(np.ones(N)) * mu
            while True:                                                   # a random partial skill matrix that can carry the load
                X = (rng.random((M, N)) < 0.6).astype(int)
                if X.sum(axis=0).min() > 0 and X.sum(axis=1).min() > 0 and stable(X, lam, mu): break
            h = rng.uniform(1, 1000, N); b = 50 * (lam * h).sum() / lam.sum(); f = 10_000.0; c = 0.05 * f * np.ones(N)
            dist, _ = _one_rep((lam, mu, X, None, 1 + k, 20_000, 500)); dist /= dist.sum(axis=1, keepdims=True)
            S = [int(np.searchsorted(np.cumsum(dist[i]), (b - h[i]) / b)) for i in range(N)]
            fh.write(json.dumps({"instance": k + 1, "factors": {"N": N, "M": M, "rho": rho, "hmin": 1, "relation": "IND", "f": f, "c": 0.05},
                                 "lambda": lam.tolist(), "mu": mu.tolist(), "h": h.tolist(), "b": float(b), "f": f, "c": c.tolist(),
                                 "X": X.tolist(), "S": S, "eps": 0.01}) + "\n")
    return path
