"""Tests of simulation_fast.py against exact results (and, if simulation_pso_paper_v2.py is present, against the Python v2).
Run from the folder that contains simulation_codes/ and the exact solvers:  python test_simulation_fast.py"""
import json, time, numpy as np, warnings; warnings.filterwarnings("ignore")
from math import factorial
from scipy.stats import t as st
from simulation_codes import simulation_fast as sf
from fcfs_exact import fcfs_multiclass_mmc
from exact_priority2 import priority2_exact
from exact3 import three_level_exact

def reps(lam, R, score=None, n=35, nreq=100_000, seed0=10, maxq=2000):
    rows, pw = [], []
    for r in range(n):
        o = sf.simulate(lam, R, score, nreq, 0, seed0 + 20 * r, maxq)
        rows.append((o["dist"] * np.arange(maxq)).sum(axis=1)); pw.append(o)
    rows = np.array(rows); return rows.mean(0), st.ppf(0.975, n - 1) * rows.std(0, ddof=1) / np.sqrt(n), pw

def report(name, m, hw, ex):
    inside = np.abs(m - ex) <= hw
    print(f"{name:52s} " + "  ".join(f"{a:.4f}±{h:.4f} (exact {e:.4f}){'' if ok else ' *'}" for a, h, e, ok in zip(m, hw, ex, inside))[:400])
    return inside.all()

ok = []
t0 = time.time()
# 1. FCFS, full cross-training, against the exact multi-class M/M/c solution
for lam, mu, c in (([0.3, 0.6, 1.2], [0.5, 1.0, 2.0], 3), ([0.24, 2.4], [0.3, 3.0], 2), ([0.2, 0.5, 0.9, 1.6], [0.4, 0.8, 1.5, 2.5], 4)):
    lam, mu = np.array(lam, float), np.array(mu, float); m, hw, _ = reps(lam, np.ones((c, len(lam))) * mu[None, :])
    ok.append(report(f"1. FCFS full, {len(lam)} SKUs, {c} techs", m, hw, fcfs_multiclass_mmc(lam, mu, c)["mean"]))
# 2. dedicated technicians: two M/M/1 at rho 0.5
m, hw, _ = reps(np.array([0.5, 0.5]), np.eye(2)); ok.append(report("2. dedicated technicians (M/M/1 each)", m, hw, np.array([1.0, 1.0])))
# 3. pairs: two independent blocks of two technicians
chain4 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
lam4, mu4 = np.array([0.3, 0.6, 1.2, 0.6]), np.array([0.5, 1.0, 2.0, 1.0]); pairs = np.kron(np.eye(2), np.ones((2, 2)))
m, hw, _ = reps(lam4, pairs * mu4[None, :]); ex = np.concatenate([fcfs_multiclass_mmc(lam4[b], mu4[b], 2)["mean"] for b in ([0, 1], [2, 3])])
ok.append(report("3. pairs (two blocks of two technicians)", m, hw, ex))
# 4. probability of waiting, full cross-training (all technicians busy)
lam, mu = np.array([0.3, 0.6, 1.2]), np.array([0.5, 1.0, 2.0]); _, _, outs = reps(lam, np.ones((3, 3)) * mu[None, :])
pw = np.array([1 - (o["p_nowait"] * o["arrivals"]).sum() / o["arrivals"].sum() for o in outs])
pex = fcfs_multiclass_mmc(lam, mu, 3)["total"][3:].sum(); hwp = st.ppf(0.975, len(pw) - 1) * pw.std(ddof=1) / np.sqrt(len(pw))
print(f"4. P(wait), 3 SKUs, 3 techs: {pw.mean():.5f} ± {hwp:.5f} vs exact {pex:.5f}"); ok.append(abs(pw.mean() - pex) <= hwp)
# 5. two priority levels (fast preempts slow), rho 0.8, 3 technicians
mu2 = np.array([np.e, np.exp(-1)]); lam2 = np.array([0.5, 0.5]) * 0.8 * 3 * mu2
r2 = priority2_exact(lam2[0], mu2[0], lam2[1], mu2[1], 3); ex2 = np.array([(x * np.arange(len(x))).sum() for x in (r2["high"], r2["low"])])
m, hw, _ = reps(lam2, np.ones((3, 2)) * mu2[None, :], score=np.ones((3, 2)) * (1 / mu2)[None, :]); ok.append(report("5. two priority levels, rho 0.8", m, hw, ex2))
# 6. three priority levels, 2 technicians
lam3, mu3 = np.array([0.4, 0.3, 0.25]), np.array([2.0, 1.0, 0.5]); ex3, _ = three_level_exact(lam3, mu3, 2, K=(10, 35, 100))
# pooled over four seed blocks (240 x 200,000): a single block misses one interval by chance (-2.2 sd for level 2)
R3, sc3 = np.ones((2, 3)) * mu3[None, :], np.ones((2, 3)) * np.array([1, 2, 3.0])[None, :]
rows3 = np.array([(sf.simulate(lam3, R3, sc3, 200_000, 0, s0 + 20 * r, 2000)["dist"] * np.arange(2000)).sum(axis=1)
                  for s0 in (10, 50_000, 90_000, 130_000) for r in range(60)])
m, hw = rows3.mean(0), st.ppf(0.975, len(rows3) - 1) * rows3.std(0, ddof=1) / np.sqrt(len(rows3))
ok.append(report("6. three priority levels (240 runs)", m, hw, np.array(ex3)))
# 7. utilization identity, partial skills, three priority levels
lamc, muc = np.array([0.4, 0.5, 0.6]), np.array([1.0, 1.2, 0.8]); chain = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
o = sf.simulate(lamc, chain * muc[None, :], np.where(chain > 0, np.array([1, 2, 3.0])[None, :], np.inf), 400_000, 0, 5, 2000)
print(f"7. utilization per SKU {np.round(o['util'].sum(0), 3)} vs lambda/mu {np.round(lamc / muc, 3)}"); ok.append(np.allclose(o["util"].sum(0), lamc / muc, atol=0.01))
# 8. agreement with the Python simulator v2: four-SKU chain, rho 0.6, FCFS and three priority levels
try:
    import simulation_pso_paper_v2 as v2
except ImportError:
    v2 = None
R4 = chain4 * mu4[None, :]
for label, prio in ((("FCFS", None), ("3 levels", np.array([1, 2, 3, 1.0]))) if v2 is not None else ()):
    costs = R4 if prio is None else R4 * prio[None, :]
    py = np.array([sum(sum(q * p for q, p in dd[k].items()) for k in dd) for dd in
                   (v2.simulation(lam4.sum(), np.cumsum(lam4 / lam4.sum()), R4, costs, [0.0], 100_000, 0, 7000 + 20 * r)[0] for r in range(20))])
    fa = np.array([(sf.simulate(lam4, R4, None if prio is None else sf.scores_from_costs(R4, costs), 100_000, 0, 9000 + 20 * r, 2000)["dist"] * np.arange(2000)).sum() for r in range(20)])
    diff = fa.mean() / py.mean() - 1; se = np.sqrt(fa.var(ddof=1) / 20 + py.var(ddof=1) / 20) / py.mean()
    print(f"8. chain, rho 0.6, {label:8s}: compiled {fa.mean():.4f} vs Python v2 {py.mean():.4f} ({diff:+.2%} ± {1.96*se:.2%})"); ok.append(abs(diff) <= 1.96 * se)

# 9. priority order checked after every event (debug counter in the compiled loop), partial skills, three rules
lamq, muq = np.array([0.5, 0.6, 0.7, 0.4]), np.array([1.0, 1.2, 0.8, 0.6]); Rq = chain4 * muq[None, :]
for label, sc in (("FCFS", None), ("3 levels", np.where(chain4 > 0, np.array([1, 2, 3, 2.0])[None, :], np.inf)),
                  ("mu rule", np.where(chain4 > 0, (1 / muq)[None, :], np.inf))):
    o = sf.simulate(lamq, Rq, sc, 200_000, 0, 9, 2000, check_invariant=True)
    print(f"9. priority order, chain, {label:8s}: violations {o['invariant_violations']} of {o['events']:,} events"); ok.append(o["invariant_violations"] == 0)
# 10. priorities with partial skills against an exact answer: two blocks of two technicians, two priority levels per block
lamp, mup = np.array([1.2, 0.3, 0.9, 0.25]), np.array([2.5, 0.5, 1.5, 0.4]); Rp = pairs * mup[None, :]
exp_ = []
for hi, lo in ((0, 1), (2, 3)):
    r = priority2_exact(lamp[hi], mup[hi], lamp[lo], mup[lo], 2); exp_ += [(x * np.arange(len(x))).sum() for x in (r["high"], r["low"])]
m, hw, _ = reps(lamp, Rp, score=np.where(pairs > 0, np.array([1, 2, 1, 2.0])[None, :], np.inf))
ok.append(report("10. pairs, two priority levels per block", m, hw, np.array(exp_)))
print(f"\n{sum(ok)} of {len(ok)} checks passed  [{time.time()-t0:.0f}s]")
