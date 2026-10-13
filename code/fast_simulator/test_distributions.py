"""Tests of the general-distribution kernel of simulation_fast (_run_general): exponential case against the exact
multi-server solutions, M/G/1 (Pollaczek-Khinchine, FCFS and preemptive-resume priorities) and GI/M/1 (Erlang, H2,
deterministic interarrival) against the exact mean number in system.   python test_distributions.py"""
import sys, time, numpy as np
from scipy import stats as st, optimize
sys.path.insert(0, "."); from simulation_codes import simulation_fast as sf
from fcfs_exact import fcfs_multiclass_mmc
from exact_priority2 import priority2_exact
from exact_nonpreemptive2 import nonpreemptive2_exact

MAXQ = 2000


def reps(lam, R, score=None, arrival=None, service=None, n=35, nreq=100_000, seed0=10, preemptive=True, warm=0):
    rows = []
    for r in range(n):
        o = sf.simulate(lam, R, score, nreq, warm, seed0 + 20 * r, MAXQ, arrival=arrival, service=service, preemptive=preemptive)
        rows.append((o["dist"] * np.arange(MAXQ)).sum(axis=1))
    rows = np.array(rows); return rows.mean(0), st.t.ppf(0.975, n - 1) * rows.std(0, ddof=1) / np.sqrt(n)


def report(name, m, hw, ex):
    inside = np.abs(m - ex) <= hw
    print(f"{name:58s} " + "  ".join(f"{a:.4f}±{h:.4f} (exact {e:.4f}){'' if k else ' *'}" for a, h, e, k in zip(m, hw, ex, inside)))
    return bool(inside.all())


def cv2(name, cv):
    if name == "erlang": return 1.0 / round(1.0 / cv ** 2)
    return {"exponential": 1.0, "lognormal": cv ** 2, "h2": cv ** 2, "deterministic": 0.0}[name]


def lst(name, cv, rate, s):
    """Laplace-Stieltjes transform of an interarrival time with mean 1/rate and the sampler's shape."""
    if name == "exponential": return rate / (rate + s)
    if name == "erlang": k = round(1.0 / cv ** 2); return (k * rate / (k * rate + s)) ** k
    if name == "h2":
        c2 = cv ** 2; p = 0.5 * (1 + np.sqrt((c2 - 1) / (c2 + 1))); l1, l2 = 2 * p * rate, 2 * (1 - p) * rate
        return p * l1 / (l1 + s) + (1 - p) * l2 / (l2 + s)
    if name == "deterministic": return np.exp(-s / rate)


ok = []; t0 = time.time()
# 1. exponential through the general kernel: FCFS full cross-training (exact multi-class M/M/c), two-level priorities
for lam, mu, c in (([0.3, 0.6, 1.2], [0.5, 1.0, 2.0], 3), ([0.2, 0.5, 0.9, 1.6], [0.4, 0.8, 1.5, 2.5], 4)):
    lam, mu = np.array(lam, float), np.array(mu, float); m, hw = reps(lam, np.ones((c, len(lam))) * mu[None, :], arrival=("exponential", 1.0), service=("exponential", 1.0))
    ok.append(report(f"1. exp kernel, FCFS full, {len(lam)} SKUs, {c} techs", m, hw, fcfs_multiclass_mmc(lam, mu, c)["mean"]))
lam2, mu2 = np.array([0.6, 0.9]), np.array([1.0, 0.6]); R2 = np.ones((3, 2)) * mu2[None, :]; sc2 = np.where(R2 > 0, 1.0 / R2, np.inf)
r2 = priority2_exact(lam2[0], mu2[0], lam2[1], mu2[1], 3); ex2 = np.array([(x * np.arange(len(x))).sum() for x in (r2["high"], r2["low"])])
m, hw = reps(lam2, R2, sc2, arrival=("exponential", 1.0), service=("exponential", 1.0)); ok.append(report("1. exp kernel, preemptive priorities, 3 techs", m, hw, ex2))
# 2. M/G/1 FCFS, one class: Pollaczek-Khinchine L = rho + rho^2 (1 + cv^2) / (2 (1 - rho))
for name, cv in (("erlang", 0.5), ("erlang", 1 / np.sqrt(2)), ("lognormal", 1.5), ("lognormal", 2.0), ("h2", 2.0), ("deterministic", 0.0)):
    # lognormal cv 2: the per-replication mean is heavy-tailed (E[S^2] converges slowly), so 140 replications are used
    rho = 0.7; m, hw = reps(np.array([rho]), np.array([[1.0]]), service=(name, cv), n=140 if name == "lognormal" else 70, seed0=50_000); ex = rho + rho ** 2 * (1 + cv2(name, cv)) / (2 * (1 - rho))
    ok.append(report(f"2. M/G/1 FCFS, {name} cv {cv:.2f}", m, hw, np.array([ex])))
# 3. M/G/1 FCFS, two classes with different distributions: Wq common, L_i = lam_i (Wq + E S_i)
lam3, mu3 = np.array([0.3, 0.2]), np.array([1.0, 0.5]); spec3 = (["erlang", "lognormal"], np.array([0.5, 2.0]))
ES2 = (1 / mu3) ** 2 * (1 + np.array([cv2("erlang", 0.5), cv2("lognormal", 2.0)])); rho3 = (lam3 / mu3).sum(); Wq = (lam3 * ES2).sum() / 2 / (1 - rho3)
m, hw = reps(lam3, mu3[None, :].copy(), service=spec3); ok.append(report("3. M/G/1 FCFS, Erlang + lognormal classes", m, hw, lam3 * (Wq + 1 / mu3)))
# 4. M/G/1 preemptive resume, two classes: E T_k = E S_k / (1 - s_{k-1}) + sum_{i<=k} lam_i E S_i^2 / 2 / ((1 - s_{k-1})(1 - s_k))
for spec in ((["erlang", "erlang"], np.array([0.5, 0.5])), (["lognormal", "lognormal"], np.array([2.0, 2.0])), (["h2", "erlang"], np.array([2.0, 0.5])), (["deterministic", "deterministic"], np.array([0.0, 0.0]))):
    lam4, mu4 = np.array([0.4, 0.25]), np.array([1.0, 0.5]); R4 = mu4[None, :].copy(); sc4 = 1.0 / R4
    c2s = np.array([cv2(n, c) for n, c in zip(*spec)]); ES2 = (1 / mu4) ** 2 * (1 + c2s); sig = np.cumsum(lam4 / mu4); s0 = np.r_[0.0, sig[:-1]]
    ET = 1 / mu4 / (1 - s0) + np.cumsum(lam4 * ES2 / 2) / ((1 - s0) * (1 - sig))
    m, hw = reps(lam4, R4, sc4, service=spec, n=70, seed0=50_000); ok.append(report(f"4. M/G/1 preemptive resume, {spec[0][0]}+{spec[0][1]}", m, hw, lam4 * ET))
# 5. GI/M/1: sigma = A*(mu (1 - sigma)), L = rho / (1 - sigma)
for name, cv in (("erlang", 0.5), ("erlang", 1 / np.sqrt(2)), ("h2", 2.0), ("deterministic", 0.0)):
    rate, mu = 0.7, 1.0; sg = optimize.brentq(lambda x: x - lst(name, cv, rate, mu * (1 - x)), 1e-12, 1 - 1e-12)
    m, hw = reps(np.array([rate]), np.array([[mu]]), arrival=(name, cv)); ok.append(report(f"5. GI/M/1, {name} arrivals cv {cv:.2f}", m, hw, np.array([rate / mu / (1 - sg)])))
# 6. GI/G/1 sanity: Erlang arrivals + Erlang service below the M/M/1 value; H2 + lognormal above it (Kingman direction)
rho = 0.7; mmm = rho / (1 - rho)
lo, _ = reps(np.array([rho]), np.array([[1.0]]), arrival=("erlang", 0.5), service=("erlang", 0.5)); hi, _ = reps(np.array([rho]), np.array([[1.0]]), arrival=("h2", 2.0), service=("lognormal", 2.0))
print(f"6. GI/G/1 ordering: Erlang/Erlang {lo[0]:.3f} < M/M/1 {mmm:.3f} < H2/lognormal {hi[0]:.3f}"); ok.append(bool(lo[0] < mmm < hi[0]))
# 7-8: total load 0.9 -> runs of 500,000 failures after a warm-up of 20,000 (with 100,000 from empty the low class is
# biased down by ~2%; checked: 100 x 1M with warm-up reproduces the exact values to 0.2%)
# 7. non-preemptive priorities. M/G/1 (Cobham): W_k = W0 / ((1 - s_{k-1})(1 - s_k)), W0 = sum lam_i E S_i^2 / 2
for spec in (None, (["erlang", "erlang"], np.array([0.5, 0.5])), (["lognormal", "lognormal"], np.array([2.0, 2.0])), (["deterministic", "deterministic"], np.array([0.0, 0.0]))):
    lam7, mu7 = np.array([0.4, 0.25]), np.array([1.0, 0.5]); R7 = mu7[None, :].copy(); sc7 = 1.0 / R7
    c2s = np.ones(2) if spec is None else np.array([cv2(n, c) for n, c in zip(*spec)]); ES2 = (1 / mu7) ** 2 * (1 + c2s)
    sig = np.cumsum(lam7 / mu7); s0 = np.r_[0.0, sig[:-1]]; W = (lam7 * ES2).sum() / 2 / ((1 - s0) * (1 - sig))
    m, hw = reps(lam7, R7, sc7, service=spec, n=140 if spec is not None and spec[0][0] == "lognormal" else 70, seed0=50_000, preemptive=False, nreq=500_000, warm=20_000)
    ok.append(report(f"7. M/G/1 non-preemptive priorities, {'exponential' if spec is None else spec[0][0]}", m, hw, lam7 * (W + 1 / mu7)))
# 8. non-preemptive M/M/c, equal repair rates, three classes: W_k = C(c, a) / (c mu) / ((1 - s_{k-1})(1 - s_k)), s with rho = lam / (c mu)
from math import factorial
lam8, mu8, c8 = np.array([0.5, 0.6, 0.7]), 1.0, 2; a = lam8.sum() / mu8; rho = a / c8
erl = (a ** c8 / factorial(c8) / (1 - rho)) / (sum(a ** k / factorial(k) for k in range(c8)) + a ** c8 / factorial(c8) / (1 - rho))
sig = np.cumsum(lam8 / (c8 * mu8)); s0 = np.r_[0.0, sig[:-1]]; W8 = erl / (c8 * mu8) / ((1 - s0) * (1 - sig))
R8 = np.ones((c8, 3)) * mu8; sc8 = np.tile(np.array([1.0, 2.0, 3.0]), (c8, 1))
m, hw = reps(lam8, R8, sc8, n=70, seed0=50_000, preemptive=False, nreq=500_000, warm=20_000); ok.append(report("8. M/M/2 non-preemptive priorities, 3 classes", m, hw, lam8 * (W8 + 1 / mu8)))
# 9. FCFS is unaffected by the switch (equal scores never interrupt): same seed, identical output
o1 = sf.simulate(lam8, R8, None, 50_000, 0, 7, MAXQ, preemptive=False); o2 = sf.simulate(lam8, R8, None, 50_000, 0, 7, MAXQ, arrival=("exponential", 1.0))
same = np.allclose(o1["dist"], o2["dist"]) and o1["events"] == 0 and o2["events"] == 0; print(f"9. FCFS identical with and without the switch, no interruptions: {same}"); ok.append(bool(same))
# 10. non-preemptive, several technicians, UNEQUAL repair rates: exact two-group solver (exact_nonpreemptive2.py)
for lH, mH, lL, mL, c in ((0.6, 1.0, 0.5, 0.5, 2), (0.9, 1.0, 0.9, 0.5, 3), (0.5, 0.5, 1.2, 1.5, 2)):
    r = nonpreemptive2_exact(lH, mH, lL, mL, c); ex = np.array([(x * np.arange(len(x))).sum() for x in (r["high"], r["low"])])
    Rn = np.ones((c, 2)) * np.array([mH, mL]); scn = np.tile(np.array([1.0, 2.0]), (c, 1))
    m, hw = reps(np.array([lH, lL]), Rn, scn, n=70, seed0=50_000, preemptive=False, nreq=500_000, warm=20_000)
    ok.append(report(f"10. M/M/{c} non-preemptive, muH {mH} muL {mL} (exact solver)", m, hw, ex))
print(f"\n{sum(ok)}/{len(ok)} checks passed in {time.time() - t0:.0f} s"); sys.exit(0 if all(ok) else 1)
