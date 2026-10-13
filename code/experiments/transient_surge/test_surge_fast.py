"""
test_surge_fast.py -- verification and validation of the surge engines (surge_tools.surge_fast, compiled, and
surge_tools.surge_simulation, the original Python engine).

  1. surge_fast with surgeMultiplier = 1 reproduces simulation_fast (validated in Appendix B) draw for draw:
     number-in-system distributions and utilizations equal to rounding, for a partial design under FCFS and the mu rule.
  2. Exact transient check: with many fully cross-trained technicians the shop is an M(t)/M/inf system, so the mean
     number in repair of SKU k solves m' = lambda_k(t) - mu_k m; its bin averages are known in closed form through
     warm-up, baseline, surge and recovery. Both engines must cover it (95% intervals, per bin and per phase).
  3. Backorders: with stock S, the integrated backorders equal E[(N - S)+] computed from the same run's
     number-in-system distribution (post-warm-up), and the bin trajectory of backorders averages to the phase means.
  4. The two engines agree on a test-bed design where their rules coincide in law (full cross-training, FCFS:
     identical technicians), and the size of the difference is reported where the rules differ (partial design,
     mu rule: idle-technician choice and interruption ties).
  5. Reproducibility: same seed, identical output.

    python3 test_surge_fast.py            (about 10 minutes on 2 cores; QUICK=1 skips the slow Python-engine parts)
"""
import os, sys, json, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from simulation_codes import simulation_fast as sf
from surge_tools import surge_fast as fs
from surge_tools import surge_simulation as ss

QUICK = os.environ.get("QUICK", "0") == "1"
PY_CORES = int(os.environ.get("PY_CORES", "2"))
CASES = os.path.join(HERE, "test_designs")
ok = []


def load(c):
    d = json.load(open(os.path.join(CASES, f"fullRangeResultsPSO_new_2019_case_{c:04d}.json")))
    return d, np.array(d["failure_rates"]), np.array(d["service_rates"]), np.array(d["best_assignment"], float)


def report(name, passed, detail):
    ok.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


# ---------------------------------------------------------------- 1. multiplier 1 == simulation_fast
for c in (1, 128):
    d, lam, mu, X = load(c)
    for rule, pr in (("FCFS", None), ("mu rule", 1 / mu)):
        R = X * mu
        score = None if pr is None else np.where(R > 0, np.round(np.broadcast_to(pr, R.shape), 6), np.inf)
        Lam = lam.sum(); cdf = np.cumsum(lam / Lam); cdf[-1] = 1
        for warm in (0, 10_000):
            a = sf.simulate(lam, R, score, n_requests=100_000, warmup=warm, seed=777, maxq=500)
            b = fs.simulation(Lam, cdf, R, R * (1.0 if pr is None else pr[None, :]), surgeMultiplier=1.0, seed=777,
                              maxQueue=500, numberWarmingUpRequests=warm)
            dp = np.abs(a["dist"] - b["number_in_system_pmf"]).max()
            du = np.abs(a["util"] - b["utilization"]).max() if warm == 0 else 0.0
            report(f"1. multiplier 1 vs simulation_fast, case {c}, {rule}, warm-up {warm}", dp < 1e-12 and du < 1e-12,
                   f"max |pmf diff| {dp:.1e}" + (f", max |util diff| {du:.1e}" if warm == 0 else ""))


# ---------------------------------------------------------------- 2. exact M(t)/M/inf transient
def exact_bins(lam, mu, edges_t, phase_edges_t, gamma):
    """Bin averages of m_k(t), m' = lam_k g(t) - mu_k m, m(0) = 0, for every SKU."""
    pts = np.unique(np.r_[edges_t, phase_edges_t])
    def g(t):
        return gamma if phase_edges_t[2] <= t < phase_edges_t[3] else 1.0
    m = np.zeros_like(lam); t0 = 0.0; area = {}
    allpts = np.unique(np.r_[0.0, pts])
    for u, v in zip(allpts[:-1], allpts[1:]):
        a = lam * g(u) / mu
        e = np.exp(-mu * (v - u))
        area[(u, v)] = a * (v - u) + (m - a) * (1 - e) / mu
        m = a + (m - a) * e
    out = np.zeros((len(edges_t) - 1, len(lam)))
    for i, (u, v) in enumerate(zip(edges_t[:-1], edges_t[1:])):
        out[i] = sum(ar for (x, y), ar in area.items() if x >= u and y <= v) / (v - u)
    return out


lam = np.array([0.5, 0.3, 0.2]); mu = np.array([0.05, 0.1, 0.2]); M = 150; gamma = 2.0
R = np.ones((M, 3)) * mu; Lam = lam.sum(); cdf = np.cumsum(lam / Lam); cdf[-1] = 1
kw = dict(expectedTotalNumberRequests=50_000, numberWarmingUpRequests=5_000, numberBaselineRequests=10_000,
          numberSurgeRequests=5_000, surgeMultiplier=gamma, sampleEveryArrivals=500.0, queueReportEveryArrivals=5_000.0)
sched = ss.make_surge_schedule(Lam, 50_000, 5_000, 10_000, 5_000, gamma, 500.0, queueReportEveryArrivals=5_000.0)
exact = exact_bins(lam, mu, sched["bin_edges_x"] / Lam, sched["phase_edges_x"] / Lam, gamma)


def check_transient(name, runs):
    sys_ = np.stack([r["number_in_system_by_skill"] for r in runs])          # rep, bin, sku
    n = sys_.shape[0]; mean = sys_.mean(0); sem = sys_.std(0, ddof=1) / np.sqrt(n)
    z = (mean - exact) / sem; tcrit = stats.t.ppf(0.975, n - 1)
    cover = np.mean(np.abs(z) <= tcrit)
    tot = sys_.sum(2); zt = (tot.mean(0) - exact.sum(1)) / (tot.std(0, ddof=1) / np.sqrt(n))
    # phase means of the total
    ph = sched["bin_phases"]; res = []
    for p in (1, 2, 3):
        w = np.diff(sched["bin_edges_x"])[ph == p]
        sim = (tot[:, ph == p] * w).sum(1) / w.sum(); ex = (exact.sum(1)[ph == p] * w).sum() / w.sum()
        res.append((ss.PHASES[p], sim.mean(), ex, stats.ttest_1samp(sim, ex).pvalue))
    passed = 0.90 <= cover and min(r[3] for r in res) > 0.001
    report(f"2. M(t)/M/inf transient, {name}", passed,
           f"{n} reps; per-bin per-SKU coverage of the exact mean {cover:.1%} (nominal 95%); "
           + "; ".join(f"{p} {s:.3f} vs exact {e:.3f} (p = {pv:.2f})" for p, s, e, pv in res))


runs_fast = [fs.simulation(Lam, cdf, R, R, seed=1000 + 20 * i, maxQueue=500, **kw) for i in range(35)]
check_transient("compiled engine", runs_fast)
if not QUICK:
    t = time.time()
    b = ss.simulation_batch_run(lam, mu, np.ones((M, 3)), priorities=None, replications=12, nCores=PY_CORES,
                                startSeed=5, collectWaitingTimeStatisticsInput=False, keepReplicationResults=True,
                                maxQueue=500, **kw)
    check_transient(f"Python engine ({time.time() - t:.0f} s)", b["raw_replications"])


# ---------------------------------------------------------------- 3. backorders
d, lam, mu, X = load(1)
S = np.array(d["stocks"]); Lam = lam.sum(); cdf = np.cumsum(lam / Lam); cdf[-1] = 1; R = X * mu
r = fs.simulation(Lam, cdf, R, R / mu[None, :], surgeMultiplier=1.5, seed=31, maxQueue=2000, stocks=S)
pmf = r["number_in_system_pmf"]; n = np.arange(pmf.shape[1])
ebo_pmf = (pmf * np.maximum(n[None, :] - S[:, None], 0)).sum(1)
dur = np.array([r["phases"][p]["duration"] for p in ss.PHASES[1:]])
ebo_phase = sum(r["phases"][p]["mean_backorders_by_skill"] * w for p, w in zip(ss.PHASES[1:], dur)) / dur.sum()
w = np.diff(r["normalized_time_edges"])
ebo_bins = (r["backorders_by_skill"] * w[:, None]).sum(0) / w.sum()
e1, e2 = np.abs(ebo_pmf - ebo_phase).max(), np.abs(ebo_bins - ebo_phase).max()
report("3. backorders: integrated vs from the distribution vs from the bins", max(e1, e2) < 1e-9 and r["number_in_system_overflow"].max() == 0,
       f"max differences {e1:.1e}, {e2:.1e}; post-warm-up EBO {ebo_phase.sum():.3f}; surge peak {r['phases']['surge']['max_backorders_total']}")


# ---------------------------------------------------------------- 4. compiled vs Python engine on a test-bed design
def compare(name, X, pr, reps_fast=200, reps_py=20, expect_equal=True):
    R = X * mu
    common = dict(surgeMultiplier=1.5, maxQueue=500, collectWaitingTimeStatisticsInput=False)
    t = time.time()
    f = fs.simulation_batch_run(lam, mu, X, priorities=pr, replications=reps_fast, startSeed=50_000, **common)
    tf = time.time() - t
    t = time.time()
    p = ss.simulation_batch_run(lam, mu, X, priorities=pr, replications=reps_py, nCores=PY_CORES, startSeed=7, **common)
    tp = time.time() - t
    rows = []
    for label, get in (("baseline queue", lambda b: b["baseline"]["queue_total"]["replications"]),
                       ("surge queue", lambda b: b["phases"]["surge"]["mean_queue_total"]["replications"]),
                       ("recovery queue", lambda b: b["phases"]["recovery"]["mean_queue_total"]["replications"]),
                       ("surge peak", lambda b: b["phases"]["surge"]["max_queue_total"]["replications"]),
                       ("baseline system", lambda b: b["phases"]["baseline"]["mean_number_in_system_by_skill"]["replications"].sum(1))):
        a, c = np.asarray(get(f), float), np.asarray(get(p), float)
        rows.append((label, a.mean(), c.mean(), stats.ttest_ind(a, c, equal_var=False).pvalue))
    rf, rp = f["recovery"]["recovered_fraction"], p["recovery"]["recovered_fraction"]
    detail = "; ".join(f"{l} {a:.3f} vs {c:.3f} (p = {pv:.2f})" for l, a, c, pv in rows) + \
             f"; recovered {rf:.0%} vs {rp:.0%}; time per replication {tf / reps_fast:.2f} s vs {tp * PY_CORES / reps_py:.1f} s"
    if expect_equal:
        report(f"4. compiled vs Python engine, {name}", min(r[3] for r in rows) > 0.001, detail)
    else:
        print(f"INFO  4. compiled vs Python engine, {name} (rules differ): {detail}", flush=True)


if not QUICK:
    compare("case 1, full cross-training, FCFS", np.ones_like(X), None)
    compare("case 1, PSO design, mu rule", X, 1 / mu, expect_equal=False)


# ---------------------------------------------------------------- 5. reproducibility
a = fs.simulation(Lam, cdf, R, R, surgeMultiplier=2.0, seed=99, stocks=S)
b = fs.simulation(Lam, cdf, R, R, surgeMultiplier=2.0, seed=99, stocks=S)
report("5. reproducibility", np.array_equal(a["backorders_by_skill"], b["backorders_by_skill"])
       and np.array_equal(a["queue_by_skill"], b["queue_by_skill"]), "same seed, identical trajectories")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
