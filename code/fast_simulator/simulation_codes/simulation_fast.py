"""
simulation_fast.py -- compiled (Numba) version of the repair-shop simulator, with the rules of
simulation_pso_paper_v2.py. 3 October 2026.

Model (same as v2):
  - Poisson failures, rate lambda_k per SKU; repair of SKU k by technician j: exponential, rate R[j, k]
    (R[j, k] = 0: technician j lacks the skill).
  - Priority score of SKU k at technician j: cost / rate (as in the Python simulators); equal scores = FCFS.
    Lower score = higher priority.
  - Arrival: an idle qualified technician takes the part (the one idle longest; ties: lowest index). Otherwise, among
    qualified technicians repairing a part of strictly lower priority, the one repairing the lowest-priority part is
    interrupted (ties: the most recently failed part); the interrupted part returns to the head of its queue and is
    dispatched again by the same rule (cascade).
  - Completion: the technician takes the waiting part of highest priority it is qualified for (ties: earliest
    failure). A technician always starts the head of the chosen SKU's queue (FCFS within a SKU).
  - Restart semantics: a (re)started repair draws a fresh exponential time.
  - Horizon: expectedTotalNumberRequests / Lambda; statistics after numberWarmingUpRequests / Lambda.
Differences from the Python simulators (statistically immaterial, all improvements):
  - statistics are flushed to the end of the horizon (the Python versions stop at each SKU's last event);
  - utilization includes repairs in progress at the end and service before interruptions;
  - counts beyond maxQueue are kept in an overflow share instead of being dropped, so truncation can be detected;
  - waiting times are recorded when a part first starts repair.
Random numbers come from Numba's generator, seeded per replication: results are reproducible for a given seed,
but not identical to the Python simulators' streams.

Also here: checkSkillAssignment, the feasibility check (formerly checkSkillAssignment_scipy.py), used by the interface
before its cache lookup and by the comparison and screening scripts. The simulator itself does not call it (it must
stay able to simulate transiently overloaded shops); simulate() only refuses SKUs that no technician can repair.
"""
import numpy as np
from numba import njit
from scipy.optimize import linprog

_INF = np.inf


@njit(cache=True)
def _stat(k, t, count, last, dist, over, warm, maxq):
    if t > warm:
        start = last[k] if last[k] > warm else warm
        c = count[k]
        if c < maxq:
            dist[k, c] += t - start
        else:
            over[k] += t - start
    last[k] = t


@njit(cache=True)
def _start(j, k, t, R, qa, qf, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait):
    h = qhead[k]
    a = qa[k, h]
    f = qf[k, h]
    qhead[k] = (h + 1) % cap
    qlen[k] -= 1
    if f == 0:                                         # first start of this part: record its waiting time
        w = t - a
        if w <= 0.0:
            nowait[k] += 1.0
        for b in range(buckets.shape[0]):
            if w <= buckets[b]:
                whist[k, b] += 1.0
    busy_k[j] = k
    busy_a[j] = a
    busy_since[j] = t
    comp[j] = t + np.random.exponential(1.0 / R[j, k])
    starts[j, k] += 1.0


@njit(cache=True)
def _run(lam, R, score, n_requests, warm_requests, seed, maxq, buckets, cap, check_inv, preempt):
    np.random.seed(seed)
    N = lam.shape[0]
    M = R.shape[0]
    Lam = lam.sum()
    cum = np.cumsum(lam / Lam)
    stop = n_requests / Lam
    warm = warm_requests / Lam
    qa = np.empty((N, cap))
    qf = np.zeros((N, cap), np.int8)
    qhead = np.zeros(N, np.int64)
    qlen = np.zeros(N, np.int64)
    count = np.zeros(N, np.int64)
    last = np.zeros(N)
    dist = np.zeros((N, maxq))
    over = np.zeros(N)
    busy_k = -np.ones(M, np.int64)
    busy_a = np.zeros(M)
    busy_since = np.zeros(M)
    comp = np.full(M, _INF)
    idle_since = np.zeros(M)
    util = np.zeros((M, N))
    starts = np.zeros((M, N))
    arrivals = np.zeros(N)
    nowait = np.zeros(N)
    whist = np.zeros((N, buckets.shape[0]))
    overflow_queue = False
    n_events = 0
    n_violations = 0

    t = 0.0
    next_arr = np.random.exponential(1.0 / Lam)
    while True:
        jmin = -1
        tc = _INF
        for j in range(M):
            if comp[j] < tc:
                tc = comp[j]
                jmin = j
        t_next = next_arr if next_arr <= tc else tc
        if t_next >= stop:
            break
        t = t_next
        if next_arr <= tc:                                            # ---------------- arrival
            u = np.random.random()
            k = 0
            while k < N - 1 and cum[k] <= u:
                k += 1
            _stat(k, t, count, last, dist, over, warm, maxq)
            count[k] += 1
            arrivals[k] += 1.0
            if qlen[k] >= cap:
                overflow_queue = True
                break
            idx = (qhead[k] + qlen[k]) % cap
            qa[k, idx] = t
            qf[k, idx] = 0
            qlen[k] += 1
            kk = k
            while kk >= 0:                                            # dispatch, with cascade
                jbest = -1
                best_idle = _INF
                for j in range(M):
                    if R[j, kk] > 0.0 and busy_k[j] < 0 and idle_since[j] < best_idle:
                        best_idle = idle_since[j]
                        jbest = j
                if jbest >= 0:
                    _start(jbest, kk, t, R, qa, qf, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
                    break
                if not preempt:                                       # non-preemptive priorities
                    break
                jv = -1
                vs = -_INF
                va = -_INF
                for j in range(M):
                    if R[j, kk] > 0.0 and busy_k[j] >= 0:
                        cs = score[j, busy_k[j]]
                        if score[j, kk] < cs and (cs > vs or (cs == vs and busy_a[j] > va)):
                            jv = j
                            vs = cs
                            va = busy_a[j]
                if jv < 0:
                    break
                v = busy_k[jv]
                util[jv, v] += t - busy_since[jv]                   # service delivered before the interruption
                h = (qhead[v] - 1) % cap                             # interrupted part back to the head of its queue
                qhead[v] = h
                qa[v, h] = busy_a[jv]
                qf[v, h] = 1
                qlen[v] += 1
                busy_k[jv] = -1
                comp[jv] = _INF
                _start(jv, kk, t, R, qa, qf, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
                kk = v                                                # the interrupted part looks for a technician
            next_arr = t + np.random.exponential(1.0 / Lam)
        else:                                                         # ---------------- completion
            j = jmin
            k = busy_k[j]
            util[j, k] += t - busy_since[j]
            _stat(k, t, count, last, dist, over, warm, maxq)
            count[k] -= 1
            busy_k[j] = -1
            comp[j] = _INF
            idle_since[j] = t
            best = -1
            bs = _INF
            ba = _INF
            for kk in range(N):
                if R[j, kk] > 0.0 and qlen[kk] > 0:
                    s = score[j, kk]
                    a = qa[kk, qhead[kk]]
                    if s < bs or (s == bs and a < ba):
                        best = kk
                        bs = s
                        ba = a
            if best >= 0:
                _start(j, best, t, R, qa, qf, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
        if check_inv:                                                 # debug: priority order after every event
            n_events += 1
            bad = False
            for k2 in range(N):
                if qlen[k2] > 0:
                    for j2 in range(M):
                        if R[j2, k2] > 0.0 and (busy_k[j2] < 0 or score[j2, busy_k[j2]] > score[j2, k2]):
                            bad = True
                            break
                if bad:
                    break
            if bad:
                n_violations += 1
    for k in range(N):                                                # flush to the end of the horizon
        _stat(k, stop, count, last, dist, over, warm, maxq)
    for j in range(M):
        if busy_k[j] >= 0:
            util[j, busy_k[j]] += stop - busy_since[j]
    span = stop - warm
    return dist / span, over / span, util / stop, starts, arrivals, nowait, whist, overflow_queue, n_events, n_violations


# ----------------------------------------------------------------------------------------------- general distributions
# Kernel for the sensitivity study (10 Oct 2026): per-SKU renewal failure streams and general repair-time distributions,
# preemptive RESUME with the remaining (unscaled) repair time stored with the part. Distribution codes (per SKU):
# 0 exponential, 1 Erlang-k (k = round(1 / cv^2)), 2 lognormal, 3 hyperexponential H2 with balanced means (cv >= 1),
# 4 deterministic. The repair time of a part is drawn at its failure with mean 1 (unscaled) and scaled by 1 / R[j, k] at
# the technician that repairs it; on interruption the remaining time is converted back to unscaled units. With
# exponential distributions the results are equal in law to _run (not draw for draw: the draw order differs).
# The dispatch rules are those of _run. Numba only (no C port).
@njit(cache=True)
def _draw(kind, mean, cv):
    if kind == 1:
        k = int(1.0 / (cv * cv) + 0.5)
        if k < 1:
            k = 1
        x = 0.0
        for _ in range(k):
            x += np.random.exponential(mean / k)
        return x
    if kind == 2:
        s2 = np.log(1.0 + cv * cv)
        return np.random.lognormal(np.log(mean) - 0.5 * s2, np.sqrt(s2))
    if kind == 3:
        c2 = cv * cv
        if c2 <= 1.0:
            return np.random.exponential(mean)
        p = 0.5 * (1.0 + np.sqrt((c2 - 1.0) / (c2 + 1.0)))
        if np.random.random() < p:
            return np.random.exponential(mean / (2.0 * p))
        return np.random.exponential(mean / (2.0 * (1.0 - p)))
    if kind == 4:
        return mean
    return np.random.exponential(mean)


@njit(cache=True)
def _start_g(j, k, t, R, qa, qf, qr, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait):
    h = qhead[k]
    a = qa[k, h]
    f = qf[k, h]
    r = qr[k, h]
    qhead[k] = (h + 1) % cap
    qlen[k] -= 1
    if f == 0:
        w = t - a
        if w <= 0.0:
            nowait[k] += 1.0
        for b in range(buckets.shape[0]):
            if w <= buckets[b]:
                whist[k, b] += 1.0
    busy_k[j] = k
    busy_a[j] = a
    busy_since[j] = t
    comp[j] = t + r / R[j, k]
    starts[j, k] += 1.0


@njit(cache=True)
def _run_general(lam, R, score, n_requests, warm_requests, seed, maxq, buckets, cap, adist, acv, sdist, scv, preempt):
    np.random.seed(seed)
    N = lam.shape[0]
    M = R.shape[0]
    Lam = lam.sum()
    stop = n_requests / Lam
    warm = warm_requests / Lam
    qa = np.empty((N, cap))
    qf = np.zeros((N, cap), np.int8)
    qr = np.empty((N, cap))
    qhead = np.zeros(N, np.int64)
    qlen = np.zeros(N, np.int64)
    count = np.zeros(N, np.int64)
    last = np.zeros(N)
    dist = np.zeros((N, maxq))
    over = np.zeros(N)
    busy_k = -np.ones(M, np.int64)
    busy_a = np.zeros(M)
    busy_since = np.zeros(M)
    comp = np.full(M, _INF)
    idle_since = np.zeros(M)
    util = np.zeros((M, N))
    starts = np.zeros((M, N))
    arrivals = np.zeros(N)
    nowait = np.zeros(N)
    whist = np.zeros((N, buckets.shape[0]))
    overflow_queue = False
    n_interrupt = 0
    next_arr = np.full(N, _INF)
    for k in range(N):
        if lam[k] > 0.0:
            next_arr[k] = _draw(adist[k], 1.0 / lam[k], acv[k])
    t = 0.0
    while True:
        jmin = -1
        tc = _INF
        for j in range(M):
            if comp[j] < tc:
                tc = comp[j]
                jmin = j
        ka = -1
        ta = _INF
        for k in range(N):
            if next_arr[k] < ta:
                ta = next_arr[k]
                ka = k
        t_next = ta if ta <= tc else tc
        if t_next >= stop:
            break
        t = t_next
        if ta <= tc:                                                  # ---------------- arrival of SKU ka
            k = ka
            next_arr[k] = t + _draw(adist[k], 1.0 / lam[k], acv[k])
            _stat(k, t, count, last, dist, over, warm, maxq)
            count[k] += 1
            arrivals[k] += 1.0
            if qlen[k] >= cap:
                overflow_queue = True
                break
            idx = (qhead[k] + qlen[k]) % cap
            qa[k, idx] = t
            qf[k, idx] = 0
            qr[k, idx] = _draw(sdist[k], 1.0, scv[k])                 # unscaled repair time, drawn at the failure
            qlen[k] += 1
            kk = k
            while kk >= 0:                                            # dispatch, with cascade
                jbest = -1
                best_idle = _INF
                for j in range(M):
                    if R[j, kk] > 0.0 and busy_k[j] < 0 and idle_since[j] < best_idle:
                        best_idle = idle_since[j]
                        jbest = j
                if jbest >= 0:
                    _start_g(jbest, kk, t, R, qa, qf, qr, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
                    break
                if not preempt:                                       # non-preemptive: the part waits for a completion
                    break
                jv = -1
                vs = -_INF
                va = -_INF
                for j in range(M):
                    if R[j, kk] > 0.0 and busy_k[j] >= 0:
                        cs = score[j, busy_k[j]]
                        if score[j, kk] < cs and (cs > vs or (cs == vs and busy_a[j] > va)):
                            jv = j
                            vs = cs
                            va = busy_a[j]
                if jv < 0:
                    break
                v = busy_k[jv]
                util[jv, v] += t - busy_since[jv]
                h = (qhead[v] - 1) % cap                             # interrupted part back to the head of its queue
                qhead[v] = h
                qa[v, h] = busy_a[jv]
                qf[v, h] = 1
                qr[v, h] = (comp[jv] - t) * R[jv, v]                 # remaining time, unscaled (resume)
                n_interrupt += 1
                qlen[v] += 1
                busy_k[jv] = -1
                comp[jv] = _INF
                _start_g(jv, kk, t, R, qa, qf, qr, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
                kk = v
        else:                                                         # ---------------- completion
            j = jmin
            k = busy_k[j]
            util[j, k] += t - busy_since[j]
            _stat(k, t, count, last, dist, over, warm, maxq)
            count[k] -= 1
            busy_k[j] = -1
            comp[j] = _INF
            idle_since[j] = t
            best = -1
            bs = _INF
            ba = _INF
            for kk in range(N):
                if R[j, kk] > 0.0 and qlen[kk] > 0:
                    s = score[j, kk]
                    a = qa[kk, qhead[kk]]
                    if s < bs or (s == bs and a < ba):
                        best = kk
                        bs = s
                        ba = a
            if best >= 0:
                _start_g(j, best, t, R, qa, qf, qr, qhead, qlen, cap, busy_k, busy_a, busy_since, comp, starts, whist, buckets, nowait)
    for k in range(N):
        _stat(k, stop, count, last, dist, over, warm, maxq)
    for j in range(M):
        if busy_k[j] >= 0:
            util[j, busy_k[j]] += stop - busy_since[j]
    span = stop - warm
    return dist / span, over / span, util / stop, starts, arrivals, nowait, whist, overflow_queue, n_interrupt, 0


DIST_CODES = {"exponential": 0, "exp": 0, "erlang": 1, "lognormal": 2, "hyperexponential": 3, "h2": 3, "deterministic": 4}


def _dist_spec(spec, N):
    """None -> exponential; ("erlang", 2.0) for all SKUs; or (list of names, array of CVs) per SKU. Returns (codes, cvs)."""
    if spec is None:
        return np.zeros(N, np.int64), np.ones(N)
    names, cvs = spec
    if isinstance(names, str):
        names = [names] * N
    cvs = np.broadcast_to(np.asarray(cvs, float), (N,)).copy()
    return np.array([DIST_CODES[n.lower()] for n in names], np.int64), cvs


# ----------------------------------------------------------------------------------------------- C backend
# csim/simfast.c is a statement-for-statement C port of _run with Numba's random-number generator reproduced, so its
# output is identical (checked to rounding) and it is about 3 times faster. Used when the library has been built
# (gcc -O3 -march=native -shared -fPIC -o csim/libsimfast.so csim/simfast.c -lm); SIM_BACKEND=numba forces Numba.
# check_invariant=True always uses Numba (the C port does not count events).
import os as _os, ctypes as _ctypes
BACKEND = "numba"
_clib = None
_libpath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "csim", "libsimfast.so")
if _os.environ.get("SIM_BACKEND", "c") != "numba" and _os.path.exists(_libpath):
    try:
        _clib = _ctypes.CDLL(_libpath)
        _dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        _clib.run.restype = _ctypes.c_int
        _clib.run.argtypes = [_dp, _dp, _dp, _ctypes.c_int, _ctypes.c_int, _ctypes.c_double, _ctypes.c_double, _ctypes.c_uint,
                              _ctypes.c_int, _dp, _ctypes.c_int, _ctypes.c_long, _dp, _dp, _dp, _dp, _dp, _dp, _dp, _ctypes.c_int]
        BACKEND = "c"
    except OSError:
        _clib = None


def _run_c(lam, R, score, n_requests, warm_requests, seed, maxq, buckets, cap, preempt=True):
    N = lam.shape[0]; M = R.shape[0]; nb = buckets.shape[0]
    dist = np.zeros((N, maxq)); over = np.zeros(N); util = np.zeros((M, N)); starts = np.zeros((M, N))
    arrivals = np.zeros(N); nowait = np.zeros(N); whist = np.zeros((N, nb))
    ovf = _clib.run(np.ascontiguousarray(lam), np.ascontiguousarray(R), np.ascontiguousarray(score), N, M, n_requests,
                    warm_requests, seed & 0xffffffff, maxq, np.ascontiguousarray(buckets), nb, cap,
                    dist, over, util, starts, arrivals, nowait, whist, int(preempt))
    return dist, over, util, starts, arrivals, nowait, whist, bool(ovf), 0, 0


# ----------------------------------------------------------------------------------------------- Python interface
def simulate(lam, R, score=None, n_requests=100_000, warmup=0, seed=10, maxq=2000, buckets=(0.0,), check_invariant=False,
             arrival=None, service=None, preemptive=True):
    """One replication. lam [N]; R [M, N] repair rates (0 = no skill); score [M, N] priority scores
    (lower = higher priority; None = FCFS). Returns a dict of arrays.
    arrival / service: None (exponential, the published kernels) or a distribution spec (name, cv) for all SKUs or
    (names, cvs) per SKU -- names exponential, erlang, lognormal, hyperexponential (h2), deterministic; then the
    general kernel is used (preemptive resume with stored remaining time, per-SKU renewal failure streams).
    preemptive=False: non-preemptive priorities (an arriving part never interrupts a repair; the technician takes the
    highest-priority waiting part at a completion); with exponential times it runs in the fast kernels (C or _run, same
    draws as the preemptive run until the first would-be interruption), otherwise in the general kernel. With the
    general kernel the result's "events" field holds the number of interruptions.
    check_invariant=True (debug) counts the events after which a waiting part had a qualified technician that was idle
    or repairing a part of strictly lower priority (should be 0)."""
    lam = np.asarray(lam, float); R = np.asarray(R, float); M, N = R.shape
    unrepairable = np.nonzero((lam > 0) & ~(R > 0).any(axis=0))[0]
    if len(unrepairable):                     # structural check only; capacity (overload) is the caller's question
        raise ValueError(f"SKUs {unrepairable.tolist()} fail but no technician can repair them")
    if score is None:
        score = np.where(R > 0, 1.0, np.inf)
    score = np.where(R > 0, np.asarray(score, float), np.inf)
    p = lam / lam.sum()
    cap = int(1.3 * p.max() * n_requests) + 2000
    args = (lam, R, score, float(n_requests), float(warmup), int(seed), int(maxq), np.asarray(buckets, float), cap)
    if arrival is not None or service is not None:
        adist, acv = _dist_spec(arrival, N); sdist, scv = _dist_spec(service, N)
        dist, over, util, starts, arrivals, nowait, whist, overflow_queue, n_events, n_viol = _run_general(*args, adist, acv, sdist, scv, bool(preemptive))
    else:
        run = _run_c if (_clib is not None and not check_invariant) else _run
        dist, over, util, starts, arrivals, nowait, whist, overflow_queue, n_events, n_viol = (
            run(*args, bool(preemptive)) if run is _run_c else _run(*args, bool(check_invariant), bool(preemptive)))
    if overflow_queue:
        raise RuntimeError("queue capacity exceeded")
    return {"dist": dist, "overflow": over, "util": util, "starts": starts, "arrivals": arrivals,
            "p_nowait": nowait / np.maximum(arrivals, 1), "wait_hist": whist / np.maximum(arrivals, 1)[:, None],
            "events": int(n_events), "invariant_violations": int(n_viol)}


def scores_from_costs(R, C):
    """Priority score as in the Python simulators: round(cost / rate, 6) where the technician has the skill."""
    R = np.asarray(R, float); C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(R > 0, np.round(C / np.where(R > 0, R, 1.0), 6), np.inf)


def simulation(requestArrivalRate=5.0, skillDistribution=(), skillServerRates=(), skillServerCosts=(),
               histogramBucketsInput=(), expectedTotalNumberRequests=10000, numberWarmingUpRequests=1000, seed=10,
               collectWaitingTimeStatisticsInput=False, maxQueue=2000):
    """Drop-in for simulation() of the Python simulators: returns (distribution dict per SKU, waiting histogram,
    utilization [M, N], assignment statistics [M, N]). NOTE the default warm-up of 1,000 requests, as in the originals."""
    cum = np.asarray(skillDistribution, float)
    lam = requestArrivalRate * np.diff(np.concatenate([[0.0], cum]))
    R = np.asarray(skillServerRates, float)
    buckets = np.asarray(histogramBucketsInput if len(histogramBucketsInput) else [0.0], float)
    out = simulate(lam, R, scores_from_costs(R, skillServerCosts), expectedTotalNumberRequests,
                   numberWarmingUpRequests, seed, maxQueue, buckets)
    dist = {k: {q: float(v) for q, v in enumerate(out["dist"][k]) if v > 0} for k in range(len(lam))}
    assign = out["starts"] / np.maximum(out["starts"].sum(axis=0, keepdims=True), 1)
    return dist, out["wait_hist"], out["util"], assign


def simulation_batch_run(failureRates, serviceRates, skillServerAssignment, priorities=None, histogramBucketsInput=(0.0,),
                         expectedTotalNumberRequests=10000, numberWarmingUpRequests=1000, replications=2, maxQueue=10,
                         nCores=None, startSeed=10, arrival=None, service=None, preemptive=True):
    """Drop-in for simulation_batch_run (5 outputs, as the older simulator): averaged distribution [N, maxQueue],
    waiting histogram, utilization [M, N], assignment statistics, last replication's distribution.
    Replication i uses seed startSeed + 20 i. nCores is ignored (the compiled loop is fast enough serially)."""
    lam = np.asarray(failureRates, float); mu = np.asarray(serviceRates, float)
    X = np.asarray(skillServerAssignment, float); R = X * mu[None, :]
    score = None if priorities is None else np.where(X > 0, np.asarray(priorities, float)[None, :], np.inf)
    buckets = np.asarray(histogramBucketsInput, float)
    acc = None
    for i in range(replications):
        out = simulate(lam, R, score, expectedTotalNumberRequests, numberWarmingUpRequests, startSeed + 20 * i, maxQueue, buckets, arrival=arrival, service=service, preemptive=preemptive)
        cur = (out["dist"], out["wait_hist"], out["util"], out["starts"] / np.maximum(out["arrivals"], 1)[None, :])
        acc = [c.copy() for c in cur] if acc is None else [a + c for a, c in zip(acc, cur)]
    return acc[0] / replications, acc[1] / replications, acc[2] / replications, acc[3] / replications, cur[0]


# ----------------------------------------------------------------------------------------------- feasibility check
def checkSkillAssignment(assignement, lambdas, mus, cap=0.999):
    """Feasibility of a skill matrix (rows = technicians, columns = SKUs), as skill_set_generator.checkSkillAssignment
    but solved in-process with scipy (HiGHS) instead of PuLP/CBC: minimize maxRho subject to sum_j alpha_ij = 1 for
    every SKU, alpha_ij = 0 where the technician lacks the skill, every technician's load <= maxRho <= cap (0.999).
    Returns (feasible, alphas) with alphas[j, i] (technicians x SKUs), as the original."""
    X = np.asarray(assignement); n_servers, m_skills = X.shape
    assert m_skills == len(lambdas)
    if np.any(X.sum(axis=0) == 0):                                   # quick check: an SKU no technician can repair
        return False, None
    pairs = np.argwhere(X.T == 1)                                     # (i, j): SKU i, technician j with the skill
    k = len(pairs); load = np.asarray(lambdas, float) / np.asarray(mus, float)
    c = np.zeros(k + 1); c[-1] = 1.0                                  # variables: alphas, then maxRho
    A_eq = np.zeros((m_skills, k + 1)); A_eq[pairs[:, 0], np.arange(k)] = 1.0
    A_ub = np.zeros((n_servers, k + 1)); A_ub[pairs[:, 1], np.arange(k)] = load[pairs[:, 0]]; A_ub[:, -1] = -1.0
    bounds = [(0.0, 1.0)] * k + [(0.0, cap)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n_servers), A_eq=A_eq, b_eq=np.ones(m_skills), bounds=bounds, method="highs")
    if res.status != 0:
        return False, None
    alphas = np.zeros((n_servers, m_skills)); alphas[pairs[:, 1], pairs[:, 0]] = res.x[:k]
    return True, alphas
