"""
Exact analysis of the multi-class M/M/c queue with FCFS service and class-dependent
exponential service rates (full cross-training), as used for validating the simulator.

Key property: under FCFS a waiting part's class plays no role until it enters service,
so, given the process of the numbers in service, the classes of the waiting parts are
i.i.d. with p_i = lambda_i / Lambda.  The system is therefore an M/H_N/c queue: a
level-independent QBD above level c whose phase is the vector k = (k_1..k_N) of the
numbers of busy servers per class.  Per-class marginals follow by adding a
Binomial(n - c, p_i) number of waiting class-i parts to k_i.
"""
import numpy as np
from itertools import combinations_with_replacement
from scipy.stats import binom


def compositions(total, n_classes):
    """All vectors k in Z_+^N with sum(k) = total, in a fixed order."""
    out = []
    for combo in combinations_with_replacement(range(n_classes), total):
        k = [0] * n_classes
        for i in combo:
            k[i] += 1
        out.append(tuple(k))
    return out


def solve_R(A0, A1, A2, tol=1e-14, max_iter=200000):
    """Minimal nonnegative solution of A0 + R A1 + R^2 A2 = 0 (iteration R = A0 (-A1 - R A2)^-1)."""
    R = np.zeros_like(A0)
    for it in range(max_iter):
        R_new = A0 @ np.linalg.inv(-A1 - R @ A2)
        if np.max(np.abs(R_new - R)) < tol:
            return R_new
        R = R_new
    raise RuntimeError("R did not converge")


def fcfs_multiclass_mmc(lam, mu, c, tail_mass=1e-13, qmax=None):
    """
    Exact stationary per-class marginal distributions of the number of parts
    (in repair or waiting) for a multi-class M/M/c FCFS queue with full cross-training.

    Returns dict with 'marginals' (N x (qmax+1) array), 'mean' per class,
    'total' distribution, and diagnostics.
    """
    lam = np.asarray(lam, float); mu = np.asarray(mu, float)
    N = len(lam); Lam = lam.sum(); p = lam / Lam
    load = (lam / mu).sum()
    if load >= c:
        raise ValueError(f"unstable: offered load {load:.4f} >= c = {c}")

    S = [compositions(n, N) for n in range(c + 1)]          # phases per level 0..c
    idx = [{k: j for j, k in enumerate(Sn)} for Sn in S]

    def local(n):
        Sn = S[min(n, c)]
        return np.diag([-(Lam + sum(k[i] * mu[i] for i in range(N))) for k in Sn])

    def up(n):          # n < c : arrival starts service immediately
        U = np.zeros((len(S[n]), len(S[n + 1])))
        for a, k in enumerate(S[n]):
            for i in range(N):
                kk = list(k); kk[i] += 1
                U[a, idx[n + 1][tuple(kk)]] += lam[i]
        return U

    def down(n):        # 1 <= n <= c : a completion frees a server
        D = np.zeros((len(S[n]), len(S[n - 1])))
        for a, k in enumerate(S[n]):
            for i in range(N):
                if k[i] > 0:
                    kk = list(k); kk[i] -= 1
                    D[a, idx[n - 1][tuple(kk)]] += k[i] * mu[i]
        return D

    m = len(S[c])
    A0 = Lam * np.eye(m)
    A1 = local(c)
    A2 = np.zeros((m, m))                                    # completion, next waiting part enters
    for a, k in enumerate(S[c]):
        for i in range(N):
            if k[i] > 0:
                for j in range(N):
                    kk = list(k); kk[i] -= 1; kk[j] += 1
                    A2[a, idx[c][tuple(kk)]] += k[i] * mu[i] * p[j]
    R = solve_R(A0, A1, A2)

    # boundary equations for levels 0..c (stacked), plus normalization
    sizes = [len(S[n]) for n in range(c + 1)]
    off = np.concatenate([[0], np.cumsum(sizes)])
    T = off[-1]
    G = np.zeros((T, T))
    for n in range(c + 1):
        r = slice(off[n], off[n + 1])
        G[r, r] = local(n) if n < c else A1 + R @ A2
        if n < c:
            G[r, off[n + 1]:off[n + 2]] = up(n)
        if n >= 1:
            G[r, off[n - 1]:off[n]] = down(n)
    # pi G = 0 ; replace one column by normalization weights
    w = np.ones(T)
    w[off[c]:] = np.linalg.solve(np.eye(m) - R, np.ones(m))   # levels >= c
    M = G.copy(); M[:, 0] = w
    rhs = np.zeros(T); rhs[0] = 1.0
    x = np.linalg.solve(M.T, rhs)
    pis = [x[off[n]:off[n + 1]] for n in range(c + 1)]

    # iterate levels above c until the remaining mass is negligible
    levels = list(range(c + 1)); pi_levels = pis[:]
    cur = pis[c]; n = c; remaining = 1.0 - sum(v.sum() for v in pis)
    while remaining > tail_mass and n < 100000:
        cur = cur @ R; n += 1
        pi_levels.append(cur); levels.append(n); remaining -= cur.sum()

    nmax = levels[-1]
    Q = qmax if qmax is not None else nmax
    marg = np.zeros((N, Q + 1)); total = np.zeros(nmax + 1)
    for n, vec in zip(levels, pi_levels):
        phases = S[min(n, c)]; wq = max(n - c, 0)
        total[n] += vec.sum()
        for i in range(N):
            bpmf = binom.pmf(np.arange(wq + 1), wq, p[i]) if wq > 0 else np.array([1.0])
            for a, k in enumerate(phases):
                if vec[a] == 0:
                    continue
                lo = k[i]; hi = min(k[i] + wq, Q)
                if lo <= Q:
                    marg[i, lo:hi + 1] += vec[a] * bpmf[: hi - lo + 1]
    mean = (marg * np.arange(Q + 1)).sum(axis=1)
    return {"marginals": marg, "mean": mean, "total": total,
            "spectral_radius_R": max(abs(np.linalg.eigvals(R))),
            "mass_captured": total.sum(), "load": load, "rho": load / c}
