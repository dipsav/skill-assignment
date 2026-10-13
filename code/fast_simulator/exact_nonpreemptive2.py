"""
Exact per-group distributions for two priority groups under a NON-preemptive priority rule (high group served first at
a completion, a repair is never interrupted; FCFS within a group), full cross-training, c technicians, one repair rate
per group. As exact_priority2.py, with one more state component: since a high part can no longer take a technician from a
low repair, the number of low parts in repair s must be tracked. State (h, l, s): h high parts present, l low parts
present, s low parts in repair; high parts in repair bh = min(h, c - s). Admissible states: s <= min(l, c), bh + s <= c,
and no part waits while a technician is idle (if bh + s < c then h = bh and l = s).
Transitions:
  high failure:  h + 1                      (starts at once if a technician is idle: bh grows by the formula)
  low failure:   l + 1, and s + 1 if a technician is idle
  high repair (rate bh muH): h - 1; the freed technician takes a waiting high part if any (bh unchanged), else a
                 waiting low part (s + 1) if any
  low repair  (rate s muL):  l - 1, s - 1; the freed technician takes a waiting high part if any (bh grows), else a
                 waiting low part (s unchanged)
Solved on a truncated grid enlarged until the probability at the edges is negligible.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla


def _states(c, NH, NL):
    st = []
    for h in range(NH + 1):
        for l in range(NL + 1):
            for s in range(min(l, c) + 1):
                bh = min(h, c - s)
                if bh + s < c and (h > bh or l > s):
                    continue                                   # a part waits while a technician is idle
                st.append((h, l, s))
    return st


def _solve(lamH, muH, lamL, muL, c, NH, NL):
    st = _states(c, NH, NL); index = {x: i for i, x in enumerate(st)}; rows, cols, vals = [], [], []
    def add(a, b, r):
        if r > 0 and b in index: rows.append(index[a]); cols.append(index[b]); vals.append(r)
    for (h, l, s) in st:
        bh = min(h, c - s); idle = c - bh - s
        if h < NH: add((h, l, s), (h + 1, l, s), lamH)
        if l < NL: add((h, l, s), (h, l + 1, s + 1 if idle > 0 else s), lamL)
        if bh > 0:                                             # high repair completes
            h2 = h - 1
            if h2 >= c - s: add((h, l, s), (h2, l, s), bh * muH)           # a waiting high part takes the technician
            elif l > s:     add((h, l, s), (h2, l, s + 1), bh * muH)       # else a waiting low part
            else:           add((h, l, s), (h2, l, s), bh * muH)
        if s > 0:                                              # low repair completes
            l2, s2 = l - 1, s - 1
            if h > bh:      add((h, l, s), (h, l2, s2), s * muL)           # a waiting high part takes it (bh = min(h, c - s2))
            elif l2 > s2:   add((h, l, s), (h, l2, s), s * muL)            # else the next low part
            else:           add((h, l, s), (h, l2, s2), s * muL)
    S = len(st); Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S)); Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    A = Q.T.tolil(); A[0, :] = 1.0; b = np.zeros(S); b[0] = 1.0
    p = spla.spsolve(A.tocsc(), b); P = np.zeros((NH + 1, NL + 1))
    for (h, l, s), v in zip(st, p): P[h, l] += v
    return P


def nonpreemptive2_exact(lamH, muH, lamL, muL, c, edge_tol=1e-12, NH=60, NL=200, max_states=6_000_000):
    assert lamH / muH + lamL / muL < c, "unstable"
    while True:
        P = _solve(lamH, muH, lamL, muL, c, NH, NL)
        eH, eL = P[-1, :].sum(), P[:, -1].sum()
        if eH < edge_tol and eL < edge_tol: break
        if eH >= edge_tol: NH = int(NH * 1.6)
        if eL >= edge_tol: NL = int(NL * 1.6)
        if (NH + 1) * (NL + 1) * (c + 1) > max_states: raise RuntimeError("truncation grid too large")
    return {"high": P.sum(axis=1), "low": P.sum(axis=0), "joint": P, "grid": (NH, NL), "edge_mass": (eH, eL)}


if __name__ == "__main__":
    from math import factorial
    mean = lambda x: float((x * np.arange(len(x))).sum())
    # 1. c = 1: Cobham (M/M/1 non-preemptive), unequal rates
    lH, mH, lL, mL = 0.4, 1.0, 0.25, 0.5; r = nonpreemptive2_exact(lH, mH, lL, mL, 1)
    W0 = lH * 2 / mH ** 2 / 2 + lL * 2 / mL ** 2 / 2; s1 = lH / mH; s2 = s1 + lL / mL
    print("c=1 Cobham:  ", round(mean(r["high"]), 6), round(lH * (W0 / (1 - s1) + 1 / mH), 6), "|", round(mean(r["low"]), 6), round(lL * (W0 / ((1 - s1) * (1 - s2)) + 1 / mL), 6))
    # 2. c = 2, equal rates: M/M/c non-preemptive priority formula
    lH, lL, m, c = 0.6, 0.8, 1.0, 2; a = (lH + lL) / m; rho = a / c
    erl = (a ** c / factorial(c) / (1 - rho)) / (sum(a ** k / factorial(k) for k in range(c)) + a ** c / factorial(c) / (1 - rho))
    W1 = erl / (c * m) / (1 - lH / (c * m)); W2 = erl / (c * m) / ((1 - lH / (c * m)) * (1 - rho)); r = nonpreemptive2_exact(lH, m, lL, m, c)
    print("c=2 equal µ: ", round(mean(r["high"]), 6), round(lH * (W1 + 1 / m), 6), "|", round(mean(r["low"]), 6), round(lL * (W2 + 1 / m), 6))
    # 3. total number in system with equal rates = M/M/c (priority does not change the total)
    tot = r["joint"].sum(); print("probabilities sum to", round(tot, 12))
