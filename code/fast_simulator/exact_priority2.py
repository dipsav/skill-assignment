"""
Exact per-group distributions for two priority groups under a preemptive priority rule
(high group preempts low; FCFS within a group), full cross-training, c technicians,
one repair rate per group. With exponential repair times the pair (n_H, n_L) is a Markov
chain: high parts in repair min(n_H, c), low parts in repair min(n_L, c - min(n_H, c)).
Solved on a truncated grid that is enlarged until the probability at the edges is negligible.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def _solve(lamH, muH, lamL, muL, c, NH, NL):
    idx = lambda h, l: h * (NL + 1) + l
    S = (NH + 1) * (NL + 1); rows, cols, vals = [], [], []
    for h in range(NH + 1):
        bh = min(h, c)
        for l in range(NL + 1):
            bl = min(l, c - bh); s = idx(h, l)
            if h < NH: rows.append(s); cols.append(idx(h + 1, l)); vals.append(lamH)
            if l < NL: rows.append(s); cols.append(idx(h, l + 1)); vals.append(lamL)
            if bh: rows.append(s); cols.append(idx(h - 1, l)); vals.append(bh * muH)
            if bl: rows.append(s); cols.append(idx(h, l - 1)); vals.append(bl * muL)
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
    Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    A = Q.T.tolil(); A[0, :] = 1.0; b = np.zeros(S); b[0] = 1.0
    return spla.spsolve(A.tocsc(), b).reshape(NH + 1, NL + 1)

def priority2_exact(lamH, muH, lamL, muL, c, edge_tol=1e-12, NH=60, NL=200, max_states=4_000_000):
    assert lamH / muH + lamL / muL < c, "unstable"
    while True:
        P = _solve(lamH, muH, lamL, muL, c, NH, NL)
        eH, eL = P[-1, :].sum(), P[:, -1].sum()
        if eH < edge_tol and eL < edge_tol:
            break
        if eH >= edge_tol: NH = int(NH * 1.6)
        if eL >= edge_tol: NL = int(NL * 1.6)
        if (NH + 1) * (NL + 1) > max_states:
            raise RuntimeError("truncation grid too large")
    return {"high": P.sum(axis=1), "low": P.sum(axis=0), "joint": P, "grid": (NH, NL), "edge_mass": (eH, eL)}
