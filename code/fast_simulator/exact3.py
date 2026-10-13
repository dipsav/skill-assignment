import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
def three_level_exact(lam, mu, c, K=(20, 40, 100)):
    """Exact (truncated) distribution for three preemptive priority levels, full cross-training, c technicians."""
    K1, K2, K3 = K; S = (K1+1)*(K2+1)*(K3+1); idx = lambda a, b, d: (a*(K2+1) + b)*(K3+1) + d
    rows, cols, vals = [], [], []
    for a in range(K1+1):
        b1 = min(a, c)
        for b in range(K2+1):
            b2 = min(b, c - b1)
            for d in range(K3+1):
                b3 = min(d, c - b1 - b2); s = idx(a, b, d)
                for (cond, t, r) in ((a < K1, idx(a+1, b, d), lam[0]), (b < K2, idx(a, b+1, d), lam[1]), (d < K3, idx(a, b, d+1), lam[2]),
                                     (b1 > 0, idx(a-1, b, d), b1*mu[0]), (b2 > 0, idx(a, b-1, d), b2*mu[1]), (b3 > 0, idx(a, b, d-1), b3*mu[2])):
                    if cond: rows.append(s); cols.append(t); vals.append(r)
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S)); Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    A = Q.T.tolil(); A[0, :] = 1.0; rhs = np.zeros(S); rhs[0] = 1.0
    P = spla.spsolve(A.tocsc(), rhs).reshape(K1+1, K2+1, K3+1)
    marg = [P.sum(axis=(1, 2)), P.sum(axis=(0, 2)), P.sum(axis=(0, 1))]
    edges = [marg[0][-1], marg[1][-1], marg[2][-1]]
    return [(m*np.arange(len(m))).sum() for m in marg], edges
