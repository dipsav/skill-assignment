"""ctypes wrapper of simfast.c with the signature and outputs of simulation_fast.simulate (benchmark only)."""
import ctypes, os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, "libsimfast.so"))
_d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
_lib.run.restype = ctypes.c_int
_lib.run.argtypes = [_d, _d, _d, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_uint, ctypes.c_int,
                     _d, ctypes.c_int, ctypes.c_long, _d, _d, _d, _d, _d, _d, _d]


def simulate(lam, R, score=None, n_requests=100_000, warmup=0, seed=10, maxq=2000, buckets=(0.0,)):
    lam = np.ascontiguousarray(lam, float); R = np.ascontiguousarray(R, float); M, N = R.shape
    if score is None:
        score = np.where(R > 0, 1.0, np.inf)
    score = np.ascontiguousarray(np.where(R > 0, np.asarray(score, float), np.inf))
    b = np.ascontiguousarray(buckets, float); p = lam / lam.sum()
    cap = int(1.3 * p.max() * n_requests) + 2000
    dist = np.zeros((N, maxq)); over = np.zeros(N); util = np.zeros((M, N)); starts = np.zeros((M, N))
    arrivals = np.zeros(N); nowait = np.zeros(N); whist = np.zeros((N, len(b)))
    ovf = _lib.run(lam, R, score, N, M, float(n_requests), float(warmup), int(seed) & 0xffffffff, int(maxq), b, len(b), cap,
                   dist, over, util, starts, arrivals, nowait, whist)
    if ovf:
        raise RuntimeError("queue capacity exceeded")
    return {"dist": dist, "overflow": over, "util": util, "starts": starts, "arrivals": arrivals,
            "p_nowait": nowait / np.maximum(arrivals, 1), "wait_hist": whist / np.maximum(arrivals, 1)[:, None]}
