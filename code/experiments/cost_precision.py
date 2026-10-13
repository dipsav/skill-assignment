"""
cost_precision.py -- precision of the independent cost estimates of the re-optimized designs.

For every result file, the final design and full cross-training are simulated again with the replications of the
final evaluation (35 x 100,000 requests, seeds 1,000,000 + 20 i, no warm-up, maxQueue 2,000). With the stock levels
fixed at the reported ones, the cost of replication i is
    sum_k S_k h_k + b sum_k EBO_k,i(S_k) + technician and skill costs,
whose mean equals the reported total (EBO is linear in the distribution) -- checked below. The 95% half-width is
t_{0.975,34} * sd / sqrt(35). (It ignores that S was chosen from the same replications; at fixed S it is the usual interval.)

    python3 cost_precision.py ../PSO_results_rerun tables/pso_cost_precision.csv
"""
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import t as student_t
from simulation_codes import simulation_fast as sf

SEED, REPS, N_REQ, MAXQ = 1_000_000, 35, 100_000, 2000


def ebo(dist, S):
    """Expected backorders per SKU at base-stock levels S from queue-length distributions dist [N, maxQ]."""
    n = np.arange(dist.shape[1])
    return np.array([np.sum(dist[k] * np.maximum(n - S[k], 0)) for k in range(len(S))])


def replication_costs(lam, mu, h, b, X, S, fixed):
    R = X * mu[None, :]
    out = []
    for i in range(REPS):
        d = sf.simulate(lam, R, None, N_REQ, 0, SEED + 20 * i, MAXQ)["dist"]
        d = d / d.sum(axis=1, keepdims=True)
        out.append(np.sum(S * h) + b * ebo(d, S).sum() + fixed)
    return np.array(out)


def summary(c):
    hw = student_t.ppf(0.975, len(c) - 1) * c.std(ddof=1) / np.sqrt(len(c))
    return c.mean(), hw


if __name__ == "__main__":
    folder, out = sys.argv[1], sys.argv[2]
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*case_*.json"))):
        d = json.load(open(path))
        lam, mu, h = (np.asarray(d[k], float) for k in ("failure_rates", "service_rates", "holding_costs"))
        b, f, cs = d["penalty_cost"], d["machine_cost"], d["skill_cost"]
        X = np.asarray(d["best_assignment"], float); Xff = np.ones_like(X)
        row = {"case": int(d["caseID"].split()[-1])}
        for tag, XX, S, total in (("design", X, d["stocks"], d["total_cost"]), ("ff", Xff, d["stocks_ff"], d["total_cost_ff"])):
            fixed = f * (XX.sum(axis=1) > 0).sum() + cs * XX.sum()
            c = replication_costs(lam, mu, h, b, XX, np.asarray(S), fixed)
            m, hw = summary(c)
            row.update({f"{tag}_mean": m, f"{tag}_reported": total, f"{tag}_halfwidth": hw, f"{tag}_hw_rel": hw / m,
                        f"{tag}_hw_rel_variable": hw / (m - fixed)})        # relative to stock + backorder cost only
        rows.append(row)
        print(f"case {row['case']:4d}: design {row['design_mean']:,.0f} (reported {row['design_reported']:,.0f}) "
              f"+/- {row['design_hw_rel']:.2%} | full CT +/- {row['ff_hw_rel']:.2%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    dev = (df.design_mean / df.design_reported - 1).abs().max()
    print(f"\nlargest deviation of the recomputed mean from the reported total: {dev:.2e}")
    print(f"design half-width: mean {df.design_hw_rel.mean():.2%}, median {df.design_hw_rel.median():.2%}, max {df.design_hw_rel.max():.2%}")
