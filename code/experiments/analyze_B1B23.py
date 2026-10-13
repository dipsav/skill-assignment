"""
analyze_B1B23.py -- Section 6.1 numbers from the permanent-surge tables (B1 one-SKU, B2 all-SKU, B3 correlated).

Reads tables/B1_*.csv and tables/B23_*.csv (sku_surges.py / multi_surges.py rows: one per design set x rule x
scenario) and prints, per arm (design set / rule) and surge level:
  share of scenarios stable (LP) and stable in simulation, cost increase with the stock fixed and with the stock
  re-optimized (median and mean over the stable scenarios, relative to the nominal total), and backorders.
Paired comparisons between the two carried designs (FCFS design / FCFS vs mu design / mu) on the common stable
scenarios, and the sequential arm (FCFS design / mu) for reference. Writes tables/B1_summary.csv, B23_summary.csv.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, "tables")
ARM = {("PSO_results_tuned_polished", "fcfs"): "FCFS design / FCFS", ("PSO_results_tuned_polished", "mu"): "FCFS design / mu (sequential)",
       ("PSO_results_tuned_rule_mu_polished", "mu"): "mu design / mu", ("PSO_results_tuned_rule_hmu_polished", "hmu"): "hmu design / hmu",
       ("PSO_results_fullCT", "fcfs"): "full CT / FCFS", ("PSO_results_fullCT_rule_mu", "mu"): "full CT / mu",
       ("PSO_results_tuned_rule_mu_np_polished", "mu"): "mu-NP design / mu", ("PSO_results_tuned_rule_hmu_np_polished", "hmu"): "hmu-NP design / hmu",
       ("PSO_results_tuned_rule_mu_cov2_polished", "mu"): "mu-cov2 design / mu", ("PSO_results_tuned_cov2_polished", "fcfs"): "FCFS-cov2 design / FCFS"}


def load(prefix):
    fs = [f for f in glob.glob(os.path.join(T, f"{prefix}_*.csv")) if not any(k in os.path.basename(f) for k in ("replications", "summary", "table"))]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "preemptive" not in d: d["preemptive"] = True
    d["preemptive"] = d.preemptive.fillna(True).astype(bool)
    d["arm"] = [ARM.get((a, b), f"{a}/{b}") + ("" if p else " NP") for a, b, p in zip(d.design_set, d.rule, d.preemptive)]
    for col in ("total_fixed", "total_reopt"):
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)
    d["inc_fixed"] = d.total_fixed / d.nominal_total - 1; d["inc_reopt"] = d.total_reopt / d.nominal_total - 1
    if "scenario" not in d: d["scenario"] = "one SKU"
    return d


def summary(d, keys):
    g = d.groupby(keys).agg(n=("instance", "size"), lp_stable=("lp_feasible", "mean"), sim_stable=("sim_stable", "mean"),
                            inc_fixed_med=("inc_fixed", "median"), inc_fixed_mean=("inc_fixed", "mean"), inc_reopt_med=("inc_reopt", "median"),
                            inc_reopt_mean=("inc_reopt", "mean"), bo_fixed=("backorders_fixed", lambda s: s.replace(np.inf, np.nan).mean())).reset_index()
    return g


def paired(d, keys):
    """FCFS design/FCFS vs mu design/mu on scenarios stable under both; the stock-fixed and re-optimized increases."""
    a, b, c = "FCFS design / FCFS", "mu design / mu", "FCFS design / mu (sequential)"
    idx = [k for k in ("instance", "sku", "scenario", "technician") if k in d and k != "technician"]
    for lv, sub in d.groupby(keys):
        p = sub.pivot_table(index=idx, columns="arm", values=["sim_stable", "inc_fixed", "inc_reopt"], aggfunc="first")
        st = p["sim_stable"]; n = len(st)
        if a not in st or b not in st: print(f"{lv}: arms missing, skipped"); continue
        line = f"{lv}: stable  FCFS/FCFS {st[a].mean():.1%}  mu/mu {st[b].mean():.1%}" + (f"  FCFS/mu {st[c].mean():.1%}" if c in st else "")
        both = st[a].astype(bool) & st[b].astype(bool)
        for v in ("inc_fixed", "inc_reopt"):
            x, y = p[v][a][both], p[v][b][both]
            diff = y - x; pv = wilcoxon(diff.dropna()).pvalue if diff.notna().sum() > 10 else np.nan
            line += f" | {v}: FCFS/FCFS med {x.median():.1%} mu/mu med {y.median():.1%}, mu lower in {(diff < 0).sum()}/{both.sum()} (p {pv:.1e})"
        print(line)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for prefix, keys in (("B1", ["arm", "delta"]), ("B23", ["arm", "scenario", "delta"])):
        try: d = load(prefix)
        except ValueError: print(f"{prefix}: no tables yet"); continue
        print(f"\n=== {prefix}: {len(d)} scenario rows, {d.instance.nunique()} instances, arms {sorted(d.arm.unique())}")
        g = summary(d, keys); print(g.round(3).to_string(index=False)); g.to_csv(os.path.join(T, f"{prefix}_summary.csv"), index=False)
        # by test-bed factor: share stable and median increase (stock fixed), FCFS design/FCFS vs mu design/mu
        for fac in ("rho", "M", "N"):
            print(d[d.arm.isin(["FCFS design / FCFS", "mu design / mu"])].groupby(["arm", fac]).agg(sim_stable=("sim_stable", "mean"), inc_fixed_med=("inc_fixed", "median")).round(3).unstack(0).to_string())
        print("-- paired (scenarios stable under both designs)"); paired(d, keys[1:])
