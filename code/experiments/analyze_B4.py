"""
analyze_B4.py -- Section 6.2 numbers from the B4 records (finite-duration surge of all SKUs, three design sets).

Reads surge_results_B4/{fcfs_designs,mu_designs,hmu_designs}/*.json.gz (run_surge_cases.py records), builds the
per-replication table (tables/B4_transient.csv) and prints, per arm (design set x rule), gamma and surge length:
  baseline backorders, peak backorders during the surge (mean over replications), cumulative backorder loss against
  the design's steady-state EBO (in failure-time units), recovery time of the backorders (expected failures after the
  surge; share censored), and the same for waiting parts for reference. Paired comparisons (instance means) between
  the arms: FCFS design/FCFS vs mu design/mu (the paper's two designs), and FCFS design under mu (sequential).
"""
import glob, gzip, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "surge_results_B4")
ARMS = {("PSO_results_tuned_polished", "FCFS"): "FCFS design / FCFS", ("PSO_results_tuned_polished", "mu"): "FCFS design / mu (sequential)",
        ("PSO_results_tuned_polished", "hmu"): "FCFS design / hmu (sequential)", ("PSO_results_tuned_rule_mu_polished", "mu"): "mu design / mu",
        ("PSO_results_tuned_rule_hmu_polished", "hmu"): "hmu design / hmu"}
DESIGNS = {"PSO_results_tuned_polished": "FCFS design", "PSO_results_tuned_rule_mu_polished": "mu design", "PSO_results_tuned_rule_hmu_polished": "hmu design",
           "PSO_results_fullCT": "full CT", "PSO_results_fullCT_rule_mu": "full CT", "PSO_results_tuned_rule_mu_np_polished": "mu-NP design",
           "PSO_results_tuned_rule_hmu_np_polished": "hmu-NP design", "PSO_results_tuned_rule_mu_cov2_polished": "mu-cov2 design",
           "PSO_results_tuned_cov2_polished": "FCFS-cov2 design"}


def arm_label(r):
    """Arm of a surge record: design, rule, discipline and operational switch. The labels of the original arms are kept."""
    ds, rule = r.get("design_set", ""), r["rule"]; pre = r.get("preemptive", True); sw = r.get("switch")
    if pre and not sw and (ds, rule) in ARMS:
        return ARMS[(ds, rule)]
    lab = f"{DESIGNS.get(ds, ds)} / {rule}" + ("" if pre else " NP")
    if sw:
        lab += f" -> {sw['to']}" + ("" if sw.get("preemptive", True) else " NP") + (" during" if sw.get("end") is not None else "")
    return lab


def design_info(path, _cache={}):
    """Penalty rate b, total failure rate and nominal cost per unit time of the design (to normalize the loss)."""
    if path not in _cache:
        d = json.load(open(path)); _cache[path] = (float(d["penalty_cost"]), float(np.sum(d["failure_rates"])), float(d["total_cost"]))
    return _cache[path]


def load(root=ROOT):
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*.json.gz")):
        r = json.load(gzip.open(f, "rt")); arm = arm_label(r)
        b, lam, cost = design_info(r["design_source"]); D = r["protocol"]["numberSurgeRequests"]
        for rep in r["per_replication"]:
            rows.append({"arm": arm, "instance": r["case"], "gamma": r["gamma"], "duration": D, **r["factors"], "overflow": r["overflow_share_max"], **rep,
                         # extra backorder cost of the episode (b x loss) relative to the nominal cost of the surge period (cost rate x D / Lambda)
                         "loss_rel": b * rep["cumulative_backorder_loss_vs_nominal"] / (cost * D / lam)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load(); d.to_csv(os.path.join(HERE, "tables", "B4_transient.csv"), index=False)
    print(f"{len(d)} replication rows, {d.instance.nunique()} instances, arms: {sorted(d.arm.unique())}")
    d["recovered"] = 1 - d.recovery_backorders_censored
    inst0 = d.groupby(["arm", "gamma", "duration", "instance"]).mean(numeric_only=True).reset_index()        # instance means first
    g = inst0.groupby(["arm", "gamma", "duration"]).agg(baseline_bo=("baseline_mean_backorders", "median"), surge_bo=("surge_mean_backorders", "median"), peak_bo=("surge_peak_backorders", "median"),
                                                       peak_mean=("surge_peak_backorders", "mean"), loss_rel_med=("loss_rel", "median"), loss_rel_mean=("loss_rel", "mean"),
                                                       recovery=("recovery_backorders_arrivals", "median"), recovery_mean=("recovery_backorders_arrivals", "mean"), recovered=("recovered", "mean"),
                                                       peak_q=("surge_peak_queue", "median"), overflow=("overflow", "max")).reset_index()
    pd.set_option("display.width", 250); print(g.round(2).to_string(index=False)); g.to_csv(os.path.join(HERE, "tables", "B4_summary.csv"), index=False)
    # paired comparisons on instance means, per gamma x duration
    inst = d.groupby(["arm", "gamma", "duration", "instance"]).agg(peak=("surge_peak_backorders", "mean"), loss=("loss_rel", "mean"), rec=("recovery_backorders_arrivals", "mean"), recovered=("recovered", "mean")).reset_index()
    inst.to_csv(os.path.join(HERE, "tables", "B4_instances.csv"), index=False)
    for (gm, du), sub in inst.groupby(["gamma", "duration"]):
        p = sub.pivot(index="instance", columns="arm", values="loss"); q = sub.pivot(index="instance", columns="arm", values="peak"); rc = sub.pivot(index="instance", columns="arm", values="recovered")
        rr = sub.pivot(index="instance", columns="arm", values="rec")
        a, c = "FCFS design / FCFS", "FCFS design / mu (sequential)"
        for b in ("mu design / mu", "hmu design / hmu"):
            if a in p and b in p:
                r = (p[b] / p[a]).replace(np.inf, np.nan).dropna(); r = r[r > 0]; dr = (rr[b] - rr[a]).dropna()
                print(f"gamma {gm} x {du:5d}: {b} vs {a}: loss ratio median {r.median():.2f}, lower in {(r < 1).sum()}/{len(r)} (p = {wilcoxon(np.log(r)).pvalue:.1e}); "
                      f"peak ratio median {(q[b]/q[a]).median():.2f}; recovery shorter in {(dr < 0).sum()}/{len(dr)} (median diff {dr.median():.0f}); recovered share {rc[a].mean():.0%} vs {rc[b].mean():.0%}"
                      + (f"; sequential loss ratio median {(p[c]/p[a]).median():.2f}" if c in p and b.startswith("mu") else ""))
