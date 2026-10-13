"""
analyze_B5.py -- Section 6.3 numbers from the B5 records (absence of one technician for 2,500 or 10,000 expected failures).

Same layout as analyze_B4.py: surge_results_B5/{fcfs_designs,mu_designs,hmu_designs}/*_abs{busiest,random}.json.gz.
Prints per arm, absence kind and length: backorders before/during/after, peak, loss against the steady-state EBO,
recovery (expected failures after the return; share censored), and the absent technician's skills and utilization;
paired comparison FCFS design/FCFS vs mu design/mu. Writes tables/B5_transient.csv, tables/B5_summary.csv.
"""
import glob, gzip, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from analyze_B4 import ARMS, HERE, design_info, arm_label

ROOT = os.path.join(HERE, "..", "surge_results_B5")


def load():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "*", "*.json.gz")):
        r = json.load(gzip.open(f, "rt")); arm = arm_label(r); ab = r["absence"]
        b, lam, cost = design_info(r["design_source"]); D = r["protocol"]["numberSurgeRequests"]
        for rep in r["per_replication"]:
            rows.append({"arm": arm, "instance": r["case"], "absence": ab["kind"], "duration": D, "technician": ab["technician"],
                         "tech_skills": ab["skills"], "tech_util": ab["utilization"], **r["factors"], "overflow": r["overflow_share_max"], **rep,
                         "loss_rel": b * rep["cumulative_backorder_loss_vs_nominal"] / (cost * D / lam)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load(); d.to_csv(os.path.join(HERE, "tables", "B5_transient.csv"), index=False)
    print(f"{len(d)} replication rows, {d.instance.nunique()} instances, arms: {sorted(d.arm.unique())}")
    d["recovered"] = 1 - d.recovery_backorders_censored
    g = d.groupby(["arm", "absence", "duration"]).agg(tech_util=("tech_util", "mean"), tech_skills=("tech_skills", "mean"), baseline_bo=("baseline_mean_backorders", "mean"),
                                                     during_bo=("surge_mean_backorders", "mean"), peak_bo=("surge_peak_backorders", "mean"), loss=("loss_rel", "median"),
                                                     recovery=("recovery_backorders_arrivals", "mean"), recovered=("recovered", "mean"), peak_q=("surge_peak_queue", "mean"), overflow=("overflow", "max")).reset_index()
    pd.set_option("display.width", 250); print(g.round(2).to_string(index=False)); g.to_csv(os.path.join(HERE, "tables", "B5_summary.csv"), index=False)
    inst = d.groupby(["arm", "absence", "duration", "instance"]).agg(peak=("surge_peak_backorders", "mean"), loss=("loss_rel", "mean"), rec=("recovery_backorders_arrivals", "mean"), recovered=("recovered", "mean"), tech_util=("tech_util", "first"), tech_skills=("tech_skills", "first"), during=("surge_mean_backorders", "mean"), baseline=("baseline_mean_backorders", "mean")).reset_index()
    inst.to_csv(os.path.join(HERE, "tables", "B5_instances.csv"), index=False)
    a, b, c = "FCFS design / FCFS", "mu design / mu", "FCFS design / mu (sequential)"
    for (ab, du), sub in inst.groupby(["absence", "duration"]):
        p = sub.pivot(index="instance", columns="arm", values="loss"); q = sub.pivot(index="instance", columns="arm", values="peak"); rc = sub.pivot(index="instance", columns="arm", values="recovered")
        if a in p and b in p:
            r = (p[b] / p[a]).replace([np.inf], np.nan).dropna(); r = r[r > 0]
            print(f"{ab:8s} x {du:5d}: loss mu-design/mu vs FCFS-design/FCFS: median ratio {r.median():.2f}, lower in {(r < 1).sum()}/{len(r)} (p = {wilcoxon(np.log(r)).pvalue:.1e}); peak ratio median {(q[b]/q[a]).median():.2f}; recovered share {rc[a].mean():.0%} vs {rc[b].mean():.0%}" + (f"; sequential loss ratio median {(p[c]/p[a]).median():.2f}" if c in p else ""))
