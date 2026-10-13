"""absence_skills.py -- B5 mechanism: does the absent technician hold a skill no other technician has (so that its SKUs
have no repair capacity during the absence)? Per design set and absence kind: share of instances with a unique skill,
and the loss / recovery with and without one (tables/B5_instances.csv + design JSONs)."""
import glob, gzip, json, os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "surge_results_B5")
from analyze_B4 import ARMS

rows = []
for f in glob.glob(os.path.join(ROOT, "*", "*.json.gz")):
    r = json.load(gzip.open(f, "rt")); arm = ARMS.get((r.get("design_set", ""), r["rule"]), ""); ab = r["absence"]
    X = np.asarray(r["assignment"], int); j = ab["technician"]; cover = X.sum(axis=0)             # technicians per SKU
    unique = [int(k) for k in np.flatnonzero(X[j]) if cover[k] == 1]
    lam = np.asarray(json.load(open(r["design_source"]))["failure_rates"], float)
    rows.append({"arm": arm, "instance": r["case"], "absence": ab["kind"], "duration": r["protocol"]["numberSurgeRequests"], "technician": j, "skills": int(X[j].sum()),
                 "unique_skills": len(unique), "unique_share_of_failures": float(lam[unique].sum() / lam.sum()) if unique else 0.0, "technicians": int((X.sum(axis=1) > 0).sum()), "util": ab["utilization"]})
u = pd.DataFrame(rows); inst = pd.read_csv(os.path.join(HERE, "tables", "B5_instances.csv")).merge(u, on=["arm", "instance", "absence", "duration"])
inst["has_unique"] = inst.unique_skills > 0
pd.set_option("display.width", 200)
print(inst.groupby(["absence", "arm"]).agg(n=("instance", "size"), unique_share=("has_unique", "mean"), unique_skills=("unique_skills", "mean"), fail_share=("unique_share_of_failures", "mean"), technicians=("technicians", "mean"), util=("util", "mean")).round(3).to_string())
print("\nloss / peak / recovery by whether the absent technician holds a unique skill (medians):")
print(inst.groupby(["absence", "duration", "arm", "has_unique"]).agg(n=("instance", "size"), loss=("loss", "median"), peak=("peak", "median"), rec=("rec", "median"), recovered=("recovered", "mean")).round(2).to_string())
a, b = "FCFS design / FCFS", "mu design / mu"
for (k, D), s in inst.groupby(["absence", "duration"]):
    p = s.pivot(index="instance", columns="arm", values=["loss", "has_unique", "peak"])
    if a in p["loss"] and b in p["loss"]:
        r = p["loss"][b] / p["loss"][a]; hu = p["has_unique"].astype(bool)
        print(f"{k} {D}: loss ratio mu/FCFS median {r.median():.2f}; unique skill FCFS {hu[a].mean():.0%} mu {hu[b].mean():.0%}; ratio median when mu has unique & FCFS not: {r[hu[b] & ~hu[a]].median():.2f} (n {(hu[b] & ~hu[a]).sum()}), neither: {r[~hu[b] & ~hu[a]].median():.2f} (n {(~hu[b] & ~hu[a]).sum()}), both: {r[hu[b] & hu[a]].median():.2f} (n {(hu[b] & hu[a]).sum()})")
inst.to_csv(os.path.join(HERE, "tables", "B5_instances_skills.csv"), index=False)
