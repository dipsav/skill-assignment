"""
select_t2.py -- pick the T2 candidates from the T1 results and write tuning/t2_settings.json.

Rule (per grid: proposed, proposed_ext, reset, original): the best setting by "optimum found" and the best by the gap
after 25% of the designs (the two criteria of T1), de-duplicated; plus the published proposed setting (label "pso") and
the T1-best original (label "pso_orig"), which are the drivers' defaults. Prints the ranking tables.
    python3 tuning/select_t2.py [--out tuning/t2_settings.json]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["c1", "c2", "inertia_min", "inertia_max", "inertia_power"]
VARIANT_FLAG = {"original": {"variant": "original"}, "reset": {"variant": "reset"}}
BASE = {"original": "pso_orig", "proposed": "pso", "proposed_ext": "pso", "reset": "pso"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "t2_settings.json")); a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "t1_results_*seed1.csv"))], ignore_index=True)
    d = d[np.isfinite(d.gap)]
    if "inertia_power" not in d: d["inertia_power"] = 1.0
    d["inertia_power"] = d["inertia_power"].fillna(1.0)
    settings = {}
    for v, g in d.groupby("variant"):
        t = g.groupby(KEYS).agg(gap10=("gap_at_10", "mean"), gap25=("gap_at_25", "mean"), final=("gap", "mean"),
                                optimal=("optimal", "mean"), distinct=("distinct", "median"), runs=("gap", "size")).reset_index()
        print(f"\n== {v}: {len(t)} settings, {int(t.runs.iloc[0])} runs each; best by optimum found, then by gap at 25%")
        print(pd.concat([t.sort_values("optimal", ascending=False).head(3), t.sort_values("gap25").head(3)]).round(4).to_string(index=False))
        for crit, asc in (("optimal", False), ("gap25", True)):
            r = t.sort_values(crit, ascending=asc).iloc[0]
            label = f"{BASE[v]}:{v}_{crit}"
            params = {k: float(r[k]) for k in KEYS}; params.update(VARIANT_FLAG.get(v, {}))
            if params not in settings.values():
                settings[label] = params
    json.dump(settings, open(a.out, "w"), indent=1)
    print(f"\n{len(settings)} T2 candidate settings -> {a.out} (plus the defaults pso = published, pso_orig = T1-best original)")
    for k, v in settings.items(): print(f"  {k:30s} {v}")
