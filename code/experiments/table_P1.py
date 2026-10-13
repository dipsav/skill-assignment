"""table_P1.py -- Table for the preemption subsection (revision/tikz/P1_table.tex): total cost (base-stock levels
re-optimized) relative to the design optimized for FCFS operated under FCFS, medians over the 128 instances, for
exponential, Erlang-4 and lognormal (CV 2) repair times; share of instances stable in simulation in brackets where
below 100%. Rows: full cross-training under FCFS, mu (preemptive, non-preemptive); FCFS design under mu (P, NP); the
design optimized for preemptive mu under mu (P, NP); the design optimized for non-preemptive mu (exponential only)."""
import glob, json, os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
d = pd.read_csv(os.path.join(HERE, "tables", "P1_preemption.csv"))
for c in ("total_fixed", "total_reopt"): d[c] = d[c].replace([np.inf], np.nan)
d["arm"] = d.design + "/" + d.rule + "/" + np.where(d.rule == "fcfs", "", np.where(d.preemptive, "P", "NP"))
base = d[d.arm == "fcfs/fcfs/"].set_index(["cell", "instance"]).total_reopt; d = d.join(base.rename("ref"), on=["cell", "instance"]); d["rel"] = d.total_reopt / d.ref
g = d.groupby(["arm", "cell"]).agg(rel=("rel", "median"), stable=("sim_stable", "mean"))
case = lambda p: int(re.search(r"case_(\d+)", p).group(1))
F = {case(p): json.load(open(p))["total_cost"] for p in glob.glob(os.path.join(ROOT, "PSO_results_tuned_polished", "*case_*.json"))}
NP = {case(p): json.load(open(p))["total_cost"] for p in glob.glob(os.path.join(ROOT, "PSO_results_tuned_rule_mu_np_polished", "*case_*.json"))}
np_rel = float(np.median([NP[c] / F[c] for c in F]))
ROWS = [("full/fcfs/", "Full cross-training, FCFS"), ("full/mu/P", "Full cross-training, $\\mu$ rule, preemptive"), ("full/mu/NP", "Full cross-training, $\\mu$ rule, non-preemptive"),
        ("fcfs/fcfs/", "Design for FCFS, FCFS"), ("fcfs/mu/P", "Design for FCFS, $\\mu$ rule, preemptive"), ("fcfs/mu/NP", "Design for FCFS, $\\mu$ rule, non-preemptive"),
        ("mu/mu/P", "Design for preemptive $\\mu$, preemptive"), ("mu/mu/NP", "Design for preemptive $\\mu$, run non-preemptively"), ("A3", "Design for non-preemptive $\\mu$, non-preemptive")]
cells = [("exp", "Exponential"), ("svc_erlang4", "Erlang-4"), ("svc_logn2", "Lognormal, CV 2")]
L = [r"\begin{table}[!htb]", r"\centering",
     r"\caption{Preemptive and non-preemptive $\mu$ rule: total cost with the base-stock levels re-optimized, relative to the design optimized for FCFS and operated under FCFS, median over the 128 instances, for exponential, Erlang-4 (CV 0.5) and lognormal (CV 2) repair times; in brackets, the share of instances that are stable in simulation when it is below 100\%.}",
     r"\label{tab:preemption}", r"\small", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", "Configuration & " + " & ".join(c[1] for c in cells) + r" \\", r"\midrule"]
for key, lab in ROWS:
    if key == "A3": L.append(f"{lab} & {np_rel:.2f} & -- & -- \\\\"); continue
    vals = []
    for ck, _ in cells:
        r = g.loc[(key, ck)]; vals.append(f"{r.rel:.2f}" + (f" ({100 * r.stable:.0f}\\%)" if r.stable < 0.999 else ""))
    L.append(f"{lab} & " + " & ".join(vals) + r" \\")
    if key in ("full/mu/NP", "fcfs/mu/NP"): L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(ROOT, "revision", "tikz", "P1_table.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[8:-3])); print("NP design median rel", round(np_rel, 3))
