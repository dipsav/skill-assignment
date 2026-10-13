"""table_B23.py -- Table for Section 6.1, surges of several SKUs (revision/tikz/B23_table.tex): per scenario (all SKUs
+5/10/20%; skill set of the busiest technician +10/25%; of a random technician +10/25%) and design: stability (share of
the 128 instances stable), median cost increase with the stock fixed and with the stock re-optimized over the stable
instances, and the number of instances (stable under both) in which the mu design's increase is below the FCFS design's."""
import os, numpy as np, pandas as pd
from analyze_B1B23 import load, T
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision", "tikz", "B23_table.tex")
ARMS = [("FCFS design / FCFS", "FCFS"), ("mu design / mu", "$\\mu$"), ("hmu design / hmu", "$h\\times\\mu$")]
SCEN = [("all", "All SKUs"), ("busiest", "Skill set of the busiest technician"), ("random", "Skill set of a random technician")]

if __name__ == "__main__":
    d = load("B23"); a, b = ARMS[0][0], ARMS[1][0]
    lines = [r"\begin{table}[!htb]", r"\centering",
             r"\caption{Stability and cost increase under surges of several SKUs at once: all SKUs, the SKUs in the skill set of the busiest technician, and those of a technician drawn at random, for the design optimized for FCFS and the designs optimized for the $\mu$ and $h\times\mu$ rules, each under its own rule (128 instances, 35 replications each). Stable: share of the instances in which the shop remains stable; cost increase: median over the stable instances, relative to the nominal total cost, with the base-stock levels fixed and re-optimized; last column: instances stable under both designs in which the $\mu$ design's increase is below the FCFS design's.}",
             r"\label{tab:multi_surges}", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lc rrr rrr rrr c@{}}", r"\toprule",
             r"& & \multicolumn{3}{c}{Stable (\%)} & \multicolumn{3}{c}{Increase, stock fixed (\%)} & \multicolumn{3}{c}{Increase, stock re-opt. (\%)} & $\mu$ below \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Surged SKUs & $\delta$ & FCFS & $\mu$ & $h\times\mu$ & FCFS & $\mu$ & $h\times\mu$ & FCFS & $\mu$ & $h\times\mu$ & FCFS \\", r"\midrule"]
    for kind, label in SCEN:
        for i, (dl, s) in enumerate(d[d.scenario == kind].groupby("delta")):
            st = s[s.sim_stable]; p = s.pivot(index="instance", columns="arm", values=["sim_stable", "inc_fixed"])
            both = p["sim_stable"][a].astype(bool) & p["sim_stable"][b].astype(bool); low = int((p["inc_fixed"][b][both] < p["inc_fixed"][a][both]).sum())
            cells = [label if i == 0 else "", f"{dl:.0%}".replace("%", r"\%")]
            cells += [f"{100 * s[s.arm == arm].sim_stable.mean():.1f}" for arm, _ in ARMS]
            cells += [f"{100 * st[st.arm == arm].inc_fixed.median():.1f}" for arm, _ in ARMS]
            cells += [f"{100 * st[st.arm == arm].inc_reopt.median():.1f}" for arm, _ in ARMS]
            cells.append(f"{low} of {int(both.sum())}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines[12:-2]))
