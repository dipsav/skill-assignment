"""table_B4.py -- Table for Section 6.2 (revision/tikz/B4_table.tex) from tables/B4_instances.csv + B4_transient.csv:
per gamma and surge length, for the FCFS design under FCFS and the mu design under the mu rule: median peak backorders,
median extra backorder cost of the episode (multiples of the nominal cost of the surge period), median recovery time
(expected failures after the surge) and share of instances recovered within the horizon; last column: instances in
which the mu design has the smaller loss."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "revision", "tikz", "B4_table.tex")
inst = pd.read_csv(os.path.join(HERE, "tables", "B4_instances.csv")); a, b = "FCFS design / FCFS", "mu design / mu"
lines = [r"\begin{table}[!htb]", r"\centering",
         r"\caption{Finite-duration surge of all SKUs: the design optimized for FCFS (operated under FCFS) and the design optimized for the $\mu$ rule (operated under it), 128 instances, 35 replications each. Peak: largest number of backorders during the surge; loss: extra backorder cost of the episode in multiples of the nominal total cost of a period of the surge's length; recovery: expected failures after the end of the surge until the backorders return to their baseline; medians over the instances; recovered: share of instances that recover within the horizon. Last column: instances in which the $\mu$ design has the smaller loss.}",
         r"\label{tab:surge_all}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}cr rrrr rrrr r@{}}", r"\toprule",
         r"& & \multicolumn{4}{c}{Design for FCFS} & \multicolumn{4}{c}{Design for the $\mu$ rule} & \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"$\gamma$ & Length & Peak & Loss & Recovery & Recov. & Peak & Loss & Recovery & Recov. & $\mu$ lower \\", r"\midrule"]
for g, sub in inst.groupby("gamma"):
    for i, (D, s) in enumerate(sub.groupby("duration")):
        p = s.pivot(index="instance", columns="arm", values=["peak", "loss", "rec", "recovered"])
        r = (p["loss"][b] / p["loss"][a]); low = int((r < 1).sum())
        cells = [f"{g:g}" if i == 0 else "", f"{D:,}"]
        for arm in (a, b):
            cells += [f"{p['peak'][arm].median():,.0f}", f"{p['loss'][arm].median():.1f}", f"{p['rec'][arm].median():,.0f}", f"{p['recovered'][arm].mean():.0%}".replace("%", r"\%")]
        cells.append(f"{low} of 128")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule" if g < 2 else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table}"]
open(OUT, "w").write("\n".join(lines) + "\n"); print(OUT); print("\n".join(lines[12:-2]))
