"""table_B5.py -- Table for Section 6.3 (revision/tikz/B5_table.tex) from tables/B5_instances.csv (analyze_B5.py):
per absent technician (busiest / random) and absence length, for the FCFS design under FCFS and the mu design under
the mu rule: median utilization and skills of the absent technician, median peak backorders, median loss (multiples of
the nominal cost of a period of the absence's length), median recovery (expected failures after the return), share of
instances recovered; last column: instances in which the mu design has the smaller loss."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "revision", "tikz", "B5_table.tex")
a, b = "FCFS design / FCFS", "mu design / mu"; KIND = {"busiest": "Busiest technician", "random": "Random technician"}

if __name__ == "__main__":
    inst = pd.read_csv(os.path.join(HERE, "tables", "B5_instances.csv"))
    lines = [r"\begin{table}[!htb]", r"\centering",
             r"\caption{Absence of one technician for 2,500 or 10,000 expected failures: the design optimized for FCFS (under FCFS) and the design optimized for the $\mu$ rule (under it), 128 instances, 35 replications each. Utilization: nominal utilization of the absent technician; peak: largest number of backorders during the absence; loss: extra backorder cost of the episode in multiples of the nominal total cost of a period of the absence's length; recovery: expected failures after the technician's return until the backorders return to their baseline; medians over the instances; recovered: share of instances that recover within the horizon. Last column: instances in which the $\mu$ design has the smaller loss.}",
             r"\label{tab:absence}", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lr rrrrr rrrrr r@{}}", r"\toprule",
             r"& & \multicolumn{5}{c}{Design for FCFS} & \multicolumn{5}{c}{Design for the $\mu$ rule} & \\", r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
             r"Absent & Length & Util. & Peak & Loss & Recovery & Recov. & Util. & Peak & Loss & Recovery & Recov. & $\mu$ lower \\", r"\midrule"]
    for kind, sub in inst.groupby("absence"):
        for i, (D, s) in enumerate(sub.groupby("duration")):
            p = s.pivot(index="instance", columns="arm", values=["peak", "loss", "rec", "recovered", "tech_util"])
            r = p["loss"][b] / p["loss"][a]; low = int((r < 1).sum())
            cells = [KIND[kind] if i == 0 else "", f"{D:,}"]
            for arm in (a, b):
                cells += [f"{p['tech_util'][arm].median():.2f}", f"{p['peak'][arm].median():,.0f}", f"{p['loss'][arm].median():.1f}", f"{p['rec'][arm].median():,.0f}", f"{p['recovered'][arm].mean():.0%}".replace("%", r"\%")]
            cells.append(f"{low} of 128"); lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines[12:-2]))
