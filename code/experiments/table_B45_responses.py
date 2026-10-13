"""table_B45_responses.py -- Tables for Sections 6.2 and 6.3 comparing designs and operational responses on the
reference disruptions: (a) surge gamma 1.5 x 10,000 failures (tab:surge_responses), (b) absence of the busiest and of a
random technician for 10,000 failures (tab:absence_responses). Medians over the 128 instances of the peak backorders,
the loss (extra backorder cost in multiples of the nominal cost of a period of the disruption's length), the recovery
time (expected failures after the disruption), and the share of instances recovered. Writes revision/tikz/*.tex."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "revision", "tikz")
ARMS = [("FCFS design / FCFS", "Design for FCFS, FCFS"), ("FCFS design / FCFS -> mu NP during", "\\quad switched to non-preemptive $\\mu$ during the disruption"),
        ("FCFS design / FCFS -> mu during", "\\quad switched to preemptive $\\mu$ during the disruption"), ("FCFS design / mu NP", "Design for FCFS, non-preemptive $\\mu$ rule throughout"),
        ("mu design / mu", "Design for $\\mu$, $\\mu$ rule"), ("mu-cov2 design / mu", "Design for $\\mu$, every SKU covered twice"), ("mu-NP design / mu NP", "Design for non-preemptive $\\mu$"),
        ("full CT / FCFS", "Full cross-training, FCFS"), ("full CT / mu", "Full cross-training, $\\mu$ rule")]


def rows(inst, head, cap, label, colhead):
    L = [r"\begin{table}[!htb]", r"\centering", cap, f"\\label{{{label}}}", r"\small", r"\setlength{\tabcolsep}{4pt}", head, r"\toprule", colhead, r"\midrule"]
    return L


b4 = pd.read_csv(os.path.join(HERE, "tables", "B4_instances.csv")); s = b4[(b4.gamma == 1.5) & (b4.duration == 10000)]
L = rows(s, r"\begin{tabular}{@{}lrrrr@{}}", r"\caption{Designs and operational responses under a surge of all failure rates by $50\%$ for $10{,}000$ expected failures, 128 instances, 35 replications each: median peak number of backorders, median loss (extra backorder cost of the episode in multiples of the nominal cost of a period of the surge's length), median recovery time (expected failures after the surge) and share of instances recovered within the horizon.}",
         "tab:surge_responses", r"Design and operation & Peak & Loss & Recovery & Recovered \\")
for arm, lab in ARMS:
    g = s[s.arm == arm]
    if not len(g): continue
    L.append(f"{lab} & {g.peak.median():,.0f} & {g.loss.median():.1f} & {g.rec.median():,.0f} & {100 * g.recovered.mean():.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(os.path.join(OUT, "B4_responses_table.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[9:-3]))
b5 = pd.read_csv(os.path.join(HERE, "tables", "B5_instances.csv")); s = b5[b5.duration == 10000]
L = rows(s, r"\begin{tabular}{@{}l rrrr rrrr@{}}", r"\caption{Designs and operational responses under the absence of a technician for $10{,}000$ expected failures (busiest technician; technician drawn at random), 128 instances, 35 replications each: median peak number of backorders, median loss (multiples of the nominal cost of a period of the absence's length), median recovery time after the technician's return (expected failures) and share of instances recovered.}",
         "tab:absence_responses", r"& \multicolumn{4}{c}{Busiest technician absent} & \multicolumn{4}{c}{Random technician absent} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9} Design and operation & Peak & Loss & Recovery & Recov. & Peak & Loss & Recovery & Recov. \\")
print()
for arm, lab in ARMS:
    cells = [lab]
    for kind in ("busiest", "random"):
        g = s[(s.arm == arm) & (s.absence == kind)]
        cells += ([f"{g.peak.median():,.0f}", f"{g.loss.median():.1f}", f"{g.rec.median():,.0f}", f"{100 * g.recovered.mean():.0f}\\%"] if len(g) else ["--"] * 4)
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(os.path.join(OUT, "B5_responses_table.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[9:-3]))
