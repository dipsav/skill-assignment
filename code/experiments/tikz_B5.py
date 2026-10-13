"""tikz_B5.py -- Figure for Section 6.3 as pgfplots (revision/tikz/fig9_absence.tex): (a) mean backorders over time on
instance 128 when its busiest technician is absent for 10,000 expected failures, FCFS design under FCFS and mu design
under the mu rule (35 replications, +-2 SEM); (b) empirical distribution over the 128 instances of the recovery time
after the technician's return, both designs, absences of 2,500 and 10,000 failures (busiest technician)."""
import gzip, json, os, subprocess
import numpy as np, pandas as pd
from analyze_B4 import HERE
from tikz_figures import PREAMBLE, OUT

ROOT = os.path.join(HERE, "..", "surge_results_B5"); CASE, KIND, D = 128, "busiest", 10000
ARMS = (("fcfs_designs", "FCFS", "FCFS design / FCFS", "red", "design for FCFS"), ("mu_designs", "mu", "mu design / mu", "blue", "design for the $\\mu$ rule"))


def trajectory(folder, rule):
    r = json.load(gzip.open(os.path.join(ROOT, folder, f"surge_case_{CASE:04d}_{rule}_g1_d{D}_abs{KIND}.json.gz"), "rt")); mt = r["mean_trajectories"]
    t = np.asarray(mt["normalized_time"]) / 1000; y = np.asarray(mt["backorders_total"]); e = np.asarray(mt["backorders_total_sem"])
    keep = (t >= -10) & (t <= 60); return t[keep], y[keep], e[keep], r["absence"]


if __name__ == "__main__":
    inst = pd.read_csv(os.path.join(HERE, "tables", "B5_instances.csv"))
    s = ["\\begin{tikzpicture}", "\t\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, width=0.5\\textwidth, height=5.2cm,",
         "\t\ttick label style={/pgf/number format/fixed}, scaled ticks=false, xlabel style={align=center}, ylabel near ticks, legend style={draw=none, font=\\footnotesize}, legend cell align=left]"]
    s.append(f"\t\\nextgroupplot[xlabel={{Expected failures since the technician\\\\left (thousands)}}, ylabel={{Backorders (all SKUs)}}, xmin=-10, xmax=60, ymin=0, title={{(a) Instance {CASE}, busiest technician absent {D:,} failures}}, title style={{font=\\small}}, legend pos=north east]")
    s.append(f"\t\t\\fill[black!12] (axis cs:0,0) rectangle (axis cs:{D/1000:g},5000);")
    ymax = 1.05 * max(np.max(trajectory(f, r)[1] + 2 * trajectory(f, r)[2]) for f, r, *_ in ARMS)
    s[-2] = s[-2].replace("ymin=0,", f"ymin=0, ymax={ymax:.0f},")
    for folder, rule, arm, color, label in ARMS:
        t, y, e, ab = trajectory(folder, rule); print(arm, "absent technician", ab)
        s.append(f"\t\t\\addplot[{color}, line width=0.8pt, forget plot] coordinates {{" + " ".join(f"({a:.2f},{b:.2f})" for a, b in zip(t, y)) + "};")
        # +-2 SEM band as one closed filled polygon (upper edge forward, lower edge backward): needs no fillbetween library
        band = " ".join(f"({a:.2f},{b + 2 * c:.2f})" for a, b, c in zip(t, y, e)) + " " + " ".join(f"({a:.2f},{max(b - 2 * c, 0):.2f})" for a, b, c in zip(t[::-1], y[::-1], e[::-1]))
        s.append(f"\t\t\\addplot[{color}, fill={color}, opacity=0.2, draw=none, forget plot] coordinates {{" + band + "} \\closedcycle;")
    for folder, rule, arm, color, label in ARMS:
        s.append(f"\t\t\\addlegendimage{{{color}, line width=0.8pt}}\\addlegendentry{{{label}}}")
    s.append("\t\\nextgroupplot[xlabel={Recovery time after the return\\\\(thousand expected failures)}, ylabel={Share of instances recovered}, xmin=0, xmax=60, ymin=0, ymax=1, title={(b) Recovery, busiest technician, two absence lengths}, title style={font=\\small}, legend style={at={(0.5,-0.42)}, anchor=north, legend columns=2, draw=none, font=\\footnotesize}]")
    for Dk, style in ((2500, "dotted"), (10000, "solid")):
        for folder, rule, arm, color, label in ARMS:
            v = inst[(inst.arm == arm) & (inst.absence == KIND) & (inst.duration == Dk)].rec.dropna().sort_values().values / 1000; n = 128
            pts = [(0, 0)] + [(x, (i + 1) / n) for i, x in enumerate(v)]
            s.append(f"\t\t\\addplot[{color}, {style}, line width=0.8pt, const plot, forget plot] coordinates {{" + " ".join(f"({a:.2f},{b:.4f})" for a, b in pts) + "};")
    for folder, rule, arm, color, label in ARMS:
        s.append(f"\t\t\\addlegendimage{{{color}, line width=0.8pt}}\\addlegendentry{{{label}}}")
    for Dk, style in ((2500, "dotted"), (10000, "solid")):
        s.append(f"\t\t\\addlegendimage{{black, {style}, line width=0.8pt}}\\addlegendentry{{{Dk:,} failures}}")
    s += ["\t\\end{groupplot}", "\\end{tikzpicture}"]
    name = "fig9_absence"; open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(s) + "\n")
    open(os.path.join(OUT, name + "_standalone.tex"), "w").write(PREAMBLE + f"\t\\input{{{name}.tex}}\n\\end{{document}}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + "_standalone.tex"], cwd=OUT, capture_output=True, text=True)
    print(name, "compiled" if r.returncode == 0 else "FAILED:\n" + r.stdout[-2000:])
    for ext in (".aux", ".log"):
        p = os.path.join(OUT, name + "_standalone" + ext); os.path.exists(p) and os.remove(p)
