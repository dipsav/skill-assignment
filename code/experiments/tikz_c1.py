"""tikz_c1.py -- Figure 3 (convergence of the methods at an equal budget) as pgfplots: revision/tikz/fig3_convergence.tex
(+ standalone PDF). For each run (32 instances x 5 seeds) and method, the best search cost found so far as a step
function of the number of distinct skill matrices simulated (the trace stored by c1_benchmark.py), expressed as the gap
to the cheapest search cost reached by any of the five methods on the same instance and seed; the curves are the
medians over the 160 runs on a grid of 25 matrices. The hybrid's trace starts where its local search starts (the swarm
phase of the hybrid is not traced), so its curve begins at the hand-over point."""
import glob, json, os, subprocess
import numpy as np, pandas as pd
from tikz_figures import PREAMBLE, OUT
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "C1_results")
METHODS = [("hybrid", "Proposed BPSO with local search", "black", "solid"), ("pso", "Proposed BPSO", "blue", "solid"), ("vns", "VNS", "red", "dashed"),
           ("ga", "GA", "black!50!green", "dashdotted"), ("pso_orig", "Original BPSO", "blue", "dotted")]
GRID = np.arange(25, 2501, 25)

runs = {}
for alg, *_ in METHODS:
    for f in glob.glob(os.path.join(ROOT, alg, "*.json")):
        r = json.load(open(f)); runs[(alg, r["case"], r["seed"])] = np.array(r["trace"], float)
best = {}
for (alg, c, s), t in runs.items():
    best[(c, s)] = min(best.get((c, s), np.inf), t[:, 1].min())
curves = {}
for alg, *_ in METHODS:
    rows = []
    for (a, c, s), t in runs.items():
        if a != alg: continue
        n, v = t[:, 0], np.minimum.accumulate(t[:, 1]); g = np.full(len(GRID), np.nan)
        for i, x in enumerate(GRID):
            k = np.searchsorted(n, x, side="right") - 1
            if k >= 0: g[i] = v[k] / best[(c, s)] - 1
        rows.append(g)
    A = np.array(rows); med = np.nanmedian(A, axis=0); med[np.sum(~np.isnan(A), axis=0) < len(rows) / 2] = np.nan   # median only where at least half of the runs have a value (the hybrid's trace starts at its hand-over)
    curves[alg] = med; print(alg, "runs", len(rows), "median gap at 500/1000/2500:", [round(100 * curves[alg][GRID == x][0], 2) if not np.isnan(curves[alg][GRID == x][0]) else None for x in (500, 1000, 2500)])
s = ["\\begin{tikzpicture}", "\t\\begin{axis}[width=0.62\\textwidth, height=6cm, xlabel={Distinct skill matrices simulated}, ylabel={Gap to the best design found (\\%)}, xmin=0, xmax=2500, ymin=0, ymax=25,",
     "\t\ttick label style={/pgf/number format/fixed}, scaled ticks=false, legend style={draw=none, font=\\footnotesize}, legend cell align=left, legend pos=north east, ylabel near ticks]"]
for alg, lab, col, style in METHODS:
    pts = [(x, 100 * y) for x, y in zip(GRID, curves[alg]) if not np.isnan(y)]
    s.append(f"\t\t\\addplot[{col}, {style}, line width=0.8pt, const plot] coordinates {{" + " ".join(f"({x},{y:.3f})" for x, y in pts) + "};")
    s.append(f"\t\t\\addlegendentry{{{lab}}}")
s += ["\t\\end{axis}", "\\end{tikzpicture}"]
name = "fig3_convergence"; open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(s) + "\n")
open(os.path.join(OUT, name + "_standalone.tex"), "w").write(PREAMBLE + f"\t\\input{{{name}.tex}}\n\\end{{document}}\n")
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + "_standalone.tex"], cwd=OUT, capture_output=True, text=True)
print(name, "compiled" if r.returncode == 0 else "FAILED:\n" + r.stdout[-1500:])
for ext in (".aux", ".log"):
    p = os.path.join(OUT, name + "_standalone" + ext); os.path.exists(p) and os.remove(p)
