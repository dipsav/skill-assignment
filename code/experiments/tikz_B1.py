"""tikz_B1.py -- Figure for Section 6.1 as pgfplots (revision/tikz/fig8_sku_surges.tex), in the style of the R2 surge
box plots (one panel per surge level, one box per design/rule): relative increase of the total cost with the stock
fixed, (TC_surge - TC_nominal)/TC_nominal, over the one-SKU surge scenarios that are stable under the design; box
statistics computed here (boxplot prepared), outliers drawn as marks. Data: tables/B1_*.csv via analyze_B1B23.load."""
import os, subprocess
import numpy as np, pandas as pd
from analyze_B1B23 import load
from tikz_figures import PREAMBLE, OUT

ARMS = [("FCFS design / FCFS", "red", "FCFS"), ("mu design / mu", "blue", "$\\mu$"), ("hmu design / hmu", "black!60!green", "$h\\times\\mu$")]
YMAX = 0.5


def prepared(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75]); iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min(); hi = v[v <= q3 + 1.5 * iqr].max(); out = v[(v < lo) | (v > hi)]
    return q1, med, q3, lo, hi, out


if __name__ == "__main__":
    d = load("B1"); d = d[d.sim_stable]; deltas = sorted(d.delta.unique()); arms = [x for x in ARMS if x[0] in set(d.arm)]
    s = ["\\begin{tikzpicture}", f"\t\\begin{{groupplot}}[group style={{group size={len(deltas)} by 1, horizontal sep=0.9cm, ylabels at = edge left}},",
         "\t\tx post scale = 0.45, boxplot/draw direction=y, boxplot/every box/.style={fill, fill opacity=0.5}, tick label style={/pgf/number format/fixed}, scaled ticks=false,",
         f"\t\tymin=-0.02, ymax={YMAX}, ylabel={{$\\frac{{TC_{{\\mbox{{\\scriptsize surge}}}} - TC_{{\\mbox{{\\scriptsize nominal}}}}}}{{TC_{{\\mbox{{\\scriptsize nominal}}}}}}$}}, ylabel style={{at={{(axis description cs:0.3,0.5)}}, anchor=south}}, ylabel near ticks,",
         f"\t\txtick={{{','.join(str(i + 1) for i in range(len(arms)))}}}, xticklabels={{{', '.join(x[2] for x in arms)}}}, xlabel style={{align=center}}]"]
    for dl in deltas:
        s.append(f"\t\\nextgroupplot[title={{$\\delta={dl:.0%}$}}, title style={{font=\\small}}]".replace("%", "\\%"))
        for i, (arm, color, lab) in enumerate(arms, start=1):
            v = d[(d.arm == arm) & (d.delta == dl)].inc_fixed.dropna().values
            q1, med, q3, lo, hi, out = prepared(v); out = out[out <= YMAX * 1.5]
            head = f"\t\t\\addplot[boxplot prepared={{lower whisker={lo:.4f}, lower quartile={q1:.4f}, median={med:.4f}, upper quartile={q3:.4f}, upper whisker={hi:.4f}, draw position={i}}}, color={color}, mark=*, mark size=1pt]"
            s.append(head + (" table[y index=0] {\n" + "\n".join(f"{o:.4f}" for o in out) + "\n};" if len(out) else " coordinates {};"))
            print(f"delta {dl:.0%} {arm}: n {len(v)}, median {med:.2%}, q3 {q3:.2%}, upper whisker {hi:.2%}, outliers {len(out)} (above axis {int((v > YMAX).sum())})")
    s += ["\t\\end{groupplot}", "\\end{tikzpicture}"]
    name = "fig8_sku_surges"; open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(s) + "\n")
    open(os.path.join(OUT, name + "_standalone.tex"), "w").write(PREAMBLE + f"\t\\input{{{name}.tex}}\n\\end{{document}}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + "_standalone.tex"], cwd=OUT, capture_output=True, text=True)
    print(name, "compiled" if r.returncode == 0 else "FAILED:\n" + r.stdout[-2000:])
    for ext in (".aux", ".log"):
        p = os.path.join(OUT, name + "_standalone" + ext); os.path.exists(p) and os.remove(p)
