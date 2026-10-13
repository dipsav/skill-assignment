"""
tikz_figures.py -- Figures 4-6 as pgfplots/TikZ sources in the style of the R2 figures (Archive/figs/pso_savings.tex,
priority.tex, utilizations.tex): one group plot, seven panels (one per test-bed factor), red = low level, blue = high
level; box plots computed by pgfplots from a \\pgfplotstableread table with one column per factor level (64 instances
each); utilizations as jittered marks (one \\addplot per instance, xshift -7.75pt ... 8pt).

Writes revision/tikz/fig4_savings.tex, fig5_rules.tex, fig6_utilization.tex (to \\input in the manuscript) and
*_standalone.tex, compiled here with pdflatex for checking. Data: paper_figures.designs() (final designs) and
tables/R1_costs_tuned_polished.csv (rule applied to the FCFS design, stock re-optimized).
"""
import os, subprocess
import numpy as np, pandas as pd
from paper_figures import designs, HERE, T

OUT = os.path.join(HERE, "..", "revision", "tikz")
# (column stem, factor key, level, x tick label, panel xlabel)
PANELS = [("n_servers", "M", (5, 10), "5, 10", "Number \\\\of Technicians"), ("m_skills", "N", (10, 20), "10, 20", "Number \\\\of Skills (SKUs)"),
          ("utilization_rate", "rho", (0.65, 0.8), "0.65, 0.80", "Initial Utilization \\\\Rate"), ("holding_cost_min", "hmin", (1, 100), "1, 100", "Minimum \\\\Holding Cost"),
          ("holding_costs_variant", "hv", (0, 2), "IND\\ , \\ HPB", "Holding \\\\Cost Variant"), ("skill_cost_factor", "c", (0.01, 0.1), "0.01$f$\\ \\ , \\ \\ 0.10$f$", "Cross Training \\\\Cost Factor"),
          ("machine_cost_factor", "f", (10, 100), "10, 100", "Technician \\\\Cost Factor")]
PREAMBLE = r"""\documentclass{standalone}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage{tikz}
\usetikzlibrary{plotmarks, pgfplots.statistics}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc, chains, shapes, plotmarks, pgfplots.statistics}
\pgfplotsset{compat=newest}
\begin{document}
"""
GROUP_OPTS = r"""		xtick={1,2},
		x post scale = 0.3,
		boxplot/draw direction=y,
		boxplot/every box/.style={fill, fill opacity=0.5},
		tick label style={/pgf/number format/fixed},
		scaled ticks=false,
   		ylabel style={at={(axis description cs:0.3,0.5)}, anchor=south},
   		xlabel style={align=center},
	    ylabel near ticks,
"""


def col(stem, level): return f"{stem}{level:g}"


def table(df, value, name):
    """pgfplotstable with one column per factor level (64 instances each, in instance order)."""
    cols = {"case": np.arange(64)}
    for stem, key, levels, _, _ in PANELS:
        for lv in levels:
            cols[col(stem, lv)] = df[df[key] == lv].sort_values("case")[value].values
    t = pd.DataFrame(cols); body = t.to_string(index=False, float_format=lambda x: f"{x:.6f}")
    return "\\pgfplotstableread{\n" + body + "\n}" + name + "\n\n"


def box_rows(rows, name, ylabels, ymin_max, xlabel_rows=1):
    """rows: list of table macro names (one group-plot row each); ylabels per row; (ymin, ymax) per row."""
    s = "\\begin{tikzpicture}\n\t\\begin{groupplot}[group style={group size=7 by %d, horizontal sep=1cm, vertical sep=1.6cm, ylabels at = edge left},\n%s\t\t]\n" % (len(rows), GROUP_OPTS)
    for r, (tab, yl, (lo, hi)) in enumerate(zip(rows, ylabels, ymin_max)):
        for i, (stem, key, levels, ticks, xl) in enumerate(PANELS):
            opts = f"xticklabels={{{ticks}}}, ymin={lo}, ymax={hi}" + (f", xlabel={{{xl}}}" if r == len(rows) - 1 or xlabel_rows == len(rows) else "") + (f", ylabel={{{yl}}}" if i == 0 else "")
            s += f" \t\\nextgroupplot[{opts}]\n"
            s += f"\t\t\\addplot[boxplot,mark=*,color=red] \ttable[y={col(stem, levels[0])}] {{{tab}}};\n"
            s += f"\t\t\\addplot[boxplot,mark=*,color=blue] \ttable[y={col(stem, levels[1])}] {{{tab}}};\n\n"
    return s + "\t\\end{groupplot}\n\\end{tikzpicture}\n"


def strip_rows(dfs, ylabels):
    """Utilization strips: per panel and level, one \\addplot per instance with a jitter in xshift (as utilizations.tex)."""
    s = "\\begin{tikzpicture}[scale = 0.7]\n\t\\begin{groupplot}[group style={group size=7 by %d, horizontal sep=1cm, vertical sep=1.8cm, ylabels at = edge left},\n" % len(dfs)
    s += "\t\txtick={1,2},\n\t\tx post scale = 0.3,\n\t\ttick label style={/pgf/number format/fixed},\n   \t\tylabel style={at={(axis description cs:0.3,0.5)}, anchor=south},\n   \t\txlabel style={align=center},\n\t    ylabel near ticks,\n\t    xmin=0.2, xmax=2.8,\n\t    ymin=-0.05, ymax=1.05\n\t\t]\n"
    for r, (df, yl) in enumerate(zip(dfs, ylabels)):
        for i, (stem, key, levels, ticks, xl) in enumerate(PANELS):
            opts = f"xticklabels={{{ticks}}}" + (f", xlabel={{{xl}}}" if r == len(dfs) - 1 else "") + (f", ylabel={{{yl}}}" if i == 0 else "")
            s += f"\n\\nextgroupplot[{opts}]\n"
            for x, (lv, color) in enumerate(zip(levels, ("red", "blue")), start=1):
                sub = df[df[key] == lv].sort_values("case")
                for j, u in enumerate(sub.util.values):
                    shift = -7.75 + 0.25 * j
                    s += f"    \\addplot[{color}, only marks, mark=*, mark size=0.5pt, xshift={shift:.2f}pt] coordinates {{" + " ".join(f"({x}, {v:.10f})" for v in u) + " };\n"
    return s + "\t\\end{groupplot}\n\\end{tikzpicture}\n"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name + ".tex"), "w").write(body)
    open(os.path.join(OUT, name + "_standalone.tex"), "w").write(PREAMBLE + f"\t\\input{{{name}.tex}}\n\\end{{document}}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + "_standalone.tex"], cwd=OUT, capture_output=True, text=True)
    print(name, "compiled" if r.returncode == 0 else "FAILED:\n" + r.stdout[-1500:])
    for ext in (".aux", ".log"):
        p = os.path.join(OUT, name + "_standalone" + ext); os.path.exists(p) and os.remove(p)


if __name__ == "__main__":
    fc = designs("PSO_results_tuned_polished"); mu = designs("PSO_results_tuned_rule_mu_polished")
    fc["saving_ff"] = 1 - fc.total / fc.total_ff
    r1 = pd.read_csv(os.path.join(T, "R1_costs_tuned_polished.csv")); r1 = r1[r1.stock == "reoptimized"].pivot(index="instance", columns="rule", values="total")
    fc["seq_mu"] = 1 - r1["mu"].reindex(fc.case).values / fc.total; fc["joint_mu"] = 1 - mu.set_index("case").total.reindex(fc.case).values / fc.total
    # Fig. 4
    write("fig4_savings", table(fc, "saving_ff", "\\SavingsFF") + box_rows(["\\SavingsFF"], "", [r"$\frac{TC_{\mbox{\scriptsize Full}} - TC_{\mbox{\scriptsize BPSO}}}{TC_{\mbox{\scriptsize Full}}}$"], [(-0.03, 0.98)]))
    # Fig. 5: (a) mu rule applied to the FCFS design (stock re-optimized), (b) design optimized for the mu rule
    write("fig5_rules", table(fc, "seq_mu", "\\RuleOnFCFSDesign") + table(fc, "joint_mu", "\\JointDesign")
          + box_rows(["\\RuleOnFCFSDesign", "\\JointDesign"], "",
                     [r"(a) $\mu$ rule on the FCFS design\\ $\frac{TC_{\mbox{\scriptsize FCFS}} - TC_{\mu}}{TC_{\mbox{\scriptsize FCFS}}}$", r"(b) design optimized for the $\mu$ rule\\ $\frac{TC_{\mbox{\scriptsize FCFS}} - TC_{\mu}}{TC_{\mbox{\scriptsize FCFS}}}$"],
                     [(-1.0, 0.4), (-0.1, 0.35)]).replace("ylabel style={at={(axis description cs:0.3,0.5)}, anchor=south}", "ylabel style={at={(axis description cs:0.25,0.5)}, anchor=south, align=center}"))
    # Fig. 6: utilizations, (a) FCFS design under FCFS, (b) mu design under the mu rule
    # one file per row (two subfigures, as in R2): the ~900 \\addplot of a row are at the limit of pdflatex's memory
    write("fig6a_utilization", strip_rows([fc], ["Technician Utilization Rates"]))
    write("fig6b_utilization", strip_rows([mu], ["Technician Utilization Rates"]))
