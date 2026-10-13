"""appendix_tables.py -- Appendix A tables from the tuning runs.
  A_t1_table.tex : T1 (small instances, lookup of the brute-force dataset, 20 particles x 50 generations, seeds 1-2): for
                   each parameter of the proposed BPSO, mean over the grid settings with that value of (i) the gap to the
                   optimum after 25% of the feasible designs were evaluated, (ii) the share of runs that end at the
                   optimum; and the same for the original BPSO (its grid: c1, c2, decreasing inertia from w_max to w_min).
  A_t2_table.tex : T2 (eight test-bed instances, budget 2,500 distinct designs, seeds 1-2): per setting, mean and median
                   gap to the best of the twenty runs of the instance, skills of the final design, runs that found the best.
Writes to revision/tikz/ and prints both."""
import glob, json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "..", "revision", "tikz")
NAMES = {"c1": "$c_1$", "c2": "$c_2$", "inertia_min": "$\\omega_{\\min}$", "inertia_max": "$\\omega_{\\max}$"}
T2 = [("pso_proposed_gap25", "Proposed, tuned: $c_1=2$, $c_2=4$, $\\omega$ $0.1\\to2.0$ (selected)"), ("pso_proposed_ext_gap25", "Proposed: $c_1=2$, $c_2=3$, $\\omega$ $0.001\\to8$"),
      ("pso_reset_gap25", "Proposed with velocity re-drawn: $c_1=2$, $c_2=1$, $\\omega$ $0.001\\to8$"), ("pso", "Proposed, proof of concept: $c_1=1$, $c_2=3$, $\\omega$ $0.01\\to1.5$"),
      ("pso_proposed_optimal", "Proposed: $c_1=1$, $c_2=1$, $\\omega$ $0.01\\to1.0$"), ("pso_proposed_ext_optimal", "Proposed: $c_1=1$, $c_2=3$, $\\omega$ $0.001\\to1.0$"),
      ("pso_reset_optimal", "Proposed with velocity re-drawn: $c_1=2$, $c_2=1$, $\\omega$ $0.1\\to1.0$, $p=2$"),
      ("pso_orig_original_gap25", "Original, tuned: $c_1=c_2=2$, $\\omega$ $0.9\\to0.01$"), ("pso_orig_original_optimal", "Original: $c_1=0.5$, $c_2=2$, $\\omega$ $0.3\\to0.1$")]


def t1():
    d = pd.concat([pd.read_csv(f) for f in (os.path.join(HERE, "t1_results_seed1.csv"), os.path.join(HERE, "t1_results_seed2.csv"))], ignore_index=True)
    d = d[np.isfinite(d.gap) & np.isfinite(d.gap_at_25)]; blocks = []
    for variant, title in (("proposed", "Proposed BPSO (increasing inertia)"), ("original", "Original BPSO (decreasing inertia)")):
        v = d[d.variant == variant]; cols = []
        for par in ("c1", "c2", "inertia_min", "inertia_max"):
            g = v.groupby(par).agg(gap25=("gap_at_25", "mean"), opt=("optimal", "mean")).reset_index(); cols.append((par, g))
        blocks.append((title, cols, v.param_id.nunique(), v.groupby(["instance", "setting"]).ngroups, len(v)))
    lines = [r"\begin{table}[ht!]", r"\centering", r"\caption{Tuning on the small instances: for each value of a parameter, the mean over the settings of the grid with that value of the relative gap to the optimum after one quarter of the feasible designs has been simulated (gap$_{25}$) and the share of runs that end at the optimum (opt.).}", r"\label{tab:tunig_results_means}", r"\small"]
    for title, cols, nset, ninst, nruns in blocks:
        lines += [f"\\textit{{{title}: {nset} settings, {ninst:,} instances, {nruns:,} runs}}\\\\[2pt]", r"\begin{tabular}{" + "lrr " * 4 + "}", r"\toprule"]
        lines.append(" & ".join(f"{NAMES[p]} & gap$_{{25}}$ & opt." for p, _ in cols) + r" \\ \midrule")
        n = max(len(g) for _, g in cols)
        for i in range(n):
            cells = []
            for p, g in cols:
                cells.append(f"{g.iloc[i][p]:g} & {100 * g.iloc[i].gap25:.2f}\\% & {100 * g.iloc[i].opt:.1f}\\%" if i < len(g) else " & & ")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}\\[6pt]"]
    lines += [r"\end{table}"]
    open(os.path.join(OUT, "A_t1_table.tex"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    best = d[d.variant == "proposed"].groupby(["c1", "c2", "inertia_min", "inertia_max"]).agg(gap25=("gap_at_25", "mean"), opt=("optimal", "mean"), final=("gap", "mean")).reset_index()
    print("\nproposed grid: best by gap25:\n", best.sort_values("gap25").head(3).to_string(index=False), "\nbest by opt:\n", best.sort_values("opt", ascending=False).head(3).to_string(index=False))
    print("final gap range over settings: %.3f%% - %.3f%%; gap25 range %.2f%% - %.2f%%" % (100 * best.final.min(), 100 * best.final.max(), 100 * best.gap25.min(), 100 * best.gap25.max()))


def t2():
    rows = []
    for f in glob.glob(os.path.join(HERE, "..", "..", "T2_results", "*", "*.json")):
        r = json.load(open(f)); rows.append({"label": os.path.basename(os.path.dirname(f)), "case": r["case"], "seed": r["seed"], "cost": r["total_cost"], "skills": r["skills"], "designs": r["designs_simulated"]})
    d = pd.DataFrame(rows); d = d[d.label.isin([k for k, _ in T2])]
    d["best"] = d.groupby("case").cost.transform("min"); d["gap"] = d.cost / d.best - 1
    g = d.groupby("label").agg(mean=("gap", "mean"), median=("gap", "median"), best=("gap", lambda s: int((s < 1e-9).sum())), skills=("skills", "mean"), n=("gap", "size")).reindex([k for k, _ in T2])
    lines = [r"\begin{table}[ht!]", r"\centering", r"\caption{Confirmation on eight instances of the test bed ($N\in\{10,20\}$, $M\in\{5,10\}$, $c\in\{0.01f,0.1f\}$; budget of 2,500 distinct simulated designs, two seeds): gap of the final design to the cheapest design found by any setting on the instance, mean and median over the 16 runs; number of skills of the final design.}",
             r"\label{tab:tunig_results_max}", r"\small", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Setting & Mean gap (\%) & Median gap (\%) & Skills \\", r"\midrule"]
    for k, lab in T2:
        if k in g.index and g.loc[k, "n"] > 0: lines.append(f"{lab} & {100 * g.loc[k, 'mean']:.1f} & {100 * g.loc[k, 'median']:.1f} & {g.loc[k, 'skills']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "A_t2_table.tex"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print(g.round(4).to_string())


if __name__ == "__main__":
    t1(); t2()
