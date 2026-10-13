"""
rerun_analysis.py -- shared code for the analysis notebooks of the re-optimized PSO and GA results.

Folders (relative to the fast_simulator folder, where the notebooks live):
  ../largeCases          original 2019 PSO result files (recorded costs, designs)
  ../PSO_results_rerun   PSO_optimization_v2 results (independent-seed costs)
  ../GA_results_rerun    GA_Optimization_v2 results (same format, "algorithm": "GA")
"""
import os, re, glob, json
import numpy as np, pandas as pd

# Test-bed factors as stored in the result files; labels for tables and figures.
FACTORS = {"N": "SKUs $N$", "M": "Technicians $M$", "rho": r"Utilization $\rho$", "hmin": r"$h_{\min}$",
           "h_variant": "Holding cost/workload", "f": "Technician cost $f$", "c_factor": "Skill cost $c/f$"}
# holding_costs_variant as stored in the result files is 0 or 2 (64 cases each; there is no 1). Variant 0: holding cost
# unrelated to repair workload (mean Spearman corr(h, lambda/mu) = -0.03); variant 2: holding cost decreasing in workload
# (-0.97). Labels as in the paper -- CONFIRM that HPB is the paper's name for the second relation.
H_VARIANT_LABELS = {0: "IND", 2: "HPB"}


def case_number(path):
    m = re.search(r"case_(\d+)", os.path.basename(path)); return int(m.group(1)) if m else None


def _check():
    try:
        from simulation_codes.simulation_fast import checkSkillAssignment
        return lambda X, lam, mu: bool(checkSkillAssignment(X, lam, mu)[0])
    except Exception:                                              # fallback: the same LP with scipy
        from scipy.optimize import linprog
        def chk(X, lam, mu, cap=0.999):
            M, N = X.shape
            if np.any(X.sum(axis=0) == 0): return False
            idx = np.argwhere(X.T == 1); k = len(idx)
            Ae = np.zeros((N, k)); Au = np.zeros((M, k))
            Ae[idx[:, 0], np.arange(k)] = 1; Au[idx[:, 1], np.arange(k)] = (lam / mu)[idx[:, 0]]
            return linprog(np.zeros(k), A_ub=Au, b_ub=np.full(M, cap), A_eq=Ae, b_eq=np.ones(N), bounds=(0, None), method="highs").status == 0
        return chk


def design_stats(X):
    X = np.asarray(X, int); emp = X.sum(axis=1) > 0
    return {"technicians": int(emp.sum()), "skills": int(X.sum()),
            "skills per technician": float(X[emp].sum(axis=1).mean()) if emp.any() else 0.0,
            "technicians per SKU": float(X.sum(axis=0).mean()), "max technicians per SKU": int(X.sum(axis=0).max()),
            "SKUs with one technician": int((X.sum(axis=0) == 1).sum())}


def evaluations(log):
    """Evaluations requested by the search: sum of nevals over the log (initial population included)."""
    return int(sum(e.get("nevals", 0) or 0 for e in log)) if log else None


def best_so_far(log):
    """Best total cost found up to each logged generation, from each generation's minimum; None if the log has no
    statistics (files written before the driver stored DEAP's chapters)."""
    if not log or not isinstance(log[0].get("total"), dict):
        return None
    mins = np.array([float(e["total"].get("min", np.inf)) if isinstance(e.get("total"), dict) else np.inf for e in log])
    return np.minimum.accumulate(mins)


def normalized_curve(log):
    """best_so_far divided by its final value; generations before the first feasible design are left out (NaN)."""
    b = best_so_far(log)
    if b is None or not np.isfinite(b[-1]): return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(b), b / b[-1], np.nan)


def load_folder(folder, label):
    """One row per result file: factors, costs, design statistics, feasibility, search information."""
    check = _check(); rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*case_*.json")), key=case_number):
        d = json.load(open(path))
        lam, mu = np.asarray(d["failure_rates"], float), np.asarray(d["service_rates"], float)
        X = np.asarray(d["best_assignment"], int)
        row = {"case": case_number(path), "source": label,
               "N": int(d["m_skills"]), "M": int(d["n_servers"]), "rho": d.get("utilization_rate"),
               "hmin": d.get("holding_cost_min"), "h_variant": d.get("holding_costs_variant"),
               "f": d.get("machine_cost"), "c_factor": d.get("skill_cost_factor"),
               "total": float(d["total_cost"]), "total_ff": float(d["total_cost_ff"]),
               "stock": float(d["Stock_Costs"]), "backorders": float(d["EBO_Costs"]),
               "skills cost": float(d["Skills_Costs"]), "technicians cost": float(d["Machine_Costs"]),
               "search_total": d.get("search_total_cost"), "feasible": check(X, lam, mu),
               "max stock": int(max(d["stocks"])), "runtime_h": float(d.get("runtime", np.nan)) / 3600,
               "evaluations": evaluations(d.get("log")), "algorithm": d.get("algorithm", "PSO (2019)" if label == "old" else label),
               "X": X, "log": d.get("log")}
        row.update(design_stats(X))
        row["saving"] = 1 - row["total"] / row["total_ff"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df): df["h_variant_label"] = df["h_variant"].map(H_VARIANT_LABELS)
    return df


def factor_summary(df, value, factors=("N", "M", "rho", "hmin", "h_variant", "f", "c_factor"), fmt="{:+.1%}"):
    """Mean, median, min and max of `value` at every level of every factor (one block per factor)."""
    out = []
    for f in factors:
        if f not in df or df[f].nunique() < 1: continue
        for lv, g in df.groupby(f):
            v = g[value].dropna()
            out.append({"factor": f, "level": lv, "cases": len(v), "mean": v.mean(), "median": v.median(), "min": v.min(), "max": v.max()})
    t = pd.DataFrame(out)
    return t


def latex_table(df, path, caption, label, formats=None, index=False):
    """A plain booktabs table (no jinja2 needed)."""
    formats = formats or {}
    cols = list(df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * (1 if index else 0) + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(([""] if index else []) + [str(c) for c in cols]) + r" \\", r"\midrule"]
    for i, r in df.iterrows():
        cells = [formats.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "--" for c in cols]
        lines.append(" & ".join(([str(i)] if index else []) + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def boxplots_by_factor(ax_row, df, value, groups, colors, labels, factors=("N", "M", "rho", "hmin", "h_variant", "f", "c_factor"), scale=100):
    """One panel per factor; within each level one box per group (column `groups[0]` selects the group)."""
    gcol, gvals = groups
    for ax, f in zip(ax_row, [f for f in factors if f in df and df[f].nunique() > 1]):
        levels = sorted(df[f].dropna().unique(), key=lambda v: (isinstance(v, str), v)); pos = 0; ticks, tl = [], []
        for lv in levels:
            for gi, gv in enumerate(gvals):
                vals = scale * df[(df[f] == lv) & (df[gcol] == gv)][value].dropna()
                if len(vals):
                    ax.boxplot([vals], positions=[pos], widths=0.7, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=colors[gi]), medianprops=dict(color="black"))
                pos += 1
            ticks.append(pos - (len(gvals) + 1) / 2); tl.append(H_VARIANT_LABELS.get(lv, lv) if f == "h_variant" else lv); pos += 1
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in tl], fontsize=8); ax.set_title(FACTORS.get(f, f), fontsize=9)
        ax.axhline(0, color="0.5", lw=0.6); ax.tick_params(labelsize=8)
