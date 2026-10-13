"""analyze_E3.py -- sensitivity of the final designs to the repair-time and interarrival-time distributions (E3).
Reads tables/E3_fcfs_designs.csv and E3_mu_designs.csv; per grid cell and design: share of instances stable, median and
mean change of the total cost relative to the exponential cell (stock fixed and re-optimized), median units in stock
re-optimized; paired saving of the mu design over the FCFS design in each cell (stock fixed / re-optimized).
Writes tables/E3_summary.csv and revision/tikz/E3_table.tex."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, "tables"); OUT = os.path.join(HERE, "..", "revision", "tikz", "E3_table.tex")
CELLS = [("base", "Exponential (the model)", "1", "1"), ("svc_erlang4", "Repair Erlang-4", "1", "0.5"), ("svc_erlang2", "Repair Erlang-2", "1", "0.71"),
         ("svc_logn1.5", "Repair lognormal", "1", "1.5"), ("svc_logn2", "Repair lognormal", "1", "2"), ("arr_erlang4", "Failures Erlang-4", "0.5", "1"),
         ("arr_h2", "Failures hyperexponential", "2", "1"), ("both_low", "Both Erlang-4", "0.5", "0.5"), ("both_high", "Failures H$_2$, repair lognormal", "2", "2")]
ARM = {"PSO_results_tuned_polished": "FCFS", "PSO_results_tuned_rule_mu_polished": "mu"}


def load():
    d = pd.concat([pd.read_csv(os.path.join(T, f)) for f in ("E3_fcfs_designs.csv", "E3_mu_designs.csv") if os.path.exists(os.path.join(T, f))], ignore_index=True)
    d["design"] = d.design_set.map(ARM)
    for col in ("total_fixed", "total_reopt"): d[col] = d[col].replace([np.inf, -np.inf], np.nan)
    base = d[d.cell == "base"].set_index(["design", "instance"])
    d = d.join(base[["total_fixed", "total_reopt"]].rename(columns={"total_fixed": "base_fixed", "total_reopt": "base_reopt"}), on=["design", "instance"])
    d["chg_fixed"] = d.total_fixed / d.base_fixed - 1; d["chg_reopt"] = d.total_reopt / d.base_reopt - 1
    return d


if __name__ == "__main__":
    d = load(); pd.set_option("display.width", 250)
    g = d.groupby(["cell", "design"]).agg(n=("instance", "size"), stable=("sim_stable", "mean"), chg_fixed_med=("chg_fixed", "median"), chg_fixed_mean=("chg_fixed", "mean"),
                                        chg_reopt_med=("chg_reopt", "median"), chg_reopt_mean=("chg_reopt", "mean"), units=("units_reopt", "median"), max_util=("max_util", "median")).reset_index()
    g["order"] = g.cell.map({c[0]: i for i, c in enumerate(CELLS)}); g = g.sort_values(["order", "design"]); print(g.round(3).to_string(index=False)); g.to_csv(os.path.join(T, "E3_summary.csv"), index=False)
    # paired: mu design vs FCFS design within each cell
    rows = []
    for key, name, cva, cvs in CELLS:
        s = d[d.cell == key].pivot(index="instance", columns="design", values=["total_fixed", "total_reopt", "sim_stable"])
        if "mu" not in s["total_fixed"] or "FCFS" not in s["total_fixed"]: continue
        both = s["sim_stable"]["FCFS"].astype(bool) & s["sim_stable"]["mu"].astype(bool)
        r = {"cell": key, "name": name, "cv_a": cva, "cv_s": cvs, "stable_fcfs": s["sim_stable"]["FCFS"].mean(), "stable_mu": s["sim_stable"]["mu"].mean(), "n_both": int(both.sum())}
        for v in ("total_fixed", "total_reopt"):
            sv = (1 - s[v]["mu"][both].astype(float) / s[v]["FCFS"][both].astype(float)).dropna()
            r[f"save_{v}_mean"] = sv.mean(); r[f"save_{v}_med"] = sv.median(); r[f"save_{v}_pos"] = int((sv > 0).sum()); r[f"save_{v}_p"] = wilcoxon(sv).pvalue if len(sv) > 10 else np.nan
        for dsg in ("FCFS", "mu"):
            sub = d[(d.cell == key) & (d.design == dsg)]; r[f"chg_fixed_{dsg}"] = sub.chg_fixed.median(); r[f"chg_reopt_{dsg}"] = sub.chg_reopt.median()
        rows.append(r)
    p = pd.DataFrame(rows); print("\npaired saving of the mu design over the FCFS design per cell:"); print(p.round(3).to_string(index=False))
    lines = [r"\begin{table}[!htb]", r"\centering",
             r"\caption{Sensitivity of the final designs to the failure and repair-time distributions, 128 instances, 35 replications per cell with common random numbers. CV: coefficient of variation of the interarrival time of failures (CV$_a$) and of the repair time (CV$_s$); the published model has both equal to one. Change: median change of the total cost of the design relative to the exponential case, with the base-stock levels fixed and re-optimized for the distribution; stable: share of instances stable in simulation; saving: cost of the design optimized for the $\mu$ rule relative to the design optimized for FCFS, each under its own rule, mean and median over the instances stable under both, and the number of instances in which the $\mu$ design is cheaper.}",
             r"\label{tab:distribution_sensitivity}", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lcc rr rr rr rrr@{}}", r"\toprule",
             r"& & & \multicolumn{2}{c}{Design for FCFS} & \multicolumn{2}{c}{Design for $\mu$} & \multicolumn{2}{c}{Stable (\%)} & \multicolumn{3}{c}{Saving of the $\mu$ design (\%)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
             r"Distributions & CV$_a$ & CV$_s$ & fixed & re-opt. & fixed & re-opt. & FCFS & $\mu$ & mean & median & cheaper \\", r"\midrule"]
    for _, r in p.iterrows():
        lines.append(f"{r['name']} & {r.cv_a} & {r.cv_s} & {100 * r.chg_fixed_FCFS:+.1f} & {100 * r.chg_reopt_FCFS:+.1f} & {100 * r.chg_fixed_mu:+.1f} & {100 * r.chg_reopt_mu:+.1f} & {100 * r.stable_fcfs:.1f} & {100 * r.stable_mu:.1f} & {100 * r.save_total_reopt_mean:.1f} & {100 * r.save_total_reopt_med:.1f} & {r.save_total_reopt_pos} of {r.n_both} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(lines) + "\n"); print("\n" + "\n".join(lines[12:-2]))
