"""table_B1.py -- Table for Section 6.1 (revision/tikz/B1_table.tex): per surge level delta and design (FCFS design
under FCFS, mu design under mu, hmu design under hmu): stability rate (share of the one-SKU surge scenarios stable in
simulation; LP feasibility in brackets if different), increase of the total cost with the stock fixed (mean with 95% CI,
median) and with the stock re-optimized (mean), over the stable scenarios. Also tables/B1_table.csv."""
import os, numpy as np, pandas as pd
from scipy import stats
from analyze_B1B23 import load, T
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision", "tikz", "B1_table.tex")
ARMS = [("FCFS design / FCFS", "Design for FCFS, FCFS"), ("mu design / mu", "Design for $\\mu$, $\\mu$ rule"), ("hmu design / hmu", "Design for $h\\times\\mu$, $h\\times\\mu$ rule"),
        ("mu-NP design / mu NP", "Design for non-preemptive $\\mu$"), ("mu-cov2 design / mu", "Design for $\\mu$, every SKU covered twice"), ("full CT / mu", "Full cross-training, $\\mu$ rule")]


def ci(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), v.mean() - h, v.mean() + h


if __name__ == "__main__":
    d = load("B1"); rows = []
    lines = [r"\begin{table}[!htb]", r"\centering",
             r"\caption{Stability and cost increase under one-SKU failure surges, for the design optimized for FCFS and the designs optimized for the $\mu$ and $h\times\mu$ rules, each operated under its own rule (1,920 scenarios per surge level and design, 35 replications each). Stability: share of scenarios in which the shop remains stable; cost increase: relative to the nominal total cost, over the stable scenarios, with the base-stock levels fixed (mean with 95\% confidence interval, and median) and with the base-stock levels re-optimized for the surge (mean).}",
             r"\label{tab:stability_cost_policies}", r"\small", r"\begin{tabular}{@{}cl r rr r@{}}", r"\toprule",
             r"& & & \multicolumn{2}{c}{Cost increase, stock fixed (\%)} & Stock re-optimized \\", r"\cmidrule(lr){4-5}",
             r"$\delta$ & Design and rule & Stable (\%) & Mean [95\% CI] & Median & Mean (\%) \\", r"\midrule"]
    for dl, sub in d.groupby("delta"):
        first = True
        for arm, label in ARMS:
            s = sub[sub.arm == arm]
            if not len(s): continue
            st = s[s.sim_stable]; m, lo, hi = ci(100 * st.inc_fixed); med = 100 * st.inc_fixed.median(); mr = 100 * st.inc_reopt.mean()
            stab = f"{100 * s.sim_stable.mean():.1f}" + (f" ({100 * s.lp_feasible.mean():.1f})" if abs(s.lp_feasible.mean() - s.sim_stable.mean()) > 5e-4 else "")
            lines.append(f"{f'{dl:.0%}'.replace('%', chr(92) + '%') if first else ''} & {label} & {stab} & {m:.2f} [{lo:.2f}, {hi:.2f}] & {med:.2f} & {mr:.2f} \\\\"); first = False
            rows.append({"delta": dl, "arm": arm, "n": len(s), "stable_sim": s.sim_stable.mean(), "stable_lp": s.lp_feasible.mean(), "inc_fixed_mean": m / 100, "inc_fixed_lo": lo / 100, "inc_fixed_hi": hi / 100, "inc_fixed_median": med / 100, "inc_reopt_mean": mr / 100})
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(lines) + "\n"); pd.DataFrame(rows).to_csv(os.path.join(T, "B1_table.csv"), index=False)
    print("\n".join(lines[11:-2]))
