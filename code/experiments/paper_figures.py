"""
paper_figures.py -- Figures 4-6 of the revised manuscript from the final designs (same layout as the R2 figures:
one panel per test-bed factor, red = low level, blue = high level).

  Fig. 4  figs/fig4_savings.pdf      saving of the optimized skill assignment against full cross-training,
                                     (TC_FF - TC)/TC_FF, FCFS designs of PSO_results_tuned_polished (Section 5.2);
                                     also tables/savings_by_factor_final.csv/.tex.
  Fig. 5  figs/fig5_rules.pdf        (a) mu rule applied to the FCFS design, stock re-optimized (Section 5.3), and
                                     (b) design optimized for the mu rule (Section 5.4), both as the saving relative
                                     to the FCFS design under FCFS, (TC_FCFS - TC_mu)/TC_FCFS.
  Fig. 6  figs/fig6_utilization.pdf  technician utilization by factor, (a) FCFS design under FCFS, (b) mu design
                                     under the mu rule (employed technicians only).
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); FIGS = os.path.join(HERE, "figs"); T = os.path.join(HERE, "tables")
FACTORS = [("M", "Number\nof Technicians"), ("N", "Number\nof Skills (SKUs)"), ("rho", "Initial Utilization\nRate"), ("hmin", "Minimum\nHolding Cost"),
           ("hv", "Holding\nCost Variant"), ("c", "Cross Training\nCost Factor"), ("f", "Technician\nCost Factor")]
LEVELS = {"hv": {0: "IND", 1: "HPB", 2: "HPB"}, "c": {0.01: "0.01$f$", 0.1: "0.10$f$"}, "rho": {0.65: "0.65", 0.8: "0.80"}}
RED, BLUE = "#e8706a", "#6f8fd8"


def case_number(p): return int(re.search(r"case_(\d+)", os.path.basename(p)).group(1))


def designs(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, folder, "*case_*.json")), key=case_number):
        d = json.load(open(p)); X = np.asarray(d["best_assignment"], int); u = np.asarray(d["utilizationRatesAverage"], float).sum(axis=1)
        rows.append({"case": case_number(p), "N": d["m_skills"], "M": d["n_servers"], "rho": d["utilization_rate"], "hmin": d["holding_cost_min"],
                     "hv": d["holding_costs_variant"], "f": d["machine_cost_factor"], "c": d["skill_cost_factor"], "total": d["total_cost"], "total_ff": d["total_cost_ff"],
                     "util": u[X.sum(axis=1) > 0]})
    return pd.DataFrame(rows)


def panel_boxes(ax, df, value, scale=100):
    """One panel row: per factor, a red box (low level) and a blue box (high level) of `value`."""
    for a, (f, title) in zip(ax, FACTORS):
        lv = sorted(df[f].unique()); vals = [scale * df[df[f] == l][value].values for l in lv]
        bp = a.boxplot(vals, positions=[0, 1], widths=0.6, patch_artist=True, medianprops=dict(color="black"), flierprops=dict(marker="o", ms=2.5, mec="none"))
        for patch, col, fl in zip(bp["boxes"], (RED, BLUE), bp["fliers"]): patch.set_facecolor(col); fl.set_markerfacecolor(col)
        a.set_xticks([0, 1]); a.set_xticklabels([LEVELS.get(f, {}).get(l, f"{l:g}") for l in lv], fontsize=8); a.set_xlabel(title, fontsize=8, labelpad=3)
        a.tick_params(labelsize=8); a.set_xlim(-0.6, 1.6)
    for a in ax[1:]: a.tick_params(labelleft=False)


def panel_strips(ax, df):
    rng = np.random.default_rng(0)
    for a, (f, title) in zip(ax, FACTORS):
        for i, (l, col) in enumerate(zip(sorted(df[f].unique()), (RED, BLUE))):
            u = np.concatenate(df[df[f] == l].util.values); a.scatter(i + rng.uniform(-0.3, 0.3, len(u)), u, s=2.5, color=col, alpha=0.6, lw=0)
        a.set_xticks([0, 1]); a.set_xticklabels([LEVELS.get(f, {}).get(l, f"{l:g}") for l in sorted(df[f].unique())], fontsize=8); a.set_xlabel(title, fontsize=8, labelpad=3)
        a.set_ylim(0, 1.02); a.tick_params(labelsize=8); a.set_xlim(-0.6, 1.6)
    for a in ax[1:]: a.tick_params(labelleft=False)


def by_factor(df, value):
    out = []
    for f, title in FACTORS:
        for l, g in df.groupby(f):
            out.append({"factor": title.replace("\n", " "), "level": LEVELS.get(f, {}).get(l, f"{l:g}"), "cases": len(g), "mean": g[value].mean(), "median": g[value].median(), "min": g[value].min(), "max": g[value].max()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True); plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fc = designs("PSO_results_tuned_polished"); mu = designs("PSO_results_tuned_rule_mu_polished"); hmu = designs("PSO_results_tuned_rule_hmu_polished")
    fc["saving_ff"] = 1 - fc.total / fc.total_ff
    print(f"Fig. 4: saving vs full CT: mean {fc.saving_ff.mean():.1%}, median {fc.saving_ff.median():.1%}, min {fc.saving_ff.min():.1%}, max {fc.saving_ff.max():.1%}")
    t = by_factor(fc, "saving_ff"); t.to_csv(os.path.join(T, "savings_by_factor_final.csv"), index=False); print(t.round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 7, figsize=(7.5, 2.4), sharey=True); panel_boxes(ax, fc, "saving_ff", 1)
    ax[0].set_ylabel(r"$\frac{TC_{\mathrm{FF}}-TC_{\mathrm{BPSO}}}{TC_{\mathrm{FF}}}$", fontsize=11); fig.tight_layout(w_pad=0.4); fig.savefig(os.path.join(FIGS, "fig4_savings.pdf")); plt.close(fig)

    # Fig. 5: (a) mu rule on the FCFS design (stock re-optimized), (b) design optimized for mu; saving relative to the FCFS design under FCFS
    r1 = pd.read_csv(os.path.join(T, "R1_costs_tuned_polished.csv")); r1 = r1[(r1.stock == "reoptimized")].pivot(index="instance", columns="rule", values="total")
    fc = fc.set_index("case"); fc["seq_mu"] = 1 - r1["mu"].reindex(fc.index) / fc.total; fc["seq_hmu"] = 1 - r1["hmu"].reindex(fc.index) / fc.total
    fc["joint_mu"] = 1 - mu.set_index("case").total.reindex(fc.index) / fc.total; fc["joint_hmu"] = 1 - hmu.set_index("case").total.reindex(fc.index) / fc.total
    for v in ("seq_mu", "seq_hmu", "joint_mu", "joint_hmu"):
        print(f"Fig. 5 {v}: mean {fc[v].mean():+.1%}, median {fc[v].median():+.1%}, min {fc[v].min():+.1%}, max {fc[v].max():+.1%}, positive in {(fc[v] > 0).sum()}/128")
    fig, ax = plt.subplots(2, 7, figsize=(7.5, 4.6), sharey="row")
    panel_boxes(ax[0], fc.reset_index(), "seq_mu", 1); panel_boxes(ax[1], fc.reset_index(), "joint_mu", 1)
    ax[0, 0].set_ylim(-1.0, 0.4); ax[0, 0].set_ylabel(r"$\frac{TC_{\mathrm{FCFS}}-TC_{\mu}}{TC_{\mathrm{FCFS}}}$" + "\n(a) rule applied to the FCFS design", fontsize=9)
    ax[1, 0].set_ylabel(r"$\frac{TC_{\mathrm{FCFS}}-TC_{\mu}}{TC_{\mathrm{FCFS}}}$" + "\n(b) design optimized for the rule", fontsize=9)
    for a in ax.ravel(): a.axhline(0, color="0.5", lw=0.6, zorder=0)
    fig.tight_layout(w_pad=0.4, h_pad=1.5); fig.savefig(os.path.join(FIGS, "fig5_rules.pdf")); plt.close(fig)
    print(f"Fig. 5(a) clipped below -100%: {(fc.seq_mu < -1).sum()} instances (min {fc.seq_mu.min():+.0%})")

    # Fig. 6: utilizations
    fig, ax = plt.subplots(2, 7, figsize=(7.5, 4.6), sharey="row"); panel_strips(ax[0], fc.reset_index()); panel_strips(ax[1], mu)
    ax[0, 0].set_ylabel("Technician utilization\n(a) design for FCFS, under FCFS", fontsize=9); ax[1, 0].set_ylabel("Technician utilization\n(b) design for the $\\mu$ rule, under $\\mu$", fontsize=9)
    fig.tight_layout(w_pad=0.4, h_pad=1.5); fig.savefig(os.path.join(FIGS, "fig6_utilization.pdf")); plt.close(fig)
    for lab, d in (("FCFS design", fc), ("mu design", mu), ("hmu design", hmu)):
        u = np.concatenate(d.util.values); busiest = d.util.apply(np.max)
        print(f"Fig. 6 {lab}: technicians {len(u)}, utilization mean {u.mean():.3f}, share > 0.9 {np.mean(u > 0.9):.1%}, busiest per instance mean {busiest.mean():.3f}, above 0.95 in {(busiest > 0.95).sum()}/128")
