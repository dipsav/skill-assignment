"""
resilience_tradeoff.py -- Section 6.4: nominal cost vs cost under disruption for every design / rule / discipline /
operational response. Per instance, relative to the design optimized for FCFS operated under FCFS:
  nominal   = nominal total cost of the arm (the stock it runs with) / nominal cost of the FCFS design;
  disrupted = (nominal cost rate + extra backorder cost rate over the disruption window) / the same for... the FCFS
              design's nominal cost rate, i.e. (nominal_arm + loss_rel * cost_ref_arm) / nominal_FCFS,
              where loss_rel is the episode's extra backorder cost in multiples of the arm's reference cost over a
              period of the disruption's length (analyze_B4 / analyze_B5).
Reference disruptions: all-SKU surge gamma 1.5 x 10,000 failures (B4); busiest / random technician absent 10,000
failures (B5). Also the permanent all-SKU +10% rise (B2, stock re-optimized) and the structural indicators (R0).
Writes tables/R_tradeoff.csv and revision/tikz/R_tradeoff_table.tex.
"""
import glob, json, os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); T = os.path.join(HERE, "tables")


def case(p): return int(re.search(r"case_(\d+)", os.path.basename(p)).group(1))


def costs(folder, key="total_cost"):
    return {case(p): json.load(open(p))[key] for p in glob.glob(os.path.join(ROOT, folder, "*case_*.json"))}


F = costs("PSO_results_tuned_polished"); MU = costs("PSO_results_tuned_rule_mu_polished"); HMU = costs("PSO_results_tuned_rule_hmu_polished")
FF = costs("PSO_results_tuned_polished", "total_cost_ff"); FFMU = costs("PSO_results_fullCT_rule_mu")
MUNP = costs("PSO_results_tuned_rule_mu_np_polished"); HMUNP = costs("PSO_results_tuned_rule_hmu_np_polished"); MUC2 = costs("PSO_results_tuned_rule_mu_cov2_polished"); FC2 = costs("PSO_results_tuned_cov2_polished")
P1 = pd.read_csv(os.path.join(T, "P1_preemption.csv")); P1 = P1[P1.cell == "exp"]
p1 = lambda dg, r, pre: dict(zip(P1[(P1.design == dg) & (P1.rule == r) & (P1.preemptive == pre)].instance, P1[(P1.design == dg) & (P1.rule == r) & (P1.preemptive == pre)].total_fixed))
# arm -> (nominal cost of the arm with the stock it runs with, reference cost used in loss_rel = design file's total_cost)
ARMS = {"FCFS design / FCFS": (F, F), "FCFS design / FCFS -> mu during": (F, F), "FCFS design / FCFS -> mu NP during": (F, F),
        "FCFS design / mu (sequential)": (p1("fcfs", "mu", True), F), "FCFS design / mu NP": (p1("fcfs", "mu", False), F),
        "mu design / mu": (MU, MU), "mu design / mu NP": (p1("mu", "mu", False), MU), "hmu design / hmu": (HMU, HMU),
        "full CT / FCFS": (FF, FF), "full CT / mu": (FFMU, FFMU),
        "mu-NP design / mu NP": (MUNP, MUNP), "hmu-NP design / hmu NP": (HMUNP, HMUNP), "mu-cov2 design / mu": (MUC2, MUC2), "FCFS-cov2 design / FCFS": (FC2, FC2)}
LABEL = {"FCFS design / FCFS": "Design for FCFS, FCFS", "FCFS design / FCFS -> mu during": "Design for FCFS, $\\mu$ rule during the disruption",
         "FCFS design / FCFS -> mu NP during": "Design for FCFS, non-preemptive $\\mu$ during the disruption", "FCFS design / mu (sequential)": "Design for FCFS, $\\mu$ rule",
         "FCFS design / mu NP": "Design for FCFS, non-preemptive $\\mu$ rule", "mu design / mu": "Design for $\\mu$, $\\mu$ rule", "mu design / mu NP": "Design for $\\mu$, non-preemptive $\\mu$ rule",
         "hmu design / hmu": "Design for $h\\times\\mu$, $h\\times\\mu$ rule", "full CT / FCFS": "Full cross-training, FCFS", "full CT / mu": "Full cross-training, $\\mu$ rule",
         "mu-NP design / mu NP": "Design for non-preemptive $\\mu$, non-preemptive $\\mu$ rule", "hmu-NP design / hmu NP": "Design for non-preemptive $h\\times\\mu$, non-preemptive $h\\times\\mu$ rule",
         "mu-cov2 design / mu": "Design for $\\mu$ with every SKU covered twice, $\\mu$ rule", "FCFS-cov2 design / FCFS": "Design for FCFS with every SKU covered twice, FCFS"}

if __name__ == "__main__":
    b4 = pd.read_csv(os.path.join(T, "B4_instances.csv")); b5 = pd.read_csv(os.path.join(T, "B5_instances.csv"))
    dis = {"surge": b4[(b4.gamma == 1.5) & (b4.duration == 10000)], "busiest absent": b5[(b5.absence == "busiest") & (b5.duration == 10000)],
           "random absent": b5[(b5.absence == "random") & (b5.duration == 10000)]}
    rows = []
    for arm, (nom, ref) in ARMS.items():
        for c in F:
            if c not in nom or not np.isfinite(nom[c]): continue
            r = {"arm": arm, "instance": c, "nominal": nom[c] / F[c]}
            for k, t in dis.items():
                s = t[(t.arm == arm) & (t.instance == c)]
                r[k] = (nom[c] + float(s.loss.iloc[0]) * ref[c]) / F[c] if len(s) else np.nan
                r[k + " recovered"] = float(s.recovered.iloc[0]) if len(s) else np.nan
            rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(T, "R_tradeoff.csv"), index=False)
    g = d.groupby("arm").agg(nominal=("nominal", "median"), surge=("surge", "median"), busiest=("busiest absent", "median"), random=("random absent", "median"),
                             surge_mean=("surge", "mean"), random_mean=("random absent", "mean"), n=("instance", "size")).reindex(list(ARMS))
    TABLE_ARMS = ["FCFS design / FCFS", "FCFS design / FCFS -> mu NP during", "FCFS design / FCFS -> mu during", "FCFS design / mu NP", "mu design / mu",
                  "mu-cov2 design / mu", "mu-NP design / mu NP", "FCFS-cov2 design / FCFS", "full CT / FCFS", "full CT / mu"]
    pd.set_option("display.width", 200); print(g.round(3).to_string())
    lines = [r"\begin{table}[!htb]", r"\centering",
             r"\caption{Cost of the designs, rules and operational responses in normal operation and during a disruption, relative to the design optimized for FCFS operated under FCFS (medians over the 128 instances). Nominal: total cost under the nominal failure rates. Disrupted: total cost rate during a disruption of $10{,}000$ expected failures, including the extra backorder cost of the episode (a surge of all failure rates by $50\%$, and the absence of the busiest and of a randomly chosen technician).}",
             r"\label{tab:tradeoff}", r"\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"& Nominal & \multicolumn{3}{c}{Disrupted} \\ \cmidrule(lr){3-5}", r"Design and operation & & Surge & Busiest absent & Random absent \\", r"\midrule"]
    for arm, r in g.reindex(TABLE_ARMS).iterrows():
        f = lambda v: "--" if not np.isfinite(v) else f"{v:.2f}"
        lines.append(f"{LABEL[arm]} & {f(r.nominal)} & {f(r.surge)} & {f(r.busiest)} & {f(r.random)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "revision", "tikz", "R_tradeoff_table.tex"), "w").write("\n".join(lines) + "\n")
