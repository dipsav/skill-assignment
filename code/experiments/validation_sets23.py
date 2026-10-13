"""validation_sets23.py -- Appendix B.2: the eight test-bed-size, fully cross-trained instances (sets 2-3) under FCFS and
under the mu rule (fast SKUs preemptive priority), simulated with simulation_fast (35 x 100,000 failures, no warm-up,
seeds 20260930 + 20 i) and compared with the exact per-SKU distributions of sets23_exact.json (FCFS: multi-class M/M/c of
van Harten & Sleptchenko 2003; mu: two-group preemptive chain = Sleptchenko et al. 2005 moments, binomial thinning).
Per SKU and rule: relative error of the mean, whether the exact mean lies in the 95% interval, Kolmogorov distance of
the distributions, and the base-stock levels at fractiles 0.90/0.95/0.99/0.999 from the simulated vs the exact pmf.
Writes tables/B2_sets23.csv and revision/tikz/figB1_sets23.tex (box plots of the relative errors)."""
import json, os, sys, subprocess
import numpy as np, pandas as pd
from scipy.stats import t as student_t
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); ROOT = os.path.join(HERE, "..")
from simulation_codes import simulation_fast as sf
from tikz_figures import PREAMBLE, OUT
REPS, NREQ, SEED0, MAXQ = 35, 100_000, 20260930, 2000
FRACTILES = (0.90, 0.95, 0.99, 0.999)
base_stock = lambda p, f: int(np.searchsorted(np.cumsum(p), f - 1e-12))

exact = json.load(open(os.path.join(ROOT, "sets23_exact.json"))); rows = []
for s in exact:
    lam = np.array([u["lambda"] for u in s["skus"]]); mu = np.array([u["mu"] for u in s["skus"]]); M = s["M"]; N = len(lam)
    fast = np.array([u["group"] == "fast" for u in s["skus"]]); R = np.ones((M, N)) * mu[None, :]
    for rule in ("FCFS", "mu"):
        score = None if rule == "FCFS" else np.where(fast, 1.0, 2.0)[None, :].repeat(M, axis=0)
        means, acc = [], None
        for i in range(REPS):
            o = sf.simulate(lam, R, score, NREQ, 0, SEED0 + 20 * i, MAXQ); d = o["dist"] / o["dist"].sum(axis=1, keepdims=True)
            means.append((d * np.arange(MAXQ)).sum(axis=1)); acc = d if acc is None else acc + d
        means = np.array(means); pm = acc / REPS
        for k, u in enumerate(s["skus"]):
            pe = np.array(u[f"exact_{rule}"]["pmf"]); ex = u[f"exact_{rule}"]["mean"]; m = means[:, k].mean()
            half = student_t.ppf(0.975, REPS - 1) * means[:, k].std(ddof=1) / np.sqrt(REPS)
            L = max(len(pe), MAXQ); pe2 = np.zeros(L); pe2[:len(pe)] = pe; ps2 = np.zeros(L); ps2[:MAXQ] = pm[k]
            row = {"instance": s["instance"], "N": s["N"], "M": M, "rho": s["rho"], "rule": rule, "sku": k, "group": u["group"], "exact_mean": ex, "sim_mean": m, "halfwidth": half,
                   "rel_error": m / ex - 1, "in_interval": abs(m - ex) <= half, "kolmogorov": float(np.abs(np.cumsum(ps2) - np.cumsum(pe2)).max())}
            for f in FRACTILES: row[f"S_exact_{f}"] = base_stock(pe2, f); row[f"S_sim_{f}"] = base_stock(ps2, f)
            rows.append(row)
    print("instance", s["instance"], "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(HERE, "tables", "B2_sets23.csv"), index=False)
for rule, g in d.groupby("rule"):
    same = {f: float((g[f"S_exact_{f}"] == g[f"S_sim_{f}"]).mean()) for f in FRACTILES}; off1 = {f: float(((g[f"S_exact_{f}"] - g[f"S_sim_{f}"]).abs() <= 1).mean()) for f in FRACTILES}
    print(f"{rule}: {len(g)} SKUs; in 95% interval {g.in_interval.mean():.1%}; |rel error| mean {g.rel_error.abs().mean():.2%} max {g.rel_error.abs().max():.2%}; Kolmogorov mean {g.kolmogorov.mean():.4f} max {g.kolmogorov.max():.4f}; base stock equal {same}; within 1 unit {off1}")
print(d.groupby(["rule", "group"]).agg(rel_abs_mean=("rel_error", lambda x: x.abs().mean()), rel_abs_max=("rel_error", lambda x: x.abs().max()), cover=("in_interval", "mean")).round(4).to_string())
# Figure B.1: box plots of the relative errors (%) per rule and group
s_ = ["\\begin{tikzpicture}", "\t\\begin{axis}[width=0.6\\textwidth, height=5.5cm, boxplot/draw direction=y, ylabel={Relative error of the SKU mean (\\%)}, xtick={1,2,3,4}, xticklabels={FCFS\\\\fast, FCFS\\\\slow, $\\mu$ rule\\\\fast, $\\mu$ rule\\\\slow}, xticklabel style={align=center, font=\\footnotesize}, ymin=-6, ymax=6, tick label style={/pgf/number format/fixed}, boxplot/every box/.style={fill, fill opacity=0.5}]"]
for i, (rule, grp, col) in enumerate((("FCFS", "fast", "red"), ("FCFS", "slow", "red"), ("mu", "fast", "blue"), ("mu", "slow", "blue")), start=1):
    v = 100 * d[(d.rule == rule) & (d.group == grp)].rel_error.values
    s_.append(f"\t\t\\addplot[boxplot, color={col}, mark=*, mark size=1pt] table[y index=0] {{\n" + "\n".join(f"{x:.3f}" for x in v) + "\n};")
s_ += ["\t\\end{axis}", "\\end{tikzpicture}"]
name = "figB1_sets23"; open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(s_) + "\n")
open(os.path.join(OUT, name + "_standalone.tex"), "w").write(PREAMBLE + f"\t\\input{{{name}.tex}}\n\\end{{document}}\n")
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + "_standalone.tex"], cwd=OUT, capture_output=True, text=True); print(name, "compiled" if r.returncode == 0 else "FAILED")
for ext in (".aux", ".log"):
    p = os.path.join(OUT, name + "_standalone" + ext); os.path.exists(p) and os.remove(p)
