"""
fig_B4.py -- Figure for Section 6.2: (a) mean backorders over time on one instance for the FCFS design under FCFS and
the mu design under the mu rule (gamma 1.5, surge of 10,000 expected failures; mean of 35 replications, band = +-2 SEM);
(b) empirical distribution over the 128 instances of the recovery time (expected failures after the end of the surge)
for the same two arms and the three surge lengths. Output figs/B4_surge.pdf.
    python3 fig_B4.py [--case 128] [--gamma 1.5] [--duration 10000]
"""
import argparse, glob, gzip, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_B4 import ARMS, HERE, ROOT

SHOW = {"FCFS design / FCFS": ("Design for FCFS, FCFS", "C0", "-"), "mu design / mu": (r"Design for the $\mu$ rule, $\mu$ rule", "C3", "-")}


def record(folder, c, rule, gamma, D):
    return json.load(gzip.open(os.path.join(ROOT, folder, f"surge_case_{c:04d}_{rule}_g{gamma:g}_d{D}.json.gz"), "rt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, default=128); ap.add_argument("--gamma", type=float, default=1.5); ap.add_argument("--duration", type=int, default=10_000)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for (folder, rule), key in (("fcfs_designs", "FCFS"), "FCFS design / FCFS"), (("mu_designs", "mu"), "mu design / mu"):
        r = record(folder, a.case, rule, a.gamma, a.duration); mt = r["mean_trajectories"]
        t = np.asarray(mt["normalized_time"]) / 1000; y = np.asarray(mt["backorders_total"]); e = np.asarray(mt["backorders_total_sem"])
        lab, col, ls = SHOW[key]; ax[0].plot(t, y, color=col, ls=ls, lw=1.1, label=lab); ax[0].fill_between(t, y - 2 * e, y + 2 * e, color=col, alpha=0.2, lw=0)
    ax[0].axvspan(0, a.duration / 1000, color="0.85", lw=0, zorder=0); ax[0].set_xlabel("expected failures since the start of the surge (thousands)"); ax[0].set_ylabel("backorders (all SKUs)")
    ax[0].set_xlim(-10, 60); ax[0].legend(frameon=False, loc="upper right"); ax[0].set_title(f"(a) instance {a.case}, $\\gamma={a.gamma:g}$", fontsize=9)
    inst = pd.read_csv(os.path.join(HERE, "tables", "B4_instances.csv"))
    for D, ls in ((2500, ":"), (10000, "-"), (20000, "--")):
        for key, (lab, col, _) in SHOW.items():
            s = inst[(inst.arm == key) & (inst.gamma == a.gamma) & (inst.duration == D)].rec.sort_values().values / 1000
            if len(s): ax[1].step(np.r_[0, s], np.r_[0, np.arange(1, len(s) + 1) / len(s)], color=col, ls=ls, lw=1.1, where="post", label=f"{lab.split(',')[0]}, surge {D // 1000:d}k" if D == 10000 else None)
    ax[1].set_xlabel("recovery time after the surge (thousand expected failures)"); ax[1].set_ylabel("share of instances recovered"); ax[1].set_ylim(0, 1); ax[1].set_xlim(0, 60)
    ax[1].set_title(r"(b) surges of 2.5k ($\cdots$), 10k (---) and 20k (- -) failures, $\gamma=%g$" % a.gamma, fontsize=9); ax[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.join(HERE, "figs"), exist_ok=True); fig.savefig(os.path.join(HERE, "figs", "B4_surge.pdf")); print("figs/B4_surge.pdf")
