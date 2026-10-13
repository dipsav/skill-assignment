"""
resilience_margin.py -- R0: structural resilience indicators of a skill matrix (no simulation; one LP each).
  gamma*      stability margin: the largest uniform multiplier of all failure rates the skill matrix can carry,
              1 / rho*, rho* = the smallest achievable maximum technician load (LP of checkSkillAssignment, no cap);
  n1_share    share of the employed technicians whose absence leaves the shop stable (N-1 criterion, LP without
              that technician's row; an SKU nobody else can repair counts as unstable);
  n1_gamma    the smallest stability margin over the single-technician absences (0 if some absence is unstable);
  single      share of SKUs that only one technician can repair.
Design sets: final FCFS / mu / h x mu designs, full cross-training, the five C1 methods (and A3 when available).
Writes tables/R0_margin.csv and prints the summary.
"""
import glob, json, os, re
import numpy as np, pandas as pd
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")


def rho_star(X, lam, mu):
    X = np.asarray(X); M, N = X.shape
    if np.any(X.sum(axis=0) == 0): return np.inf
    pairs = np.argwhere(X.T == 1); k = len(pairs); load = lam / mu
    c = np.zeros(k + 1); c[-1] = 1.0
    A_eq = np.zeros((N, k + 1)); A_eq[pairs[:, 0], np.arange(k)] = 1.0
    A_ub = np.zeros((M, k + 1)); A_ub[pairs[:, 1], np.arange(k)] = load[pairs[:, 0]]; A_ub[:, -1] = -1.0
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(M), A_eq=A_eq, b_eq=np.ones(N), bounds=[(0, 1)] * k + [(0, None)], method="highs")
    return r.x[-1] if r.status == 0 else np.inf


def indicators(X, lam, mu):
    X = np.asarray(X, int); emp = np.flatnonzero(X.sum(axis=1) > 0); g = 1 / rho_star(X, lam, mu)
    n1 = []
    for j in emp:
        Y = np.delete(X, j, axis=0); r = rho_star(Y, lam, mu); n1.append(1 / r if np.isfinite(r) else 0.0)
    n1 = np.array(n1)
    return {"gamma_star": g, "n1_share": float((n1 > 1).mean()), "n1_gamma": float(n1.min()), "single": float((X.sum(axis=0) == 1).mean()),
            "technicians": int(len(emp)), "skills": int(X.sum())}


def case_number(p): return int(re.search(r"case_(\d+)", os.path.basename(p)).group(1))


if __name__ == "__main__":
    rows = []
    sets = {"FCFS design": "PSO_results_tuned_polished", "mu design": "PSO_results_tuned_rule_mu_polished", "hmu design": "PSO_results_tuned_rule_hmu_polished",
            "mu design NP": "PSO_results_tuned_rule_mu_np_polished", "hmu design NP": "PSO_results_tuned_rule_hmu_np_polished",
            "mu design cov2": "PSO_results_tuned_rule_mu_cov2_polished", "FCFS design cov2": "PSO_results_tuned_cov2_polished"}
    for label, folder in sets.items():
        for p in sorted(glob.glob(os.path.join(ROOT, folder, "*case_*.json"))):
            d = json.load(open(p)); lam, mu = np.asarray(d["failure_rates"], float), np.asarray(d["service_rates"], float)
            rows.append({"set": label, "instance": case_number(p), "rho": d["utilization_rate"], "N": d["m_skills"], "M": d["n_servers"], "c": d["skill_cost_factor"], "f": d["machine_cost_factor"],
                         "total": d["total_cost"], **indicators(d["best_assignment"], lam, mu)})
            if label == "FCFS design":
                rows.append({"set": "full cross-training", "instance": case_number(p), "rho": d["utilization_rate"], "N": d["m_skills"], "M": d["n_servers"], "c": d["skill_cost_factor"],
                             "f": d["machine_cost_factor"], "total": d["total_cost_ff"], **indicators(np.ones((d["n_servers"], d["m_skills"])), lam, mu)})
    base = {case_number(p): p for p in glob.glob(os.path.join(ROOT, "PSO_results_tuned_polished", "*case_*.json"))}
    for f in glob.glob(os.path.join(ROOT, "C1_results", "*", "*.json")):
        r = json.load(open(f)); d = json.load(open(base[r["case"]])); lam, mu = np.asarray(d["failure_rates"], float), np.asarray(d["service_rates"], float)
        rows.append({"set": f"C1 {r['algorithm']}", "instance": r["case"], "seed": r["seed"], "rho": d["utilization_rate"], "N": d["m_skills"], "M": d["n_servers"], "c": d["skill_cost_factor"],
                     "f": d["machine_cost_factor"], "total": r["total_cost"], **indicators(r["best_assignment"], lam, mu)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(HERE, "tables", "R0_margin.csv"), index=False); pd.set_option("display.width", 200)
    print(t.groupby("set").agg(n=("instance", "size"), gamma_med=("gamma_star", "median"), gamma_mean=("gamma_star", "mean"), n1_share=("n1_share", "mean"),
                              n1_all=("n1_share", lambda s: (s == 1).mean()), single=("single", "mean"), techs=("technicians", "mean"), skills=("skills", "mean")).round(3).to_string())
    print(t[t.set.isin(["FCFS design", "mu design", "full cross-training"])].groupby(["set", "rho"]).agg(gamma_med=("gamma_star", "median"), n1_share=("n1_share", "mean"), single=("single", "mean")).round(3).to_string())
