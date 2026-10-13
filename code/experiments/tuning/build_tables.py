"""
build_tables.py -- cost tables for the tuning study T1, from the brute-force dataset
(simulated_sets_version4.json: 840 small instances, N, M in 2..6; for each, the simulated queue-length distributions of
every capacity-feasible skill matrix, one per permutation of the technicians; 35 x 100,000 failures, FCFS).

Cost settings, as in ICACI 2017 and the group's tuning code (VNS_BenchMarking.tunningExperimentsBruteForceDataset):
h_max = 1,000, h_min in {1, 100}, technician cost f in {10,000, 100,000}, skill cost c = 0.1 f, penalty b = 50 x the
failure-weighted mean holding cost, holding costs inversely related to workload (generateHoldingCosts, variant 2),
drawn with seed 1000 x instance + setting. 840 x 4 = 3,360 tuning instances.

For each instance and design: EHB = optimal holding + backorder cost (SimulationInterface_fast.OptimizeStockLevels...,
infinite if a distribution's mass is short), and the total with technicians and skills. Designs are keyed by their
canonical form (rows = technicians sorted), so any matrix the optimizers propose is looked up up to technician order;
a matrix not in the table is infeasible. Output: tuning/tables.pkl.
"""
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulation_codes import SimulationInterface_fast as SIf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "Simulation Results for Multi-Class Multi-Server Queueing Systems with Cross-Training", "simulated_sets_version4.json")
SETTINGS = [(hmin, f) for hmin in (1, 100) for f in (10_000, 100_000)]
H_MAX, PENALTY_FACTOR, SKILL_FACTOR = 1000, 50, 0.1


def holding_costs(rho, hmin, hmax, rng):
    """generateHoldingCosts(variant=2) of the group's code, with a seeded generator."""
    rho_range = max(rho) - min(rho); h_range = hmax - hmin
    v = (hmax - hmin) / 20; xi = rng.uniform(-v, v, len(rho))
    h = (h_range + 10) / ((rho - min(rho)) / (rho_range if rho_range > 0 else 1) * 9 + 1) - 10 + hmin + xi
    return np.clip(h, hmin, hmax)


def canon(X):
    return np.array(sorted(map(tuple, np.asarray(X, np.int8))), np.int8).tobytes()


if __name__ == "__main__":
    tables = []
    with open(DATA) as fh:
        fh.readline()
        for idx, line in enumerate(fh, 1):
            d = json.loads(line)
            lam, mu = np.array(d["arrival_rates"], float), np.array(d["service_rates"], float)
            N, M = d["n_classes"], d["m_servers"]
            designs = []
            for a in d["assignments"].values():
                X = np.array(a["class_server_assignment"], int)
                q = a["queue_length_distributions"]
                L = max(max(int(n) for n in q[k]) for k in q) + 1
                dist = np.zeros((N, L))
                for k in q:
                    for n, p in q[k].items():
                        dist[int(k), int(n)] = p
                designs.append((X, dist))
            for s, (hmin, f) in enumerate(SETTINGS):
                rng = np.random.default_rng(1000 * idx + s)
                h = holding_costs(lam / mu, hmin, H_MAX, rng)
                b = PENALTY_FACTOR * np.sum(lam * h) / np.sum(lam); c = SKILL_FACTOR * f
                ehb = {}
                for X, dist in designs:
                    e = SIf.OptimizeStockLevelsAndCostsSimBased(h, b, dist)[0]
                    ehb[canon(X)] = float(e)
                totals = {k: e + f * (np.frombuffer(k, np.int8).reshape(M, N).sum(1) > 0).sum() + c * np.frombuffer(k, np.int8).sum()
                          for k, e in ehb.items()}
                opt = min(totals.values())
                tables.append({"instance": idx, "case": d["caseID"], "setting": s, "N": N, "M": M, "lam": lam, "mu": mu, "h": h,
                               "b": b, "f": f, "c": c, "hmin": hmin, "ehb": ehb, "optimum": opt, "n_designs": len(ehb),
                               "n_finite": int(np.isfinite(list(totals.values())).sum())})
            if idx % 100 == 0:
                print(idx, "instances", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.pkl")
    pickle.dump(tables, open(out, "wb"))
    sizes = np.array([t["n_designs"] for t in tables[::4]])
    print(f"{len(tables)} tuning instances -> {out}; designs per instance: median {np.median(sizes):.0f}, max {sizes.max()}, "
          f">= 100: {(sizes >= 100).sum()} of {len(sizes)}")
