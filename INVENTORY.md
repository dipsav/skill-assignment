# Inventory of the numerical results (13 October 2026)

All results were produced with the compiled simulator (`fast_simulator/simulation_codes`, Numba kernels with the C
port, output identical) between 4 and 11 October 2026 on the two Xeon E5-2650 v4 workstations (M5000, P40). Every
run uses fixed seeds: search seeds 1,000 + k, independent re-estimation seeds 1,000,000 + 20 i, transient surges
2,000,000 + 1,000 case + 20 i, permanent surges 3,000,000 + ..., distributions 4,000,000 + ... Costs reported in the
paper are always the independent re-estimates.

## 1. The Mendeley dataset ("small cases") is unchanged

`Simulation Results for Multi-Class Multi-Server Queueing Systems with Cross-Training`, Mendeley Data v4
(doi 10.17632/48fctynd77.4; local copy `simulated_sets_version4.json`, 512 MB, 840 cases, N, M <= 6, all feasible
assignments, 35 runs x ~100,000 arrivals, FCFS only, generated 2016).

- It was **not regenerated** in this revision. The tuning of Appendix A (T1) uses its simulated distributions as a
  lookup table (`fast_simulator/tuning/build_tables.py` -> `tables.pkl`); no small case was re-simulated.
- It contains **no priority-rule results**, so the faults of the old priority simulator do not touch it.
- Its FCFS values were checked against the new simulator (4 October): the published distributions were produced by
  the FCFS path of the newer code with a 1,000-request warm-up and the same replication seeds in every case; the
  +0.3% offset this causes is within the sampling error (independent seeds: -0.09% +- 0.12%). No error found.
- Usage notes recorded for the paper, not errors in the data: instances with fewer than 100 feasible assignments are
  left out of the tuning (a swarm enumerates them); dataset case 0254 (6 x 2) has no design with a finite cost (all
  its distributions are short of mass at the truncation) and its four tuning instances are dropped.

**Mendeley Data does not need an update for the paper's small cases.** What is new (the 128-instance designs, the
benchmark runs, the surge records) is a different dataset; it goes to the results repository below (and can be
deposited as a new Mendeley Data entry or a Zenodo record with a DOI when the paper is accepted).

The "sets 2-3" of Appendix B.2 are not part of the Mendeley dataset: they are the eight test-bed-size, fully
cross-trained instances with two repair-rate groups generated here (`sets23_instances.csv`, exact per-SKU
distributions in `sets23_exact.json`) and simulated with the new simulator.

## 2. What exists, where, and which part of the paper uses it

| Folder (working tree) | Content | Size / files | Used in |
|---|---|---|---|
| `largeCases/` | the 128 test-bed instances of Table 2 (one JSON each, as published in 2019, incl. the 2019 designs) | 4.6 MB / 128 | everything |
| `PSO_results_tuned_polished/` | **final FCFS designs** (tuned BPSO + local search, independent re-estimate) | 5 MB / 128 | 5.2, 5.3, 5.5, 5.6, 6.1-6.4 |
| `PSO_results_tuned_rule_mu_polished/`, `_hmu_polished/` | **final mu and h x mu designs** (joint optimization) | 5 MB each | 5.4, 5.5, 5.6, 6 |
| `PSO_results_tuned_rule_mu_np_polished/`, `_hmu_np_polished/` | designs optimized for the non-preemptive rules (A3) | 5 MB each | 5.5, 6.4 |
| `PSO_results_tuned_rule_mu_cov2_polished/`, `PSO_results_tuned_cov2_polished/` | coverage-two designs (B6b) | 5 MB each | 6.1, 6.4 |
| `PSO_results_fullCT/`, `PSO_results_fullCT_rule_mu/` | fully cross-trained references (stock optimized) | 0.8 MB each | 5.2, 5.4, 6 |
| `PSO_results_tuned*` (unpolished) | the swarm's designs before the local search (7 sets) | 5 MB each | 5.1 (polish gain), Appendix A |
| `PSO_results_rerun/`, `_polished/`, `_rule_mu/` | BPSO with the published (ICACI 2017) parameters, superseded | 5 MB each | Appendix A (comparison) |
| `GA_results_rerun/`, `VNS_results_rerun/` | GA and VNS on all 128 instances (2,400 requested / 2,500 distinct designs) | 7.6 / 1.4 MB | 5.1 text |
| `C1_results/{pso,pso_orig,ga,vns,hybrid}` | benchmark at an equal budget: 32 instances x 5 seeds x 5 methods, with convergence traces | 3.2 MB / 800 | 5.1 (Table 2, Fig. 3) |
| `T2_results/` | tuning confirmation on 8 instances x 2 seeds x 10 variants | 0.7 MB / 160 | Appendix A (Table A.4) |
| `fast_simulator/tuning/t1_results_{seed1,seed2,ext_seed1}.csv` | T1 tuning runs on the dataset (139k + 139k + 238k runs); `tables.pkl` (lookup tables built from the Mendeley dataset) | 54 MB + 12 MB | Appendix A (Table A.3) |
| `fast_simulator/tables/*.csv` (summaries) | one CSV per experiment: R1 (rules on the FCFS design), joint comparisons, P1 (preemption), E3 (distributions), B1/B23 (permanent surges), B4/B5 (transient), R0 (margins), R_tradeoff, C1, prune check, cost precision, wall-clock | 93 files, 4 MB without the two below | all tables |
| `fast_simulator/tables/*_replications.csv` | per-replication costs of B1, B23, E3, P1 (one row per replication) | 20 files, ~280 MB (gz 120 MB) | confidence intervals |
| `fast_simulator/tables/B4_transient.csv`, `B5_transient.csv` | per-instance, per-replication peak / loss / recovery of every surge and absence record | 208 + 99 MB (gz 71 + 32 MB) | 6.2, 6.3 |
| `surge_results_B4/` (9 design sets), `surge_results_B5/` | raw transient records: backorder trajectories per replication, one json.gz per case / rule / gamma / length (and per absence) | 596 + 271 MB / 24,960 files | 6.2, 6.3 (Figs. 7, 9) |
| `revision/tikz/*.tex`, `fast_simulator/figs/*.pdf` | LaTeX tables and TikZ figures generated from the CSVs | 31 + 10 files | the manuscript |
| `fast_simulator/*.log`, surge logs | run logs with the per-case statistics | 0.5 MB / 51 | provenance |
| `sets23_instances.csv`, `sets23_exact.json` | Appendix B.2 instances and exact references | 0.14 MB | B.2 |

Not kept: `PSO_results_rerun_discarded_20261004_195400/` (caches of the crashed first run, 354 MB),
`fast_simulator_old_20261004_195400/` (code backup), `fast_simulator.zip` and other transfer archives.

## 3. Results repository (GitHub)

`results_repo/` is a git repository built by `make_results_repo.sh` from the folders above:

| Repository folder | From | Size |
|---|---|---|
| `instances/` | `largeCases/`, `sets23_*` | 5 MB |
| `designs/<set>/` | the 21 design folders (128 JSON each) | ~100 MB uncompressed (git compresses JSON well) |
| `benchmark/C1_results`, `benchmark/T2_results` | as is | 4 MB |
| `tuning/` | T1 CSVs (gzipped), `tables.pkl`, settings, logs | 27 MB |
| `summaries/` | the per-experiment CSVs and `.tex` tables | 4 MB |
| `figures/tikz`, `figures/pdf` | TikZ sources and compiled figures | 2 MB |
| `logs/` | run logs | 0.5 MB |
| `manifest.csv` | path, bytes, sha256 of every file | |

`results_release_assets/` (not in git; to be attached to a GitHub release, or deposited on Zenodo / Mendeley Data):
`replications/*.csv.gz` (B1, B23, E3, P1 replication files, B4 and B5 transient summaries; ~225 MB) and
`surge_results_B4_<set>.tar`, `surge_results_B5_<set>.tar` (the raw records, ~870 MB in 21 tar files; GitHub release
assets may be up to 2 GB each).
