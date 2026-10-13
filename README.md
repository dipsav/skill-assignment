# Experimental results: Simulation-based Optimization of Joint Skill Assignment and Inventory Planning for Resilient Maintenance Systems

Results of the numerical experiments of manuscript CAIE-D-26-05208 (revision R2.2, October 2026). The code that
produced them is in the companion code repository (`python_simulator`, `fast_simulator`, `experiments`); every file
here can be regenerated with the scripts named below, with the seeds recorded in the files.

| Folder | Content | Paper |
|---|---|---|
| `instances/largeCases/` | the 128 instances of the test bed (Table 2): failure and repair rates, holding, backorder, technician and skill costs, one JSON per instance (`fullRangeResultsPSO_new_2019_case_NNNN.json`); the 2019 design in the same file is superseded by `designs/` | Section 5 |
| `instances/sets23_instances.csv`, `sets23_exact.json` | the eight test-bed-size instances of Appendix B.2 and their exact per-SKU queue-length distributions | Appendix B.2 |
| `designs/PSO_results_tuned_polished/` | **final designs optimized for FCFS** (modified BPSO, tuned parameters, followed by the local search; cost re-estimated with independent seeds) | 5.2, 5.3, 5.5, 5.6, 6 |
| `designs/PSO_results_tuned_rule_mu_polished/`, `.../rule_hmu_polished/` | final designs optimized for the preemptive mu and h x mu rules | 5.4-5.6, 6 |
| `designs/PSO_results_tuned_rule_mu_np_polished/`, `.../rule_hmu_np_polished/` | designs optimized for the non-preemptive rules | 5.5, 6.4 |
| `designs/PSO_results_tuned_rule_mu_cov2_polished/`, `designs/PSO_results_tuned_cov2_polished/` | designs with every SKU covered by at least two technicians (mu rule; FCFS) | 6.1, 6.4 |
| `designs/PSO_results_fullCT/`, `.../fullCT_rule_mu/` | fully cross-trained references, stock optimized, under FCFS and under the mu rule | 5.2, 5.4, 6 |
| `designs/PSO_results_tuned*` (without `_polished`) | the swarm's designs before the local search | 5.1, Appendix A |
| `designs/PSO_results_rerun*`, `GA_results_rerun/`, `VNS_results_rerun/` | BPSO with the published (proof-of-concept) parameters, the GA and the VNS on all 128 instances | 5.1, Appendix A |
| `benchmark/C1_results/{pso,pso_orig,ga,vns,hybrid}/` | benchmark at an equal budget of 2,500 simulated designs: 32 instances x 5 seeds x 5 methods, final designs, independent cost estimates and best-so-far traces | Table 2, Fig. 3 |
| `benchmark/T2_results/` | tuning confirmation on eight instances (10 BPSO variants x 2 seeds) | Table A.4 |
| `tuning/` | T1 tuning runs on the brute-force dataset (`t1_results_*.csv.gz`; 72 + 54 settings, two seeds, extended grid), the lookup tables (`tables.pkl`, built from Mendeley Data doi 10.17632/48fctynd77.4), settings and logs | Table A.3 |
| `summaries/` | one CSV per experiment (R1 rules on the FCFS design; joint comparisons; P1 preemption; E3 distributions; B1, B23 permanent surges; B4, B5 transient surges and absences; R0 structural indicators; R_tradeoff; C1; prune check; cost precision; wall-clock) and the LaTeX tables generated from them | all tables |
| `figures/tikz/`, `figures/pdf/` | TikZ sources and compiled figures | Figs. 3-9, B.1 |
| `logs/` | run logs with per-instance statistics | provenance |
| `manifest.csv` | path, size and SHA-256 of every file | |

Design files: `best_assignment` (skill matrix, technicians x SKUs), `stocks` (base-stock levels), `total_cost` and the
cost components (independent re-estimate), `*_ff` fields for the fully cross-trained reference, `rule`, `priorities`,
`preemptive`, `min_coverage`, `pso_params`, `seed`, `designs_simulated`, `evaluations_requested`, `log` (search
statistics), `unpolished_assignment` / `unpolished_total_cost` (before the local search), `utilizationRatesAverage`.

Large raw records are deposited separately (data repository, DOI: [[to be added]]), not in this repository:
`replications/` (per-replication costs of B1, B23, E3, P1 and the per-replication transient summaries
`B4_transient.csv.gz`, `B5_transient.csv.gz`, 225 MB) and `surge_results_B4_<set>.tar`, `surge_results_B5_<set>.tar`
(backorder trajectories of every surge and absence replication, one `json.gz` per case, rule, multiplier and length,
870 MB). Their `manifest.csv` (path, size, SHA-256) is included in the deposit.

Seeds: search 1,000 + instance; independent re-estimation 1,000,000 + 20 i (i = replication); transient surges
2,000,000 + 1,000 instance + 20 i; permanent surges 3,000,000 + 1,000 instance + 20 i; distributions 4,000,000 +
1,000 instance + 20 i; benchmark seeds 1-5. Simulator: `simulation_codes` 2.0 (Numba kernels; C port gives identical
output), 35 replications of 100,000 failures, no warm-up, empty start.
