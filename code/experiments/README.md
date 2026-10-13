# experiments -- the code of every simulation experiment of the paper

All scripts import the compiled simulator as `simulation_codes` (see `../setup.sh`) and are run from this folder.
Design folders (`../PSO_results_*`), surge records (`../surge_results_B4`, `../surge_results_B5`) and benchmark runs
(`../C1_results`, `../T2_results`) are read and written one level up; summary tables go to `tables/`, LaTeX tables and
TikZ figures to `../revision/tikz/`. The test bed (128 instances) is `../largeCases/`. Every run uses fixed seeds; the
cost of every final design is re-estimated with seeds not used in the search (`--startSeed` family, see the drivers).
Shell scripts `run_*.sh` are the exact queues that produced the results (set `PY` if `python3` is not the right interpreter).

## Optimizers (Section 4)
| File | |
|---|---|
| `PSO_optimization_v2.py` | The modified BPSO (increasing inertia, multiplicative update of contested bits, Eq. 12), with `variant="original"` for the BPSO of El-Maleh et al.; budget-driven inertia schedule under an evaluation budget; parallel particle evaluation (`PARALLEL_WORKERS`). |
| `GA_Optimization_v2.py` | The GA of Sleptchenko et al. (2019) on the corrected evaluation. |
| `VNS_optimization_v2.py` | Port of the VNS of Abdelwanis et al. (2024) with the budget counted in distinct simulated designs; its local search (remove / add / remove-technician moves) is the local search of Section 4.3. |
| `hybrid_optimization.py` | BPSO on 70% of the budget followed by the local search (the proposed method in the benchmark). |
| `polish_designs.py` | Local search to convergence on the final designs of a folder (the "polish" of every reported design set). |
| `rerun_pso_cases.py` | Driver for the 128 instances: `--algorithm pso|ga|vns`, `--rule fcfs|mu|hmu`, `--nonpreemptive`, `--min-coverage 2`, `--published`, `--workers`; writes one JSON per instance with the design, the independent cost estimate and the search statistics. `run_all.sh`, `run_all_ga.sh`, `run_polish_tuned.sh`, `run_A3_p40.sh` (non-preemptive designs), `run_B6b_p40.sh` (coverage-two designs). |
| `make_fullct_designs.py` | Fully cross-trained reference designs (stock optimized) under FCFS and under the mu rule. |

## Section 5
| Experiment | Scripts |
|---|---|
| 5.1 benchmark at an equal budget (C1), Fig. 3, Table 2 | `c1_benchmark.py` (five methods, 32 instances x 5 seeds, budget 2,500), `tikz_c1.py`, `wallclock_check.py` (parallel BPSO vs sequential VNS) |
| Appendix A tuning (T1 on the brute-force dataset, T2 on eight test-bed instances) | `tuning/build_tables.py` (lookup tables from the dataset of Sleptchenko et al., 2017), `tuning/t1_tuning.py`, `tuning/select_t2.py`, `tuning/appendix_tables.py` |
| 5.2 savings vs full cross-training, Fig. 4 | `rerun_analysis.py`, `paper_figures.py`, `tikz_figures.py`, `cost_precision.py` (half-widths), `prune_check.py` (redundant skills) |
| 5.3 rules on the FCFS design, Fig. 5 | `rule_comparison.py` (stock re-optimized and fixed; utilizations) |
| 5.4 designs optimized for each rule | `rerun_pso_cases.py --rule mu|hmu` + `polish_designs.py`; comparison in `paper_figures.py` and `analysis_GA_vs_PSO.ipynb` |
| 5.5 preemptive vs non-preemptive (P1), Table 4 | `preemption_comparison.py`, `table_P1.py`, `run_P1.sh` |
| 5.6 failure and repair-time distributions (E3), Table 5 | `distribution_sensitivity.py`, `analyze_E3.py`, `run_E3.sh` |

## Section 6
| Experiment | Scripts |
|---|---|
| 6.1 one-SKU surges (B1), Table 6, Fig. 8; several SKUs (B2/B3), Table C.2 | `sku_surges.py`, `multi_surges.py`, `analyze_B1B23.py`, `table_B1.py`, `table_B23.py`, `tikz_B1.py`, `run_B1.sh`, `run_B23.sh` |
| 6.2 temporary surge of all SKUs (B4), Table 7, Fig. 7, Table C.3 | `transient_surge/run_surge_cases.py` (phases, gamma, lengths, rule switch `--switch-to`, `--nonpreemptive`), `transient_surge/run_B4.sh`, `analyze_B4.py`, `table_B4.py`, `table_B45_responses.py`, `tikz_B4.py`, `fig_B4.py` |
| 6.3 absence of a technician (B5), Table 8, Fig. 9, Table C.4 | `transient_surge/run_surge_cases.py --absence busiest|random`, `transient_surge/run_B5.sh`, `analyze_B5.py`, `table_B5.py`, `tikz_B5.py`, `absence_skills.py` (single-covered skills) |
| 6.4 resilience and cost (R0-R5), Table 9 | `resilience_margin.py` (stability margin, N-1 share), `resilience_tradeoff.py` (cost rates during disruptions), `run_R1.sh` (full cross-training), `run_R24.sh` (rule switched during the disruption; non-preemptive arms), `run_R5.sh` (non-preemptive and coverage-two designs), `transient_surge/run_R1b.sh` |

## Appendix B (validation)
| | |
|---|---|
| B.1, B.6 | `../fast_simulator/test_simulation_fast.py`, `../fast_simulator/test_distributions.py` |
| B.2 test-bed-size instances, Fig. B.1 | `validation_sets23.py` (exact references in `../sets23_exact.json`, instances in `../sets23_instances.csv`) |
| B.3 agreement with the 2019 simulator | `compare_with_older.py` |
| B.5 transient simulation | `transient_surge/test_surge_fast.py` (draw-for-draw equality with the steady-state simulator, exact M(t)/M/inf transient, backorder integration, absence) |

`transient_surge/surge_tools/surge_fast.py` is the compiled transient engine (Numba) with the dispatching rules of
Section 4.4, phases, backorders with fixed stock, technician absence, non-preemptive discipline and a rule switch on a
time window; `surge_simulation.py` is the Python engine it builds on (also in `../python_simulator`).
Tests of the optimizers: `test_vns_v2.py`, `test_hybrid.py`.
