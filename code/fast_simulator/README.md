# fast_simulator -- the compiled simulator (package `simulation_codes`)

| File | What it is |
|---|---|
| `simulation_codes/simulation_fast.py` | Numba kernels: `_run` (exponential model: FCFS, preemptive and non-preemptive priority rules; dispatch: longest-idle technician, lowest-priority repair interrupted, interrupted part re-dispatched, FCFS within a level) and `_run_general` (per-SKU renewal failure streams and Erlang / lognormal / hyperexponential / deterministic repair times, preemptive resume with the stored remaining time). Public entry points `simulate(...)`, `simulation_batch_run(...)`, `checkSkillAssignment(...)` (LP feasibility check, load <= 1 - eps). The C port is used automatically when `csim/libsimfast.so` exists (`SIM_BACKEND=numba` forces Numba); both give identical output. |
| `simulation_codes/SimulationInterface_fast.py` | Evaluation of a skill matrix: simulation, newsvendor base-stock levels, cost, cache of simulated designs (`filedict.py`), evaluation budget (`SIf.budget`), rule (`SIf.priorities`), discipline (`SIf.preemptive`), coverage requirement (`SIf.min_coverage`), independent re-estimation seeds. |
| `csim/simfast.c`, `csim/sim_c.py`, `csim/build.sh` | Statement-for-statement C port of `_run` with Numba's Mersenne Twister reproduced, about 3x faster; `bash csim/build.sh` builds it (gcc). |
| `fcfs_exact.py`, `exact_priority2.py`, `exact3.py`, `exact_nonpreemptive2.py` | Exact references: multi-class M/M/c under FCFS (van Harten and Sleptchenko, 2003), two-group preemptive priorities (Sleptchenko et al., 2005), three preemptive levels, two-group non-preemptive priorities (state: high-priority parts, low-priority parts, low-priority parts in repair). |
| `test_simulation_fast.py` | 15 checks of the exponential kernels against the exact references and the dispatch logic (Appendix B.1); also against `simulation_pso_paper_v2.py` when that file is importable. |
| `test_distributions.py` | 28 checks of the general kernel: M/G/1 (Pollaczek-Khinchine), GI/M/1, M/G/1 with preemptive-resume and with non-preemptive priorities (Cobham), multi-server non-preemptive (Appendix B.6). |

Install (editable, so that the C library next to the package is found):

    bash csim/build.sh            # optional
    pip install -e .              # provides `import simulation_codes`
    python test_simulation_fast.py

Needs Python >= 3.10, numpy >= 2, scipy >= 1.14, numba >= 0.60, xxhash; gcc for the C port.
