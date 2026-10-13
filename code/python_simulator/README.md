# python_simulator -- the pure-Python simulators (reference implementations)

Plain Python (NumPy only, no compilation). They are the readable reference for the rules implemented in the compiled
simulator (`../fast_simulator`), which reproduces them and is the engine used for the numbers in the paper.

| File | What it is |
|---|---|
| `simulation_pso_paper_v2.py` | The fixed priority simulator (FCFS, preemptive priority rules). The header lists every change with respect to the original (CHANGE 1-5): idle technicians without the skill no longer take a part; the lowest-priority repair is interrupted; interrupted service is credited to utilization; an interrupted part is re-dispatched (cascade); FCFS within a priority level. Entry point `simulation_batch_run(...)`. |
| `simulation_pso_paper_original.py` | The original 2019 priority simulator, kept for reference only (it has the three faults corrected in v2; do not use for results). |
| `simulation_fcfs_2019.py` | The FCFS-only simulator used for the designs of the 2019 study (Sleptchenko et al., 2019). Sound; agrees with the compiled simulator on the designs of the test bed (Appendix B.3). |
| `surge_simulation.py` | The Python transient (surge) engine: warm-up, baseline, surge and recovery phases with time-varying failure rates. Passes the exact M(t)/M/inf check. Its dispatching differs slightly from the paper's rules (idle technician by index, not longest idle); the compiled `surge_fast.py` in `../experiments/transient_surge/surge_tools` implements the paper's rules and is the engine used in Section 6. `surge_fast.py` imports this module for its helper functions. |
| `rerun_common.py` | Loader and evaluation helpers that run the rule comparison and the one-SKU surges on `simulation_pso_paper_v2.py` (the notebook versions of Sections 5.3 and 6.1 before the compiled simulator). |

Equivalence: `../fast_simulator/test_simulation_fast.py` compares the compiled simulator with `simulation_pso_paper_v2.py`
when this file is on the Python path (copy it next to the test or add this folder to `PYTHONPATH`).
