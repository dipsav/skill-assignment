# Code of "Simulation-based Optimization of Joint Skill Assignment and Inventory Planning for Resilient Maintenance Systems" (CAIE-D-26-05208)

Three parts, usable separately:

1. `python_simulator/` -- the fixed pure-Python simulators (reference implementations; no compilation).
2. `fast_simulator/` -- the compiled simulator (Numba kernels and an optional C port) as the package `simulation_codes`,
   with the exact references and the validation tests of Appendix B.
3. `experiments/` -- every script of the simulation experiments of Sections 5-6 and the appendices, which run on the
   compiled simulator; each README maps the paper's experiments to the scripts.

The test bed (`largeCases/`) and the Appendix B.2 references are in `../instances/`; `setup.sh` links them and the
design folders of `../designs/` where the scripts expect them (one level above `experiments/`).

Quick start (Linux, Python >= 3.10):

    bash setup.sh                                   # packages, C build, links experiments/ to the package
    cd fast_simulator && python test_simulation_fast.py && cd ..
    cd experiments && python rerun_pso_cases.py ../largeCases --cases 2 --test --published --out ../PSO_results_test   # ~1 min, prints total 90,059

The designs, benchmark runs and summary tables are in the parent folders of this repository; the raw surge records on
Mendeley Data (doi:10.17632/m3v827bj3x.1).
