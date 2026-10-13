#!/bin/bash
# Prepare the code for running: Python packages, the C kernel (optional), and the links that let the experiment
# scripts find the simulator package and the data one level above experiments/ (instances and design sets of the
# repository). Default: symbolic links, no change to the Python environment. With --pip: `pip install -e fast_simulator`.
set -e; cd "$(dirname "$0")"
PY=${PY:-python3}
$PY -m pip install --quiet "numpy>=2" "scipy>=1.14" "numba>=0.60" "deap>=1.4" pandas matplotlib xxhash tqdm || true
bash fast_simulator/csim/build.sh || echo "C build skipped (no gcc?) -- the Numba kernel will be used"
if [ "${1:-}" = "--pip" ]; then
  $PY -m pip install -e fast_simulator
else
  ln -sfn ../fast_simulator/simulation_codes experiments/simulation_codes
  ln -sfn ../fast_simulator/csim experiments/csim
fi
# data where the scripts expect it: ../largeCases, ../sets23_*, ../PSO_results_* relative to experiments/
ln -sfn ../instances/largeCases largeCases
ln -sfn ../instances/sets23_exact.json sets23_exact.json
ln -sfn ../instances/sets23_instances.csv sets23_instances.csv
for d in ../designs/*/; do ln -sfn "$d" "$(basename "$d")"; done
mkdir -p revision/tikz
(cd fast_simulator && $PY -c "from simulation_codes import simulation_fast as sf; print('simulation_codes ok, backend:', sf.BACKEND)")
(cd experiments && $PY -c "from simulation_codes import SimulationInterface_fast; print('experiments see the package: ok')")
