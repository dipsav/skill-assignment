#!/bin/bash
# Prepare the release for running: build the C kernel (optional), make the package importable from experiments/.
# Default: symbolic links (no change to the Python environment). With --pip: `pip install -e fast_simulator` instead.
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
# data where the experiment scripts expect it (one level above experiments/): the instances and the design sets
ln -sfn ../instances/largeCases largeCases
ln -sfn ../instances/sets23_exact.json sets23_exact.json; ln -sfn ../instances/sets23_instances.csv sets23_instances.csv
for d in ../designs/*/; do ln -sfn "$d" "$(basename "$d")"; done
if false; then
fi
(cd fast_simulator && $PY -c "from simulation_codes import simulation_fast as sf; print('simulation_codes ok, backend:', sf.BACKEND)")
(cd experiments && $PY -c "from simulation_codes import SimulationInterface_fast; print('experiments see the package: ok')")
