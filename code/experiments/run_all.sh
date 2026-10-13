#!/bin/bash
# Re-optimize all 128 PSO cases with PSO_optimization_v2 (compiled simulator, guarded interface), 20 in parallel.
# Run from the fast_simulator folder:   bash run_all.sh
# Results: ../PSO_results_rerun/ (one JSON per case, same names as the originals); log: rerun_all.log
set -e
cd "$(dirname "$0")"
SRC=../largeCases                 # the original PSO result files
OUT=../PSO_results_rerun

echo "PSO parameters used (check them):"
python3 -c "import rerun_pso_cases as r; print('  ', r.PSO_PARAMS)"

echo "1. compile once, so the 20 workers only load the cache"
python3 -c "from simulation_codes import simulation_fast as sf; import numpy as np; sf.simulate(np.array([0.5]), np.array([[1.0]]), n_requests=1000); print('   compiled')"

echo "2. quick test on one case (tiny swarm, about a minute)"
rm -rf /tmp/rerun_test
python3 rerun_pso_cases.py "$SRC" --cases 2 --out /tmp/rerun_test --test | tail -1

echo "3. full run in the background (survives closing the terminal)"
nohup python3 rerun_pso_cases.py "$SRC" --all --workers 20 --out "$OUT" > rerun_all.log 2>&1 &
echo "   started, PID $!"
echo
echo "progress:  ls $OUT/*.json 2>/dev/null | wc -l      (of 128)"
echo "log:       tail -f rerun_all.log"
echo "stop:      pkill -f rerun_pso_cases.py   (finished cases are kept; run step 3 again to resume)"
