#!/bin/bash
# Re-run the GA on all 128 cases with GA_Optimization_v2 (compiled simulator, guarded interface), 20 in parallel.
# Start it AFTER the PSO run has finished (both use all cores). Run from the fast_simulator folder:  bash run_all_ga.sh
# Results: ../GA_results_rerun/ (one JSON per case, named like the PSO files); log: rerun_ga.log
set -e
cd "$(dirname "$0")"
SRC=../largeCases                 # the instances are read from the original PSO result files
OUT=../GA_results_rerun

if pgrep -f "rerun_pso_cases.py .* --all" > /dev/null && ! pgrep -f "algorithm ga" > /dev/null; then
    echo "the PSO run is still going -- start the GA when it has finished"; exit 1
fi
echo "GA parameters used (check them):"
python3 -c "import rerun_pso_cases as r; print('  ', r.GA_PARAMS)"

echo "1. quick test on one case (tiny population, about a minute)"
rm -rf /tmp/rerun_ga_test
python3 rerun_pso_cases.py "$SRC" --cases 2 --out /tmp/rerun_ga_test --algorithm ga --test | tail -1

echo "2. full run in the background"
nohup python3 rerun_pso_cases.py "$SRC" --all --workers 20 --algorithm ga --out "$OUT" > rerun_ga.log 2>&1 &
echo "   started, PID $!"
echo
echo "progress:  ls $OUT/*.json 2>/dev/null | wc -l      (of 128)"
echo "log:       tail -f rerun_ga.log"
echo "stop:      pkill -f 'rerun_pso_cases.py.*algorithm ga'   (finished cases are kept; run step 2 again to resume)"
