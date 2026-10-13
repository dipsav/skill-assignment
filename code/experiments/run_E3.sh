#!/bin/bash
# 10 Oct: E3 distribution sensitivity of the final designs (FCFS designs under FCFS, mu designs under mu), 9 grid cells.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
echo "$(date '+%F %T') E3 FCFS designs"; $PY distribution_sensitivity.py ../PSO_results_tuned_polished --out tables/E3_fcfs_designs.csv --workers 20 > E3_fcfs.log 2>&1
echo "$(date '+%F %T') E3 mu designs";   $PY distribution_sensitivity.py ../PSO_results_tuned_rule_mu_polished --out tables/E3_mu_designs.csv --workers 20 > E3_mu.log 2>&1
echo "$(date '+%F %T') E3 done"
