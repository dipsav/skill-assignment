#!/bin/bash
# 9 Oct: B4 finite-duration surges on the two designs per instance (FCFS design under FCFS/mu/hmu; mu design under mu).
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
echo "$(date '+%F %T') B4 FCFS designs"; $PY run_surge_cases.py ../../PSO_results_tuned_polished --out ../../surge_results_B4/fcfs_designs --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules FCFS mu hmu --workers 20 > B4_fcfs.log 2>&1
echo "$(date '+%F %T') B4 mu designs";   $PY run_surge_cases.py ../../PSO_results_tuned_rule_mu_polished --out ../../surge_results_B4/mu_designs --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules mu --workers 20 > B4_mu.log 2>&1
echo "$(date '+%F %T') B4 hmu designs";  $PY run_surge_cases.py ../../PSO_results_tuned_rule_hmu_polished --out ../../surge_results_B4/hmu_designs --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules hmu --workers 20 > B4_hmu.log 2>&1
echo "$(date '+%F %T') B4 done: $(find ../../surge_results_B4 -name '*.json.gz' | wc -l) records"
