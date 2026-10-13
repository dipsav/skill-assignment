#!/bin/bash
# 9 Oct: B5 technician absence (busiest / random technician away for 2,500 or 10,000 expected failures, no surge),
# same design arms as B4; queued after B2/B3 (run_B23.sh, PID 4043938).
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
while [ -d /proc/4043938 ]; do sleep 60; done
for abs in busiest random; do
  echo "$(date '+%F %T') B5 $abs FCFS designs"; $PY run_surge_cases.py ../../PSO_results_tuned_polished --out ../../surge_results_B5/fcfs_designs --absence $abs --durations 2500 10000 --rules FCFS mu hmu --workers 20 > B5_${abs}_fcfs.log 2>&1
  echo "$(date '+%F %T') B5 $abs mu designs";   $PY run_surge_cases.py ../../PSO_results_tuned_rule_mu_polished --out ../../surge_results_B5/mu_designs --absence $abs --durations 2500 10000 --rules mu --workers 20 > B5_${abs}_mu.log 2>&1
  echo "$(date '+%F %T') B5 $abs hmu designs";  $PY run_surge_cases.py ../../PSO_results_tuned_rule_hmu_polished --out ../../surge_results_B5/hmu_designs --absence $abs --durations 2500 10000 --rules hmu --workers 20 > B5_${abs}_hmu.log 2>&1
done
echo "$(date '+%F %T') B5 done: $(find ../../surge_results_B5 -name '*.json.gz' | wc -l) records"
