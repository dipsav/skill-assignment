#!/bin/bash
# 10 Oct: R1 surge/absence cells for full cross-training (rerun after the EBO field fix).
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
for spec in "../../PSO_results_fullCT FCFS fullct" "../../PSO_results_fullCT_rule_mu mu fullct_mu"; do
  set -- $spec
  echo "$(date '+%F %T') B4 $3"; $PY run_surge_cases.py $1 --out ../../surge_results_B4/$3 --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules $2 --workers 20 > B4_$3.log 2>&1
  for abs in busiest random; do echo "$(date '+%F %T') B5 $abs $3"; $PY run_surge_cases.py $1 --out ../../surge_results_B5/$3 --absence $abs --durations 2500 10000 --rules $2 --workers 20 > B5_${abs}_$3.log 2>&1; done
done
echo "$(date '+%F %T') R1b done"
