#!/bin/bash
# 9 Oct: B2/B3 all-SKU and correlated surges after B1 (PID 4041997), same four arms as B1.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
while [ -d /proc/4041997 ]; do sleep 60; done
for spec in "../PSO_results_tuned_polished fcfs B23_fcfs_designs" "../PSO_results_tuned_polished mu B23_fcfs_designs_under_mu" "../PSO_results_tuned_rule_mu_polished mu B23_mu_designs" "../PSO_results_tuned_rule_hmu_polished hmu B23_hmu_designs"; do
  set -- $spec; echo "$(date '+%F %T') B23 $3"; $PY multi_surges.py $1 --rule $2 --out tables/$3.csv --workers 20 > $3.log 2>&1
done
echo "$(date '+%F %T') B23 done"
