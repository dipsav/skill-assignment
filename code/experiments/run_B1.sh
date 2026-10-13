#!/bin/bash
# 9 Oct: B1 one-SKU surges after B4 (PID 4040023): FCFS designs under FCFS and under mu (sequential arm), mu designs under mu, hmu designs under hmu.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}
while [ -d /proc/4040023 ]; do sleep 60; done
for spec in "../PSO_results_tuned_polished fcfs B1_fcfs_designs" "../PSO_results_tuned_polished mu B1_fcfs_designs_under_mu" "../PSO_results_tuned_rule_mu_polished mu B1_mu_designs" "../PSO_results_tuned_rule_hmu_polished hmu B1_hmu_designs"; do
  set -- $spec; echo "$(date '+%F %T') B1 $3"; $PY sku_surges.py $1 --rule $2 --out tables/$3.csv --deltas 0.10 0.25 0.40 0.60 --workers 20 > $3.log 2>&1
done
echo "$(date '+%F %T') B1 done"
