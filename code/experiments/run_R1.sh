#!/bin/bash
# 10 Oct: R1 price of flexibility -- full cross-training (FCFS stock under FCFS; mu-optimal stock under mu) under the
# same disruptions as the designs: B1 one-SKU, B2/B3 multi-SKU, B4 finite surges, B5 technician absence.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}; S="transient_surge"
for spec in "../PSO_results_fullCT fcfs fullct" "../PSO_results_fullCT_rule_mu mu fullct_mu"; do
  set -- $spec
  echo "$(date '+%F %T') R1 B1 $3";  $PY sku_surges.py $1 --rule $2 --out tables/B1_$3.csv --workers 20 > B1_$3.log 2>&1
  echo "$(date '+%F %T') R1 B23 $3"; $PY multi_surges.py $1 --rule $2 --out tables/B23_$3.csv --workers 20 > B23_$3.log 2>&1
done
cd "$S"
R=FCFS; for spec in "../PSO_results_fullCT FCFS fullct" "../PSO_results_fullCT_rule_mu mu fullct_mu"; do
  set -- $spec
  echo "$(date '+%F %T') R1 B4 $3"; $PY run_surge_cases.py $1 --out ../surge_results_B4/$3 --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules $2 --workers 20 > B4_$3.log 2>&1
  for abs in busiest random; do
    echo "$(date '+%F %T') R1 B5 $abs $3"; $PY run_surge_cases.py $1 --out ../surge_results_B5/$3 --absence $abs --durations 2500 10000 --rules $2 --workers 20 > B5_${abs}_$3.log 2>&1
  done
done
echo "$(date '+%F %T') R1 done"
