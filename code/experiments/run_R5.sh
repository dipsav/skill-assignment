#!/bin/bash
# 11 Oct: R5 -- disruptions on the A3 (non-preemptive mu / h x mu) and B6b (coverage-two) designs: B1, B23, B4, B5; then R0.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}; S="transient_surge"
# folder | rule (sku/multi) | rule (surge) | extra flags | tag
for spec in "../PSO_results_tuned_rule_mu_np_polished mu mu --nonpreemptive mu_np" "../PSO_results_tuned_rule_hmu_np_polished hmu hmu --nonpreemptive hmu_np" \
            "../PSO_results_tuned_rule_mu_cov2_polished mu mu '' mu_cov2" "../PSO_results_tuned_cov2_polished fcfs FCFS '' fcfs_cov2"; do
  eval set -- $spec; F=$1; R1=$2; R2=$3; X=$4; T=$5
  echo "$(date '+%F %T') R5 B1 $T";  $PY sku_surges.py $F --rule $R1 $X --out tables/B1_${T}_designs.csv --workers 20 > B1_$T.log 2>&1
  echo "$(date '+%F %T') R5 B23 $T"; $PY multi_surges.py $F --rule $R1 $X --out tables/B23_${T}_designs.csv --workers 20 > B23_$T.log 2>&1
  (cd "$S"; echo "$(date '+%F %T') R5 B4 $T"; $PY run_surge_cases.py $F --out ../surge_results_B4/${T}_designs --gamma 1.25 1.5 2 --durations 2500 10000 20000 --rules $R2 $X --workers 20 > B4_$T.log 2>&1
   for abs in busiest random; do echo "$(date '+%F %T') R5 B5 $abs $T"; $PY run_surge_cases.py $F --out ../surge_results_B5/${T}_designs --absence $abs --durations 2500 10000 --rules $R2 $X --workers 20 > B5_${abs}_$T.log 2>&1; done)
done
echo "$(date '+%F %T') R5 R0 margins"; $PY resilience_margin.py > R0_margin.log 2>&1
echo "$(date '+%F %T') R5 done"
