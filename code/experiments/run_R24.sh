#!/bin/bash
# 10 Oct: R2 (adaptive capacity: FCFS-operated shop switches to the mu rule only during the disruption, preemptive and
# non-preemptive) and R4 (non-preemptive arms: FCFS design under NP mu throughout; mu design under NP mu) for B1/B23/B4/B5.
# Waits for R1 (PID 829133). The A3 (NP-optimized) designs are added when the P40 run is back.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1; PY=${PY:-python3}; S="transient_surge"
while [ -d /proc/829133 ]; do sleep 60; done
echo "$(date '+%F %T') R4 B1/B23 NP"
$PY sku_surges.py ../PSO_results_tuned_polished --rule mu --nonpreemptive --out tables/B1_fcfs_designs_under_mu_np.csv --workers 20 > B1_fcfs_np.log 2>&1
$PY sku_surges.py ../PSO_results_tuned_rule_mu_polished --rule mu --nonpreemptive --out tables/B1_mu_designs_np.csv --workers 20 > B1_mu_np.log 2>&1
$PY multi_surges.py ../PSO_results_tuned_polished --rule mu --nonpreemptive --out tables/B23_fcfs_designs_under_mu_np.csv --workers 20 > B23_fcfs_np.log 2>&1
$PY multi_surges.py ../PSO_results_tuned_rule_mu_polished --rule mu --nonpreemptive --out tables/B23_mu_designs_np.csv --workers 20 > B23_mu_np.log 2>&1
cd "$S"
G="--gamma 1.25 1.5 2 --durations 2500 10000 20000"
echo "$(date '+%F %T') R2 B4 switch"
$PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B4/fcfs_designs $G --rules FCFS --switch-to mu --switch-end disruption --workers 20 > B4_sw_p.log 2>&1
$PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B4/fcfs_designs $G --rules FCFS --switch-to mu --switch-np --switch-end disruption --workers 20 > B4_sw_np.log 2>&1
echo "$(date '+%F %T') R4 B4 NP"
$PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B4/fcfs_designs $G --rules mu --nonpreemptive --workers 20 > B4_fcfs_np.log 2>&1
$PY run_surge_cases.py ../PSO_results_tuned_rule_mu_polished --out ../surge_results_B4/mu_designs $G --rules mu --nonpreemptive --workers 20 > B4_mu_np.log 2>&1
for abs in busiest random; do
  echo "$(date '+%F %T') R2/R4 B5 $abs"
  $PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B5/fcfs_designs --absence $abs --durations 2500 10000 --rules FCFS --switch-to mu --switch-end disruption --workers 20 > B5_${abs}_sw_p.log 2>&1
  $PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B5/fcfs_designs --absence $abs --durations 2500 10000 --rules FCFS --switch-to mu --switch-np --switch-end disruption --workers 20 > B5_${abs}_sw_np.log 2>&1
  $PY run_surge_cases.py ../PSO_results_tuned_polished --out ../surge_results_B5/fcfs_designs --absence $abs --durations 2500 10000 --rules mu --nonpreemptive --workers 20 > B5_${abs}_fcfs_np.log 2>&1
  $PY run_surge_cases.py ../PSO_results_tuned_rule_mu_polished --out ../surge_results_B5/mu_designs --absence $abs --durations 2500 10000 --rules mu --nonpreemptive --workers 20 > B5_${abs}_mu_np.log 2>&1
done
echo "$(date '+%F %T') R24 done"
