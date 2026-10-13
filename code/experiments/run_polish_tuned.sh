#!/bin/bash
# 8 Oct: polish of the tuned BPSO designs, FCFS then mu (the rule is read from each file). Launch ONCE.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python3}
echo "$(date '+%F %T') polish tuned FCFS"; $PY polish_designs.py ../PSO_results_tuned --out ../PSO_results_tuned_polished --workers 20 > polish_tuned_fcfs.log 2>&1
echo "$(date '+%F %T') polish tuned mu";   $PY polish_designs.py ../PSO_results_tuned_rule_mu --out ../PSO_results_tuned_rule_mu_polished --workers 20 > polish_tuned_mu.log 2>&1
echo "$(date '+%F %T') done: $(ls ../PSO_results_tuned_polished/*.json | wc -l) + $(ls ../PSO_results_tuned_rule_mu_polished/*.json | wc -l) files"
