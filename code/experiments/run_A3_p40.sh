#!/bin/bash
# 10 Oct (P40): A3 joint designs under NON-preemptive priorities: tuned BPSO with the rule inside the search (mu, then
# h x mu), then the local-search polish; same seeds and evaluation as A2. Launch ONCE (check: ps -eo args | grep "^bash run_A3_p40.sh").
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python3}
for r in mu hmu; do
  echo "$(date '+%F %T') A3 BPSO rule $r non-preemptive"; $PY rerun_pso_cases.py ../largeCases --all --rule $r --nonpreemptive --workers 20 --out ../PSO_results_tuned_rule_${r}_np > A3_${r}_np.log 2>&1
  echo "$(date '+%F %T') A3 polish rule $r";               $PY polish_designs.py ../PSO_results_tuned_rule_${r}_np --out ../PSO_results_tuned_rule_${r}_np_polished --workers 20 > A3_polish_${r}_np.log 2>&1
done
echo "$(date '+%F %T') A3 done: $(ls ../PSO_results_tuned_rule_mu_np_polished/*.json | wc -l) + $(ls ../PSO_results_tuned_rule_hmu_np_polished/*.json | wc -l) files"
