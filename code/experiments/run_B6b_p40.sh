#!/bin/bash
# 10 Oct (P40): B6b redundancy -- designs in which every SKU is repaired by at least two technicians (coverage two),
# optimized for the mu rule (preemptive) and for FCFS; tuned BPSO + polish. Starts when A3 has finished. Launch ONCE.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python3}
until grep -q "A3 done" run_A3_p40.log 2>/dev/null; do sleep 120; done
for r in mu fcfs; do
  out=../PSO_results_tuned$([ $r = fcfs ] || echo _rule_$r)_cov2
  echo "$(date '+%F %T') B6b BPSO rule $r coverage 2"; $PY rerun_pso_cases.py ../largeCases --all --rule $r --min-coverage 2 --workers 20 --out $out > B6b_$r.log 2>&1
  echo "$(date '+%F %T') B6b polish rule $r";          $PY polish_designs.py $out --out ${out}_polished --workers 20 > B6b_polish_$r.log 2>&1
done
echo "$(date '+%F %T') B6b done"
