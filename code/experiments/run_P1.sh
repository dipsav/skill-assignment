#!/bin/bash
# 10 Oct: R7.2 preemptive vs non-preemptive priorities on the final designs and full cross-training.
cd "$(dirname "$0")"; export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "$(date '+%F %T') P1 start"; /opt/tljh/user/bin/python3 preemption_comparison.py --out tables/P1_preemption.csv --workers 20 > P1.log 2>&1; echo "$(date '+%F %T') P1 done"
