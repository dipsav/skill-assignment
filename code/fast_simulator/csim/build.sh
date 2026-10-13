#!/bin/bash
# Build the C port of the simulation kernel (optional: without it the Numba kernel is used; results are identical).
cd "$(dirname "$0")" && gcc -O3 -march=native -shared -fPIC -o libsimfast.so simfast.c -lm && echo "built libsimfast.so"
