#!/bin/bash
# The fuzzing recorded in the README: decoder 10 x 10 000 synthetic-seed inputs + 2 x 10 000 with local seeds
# (--seeds-dir), guard bands on (EGJ_CANARY=1), 2 decoding threads; then the encoder parameter fuzz. Each part
# holds the shared-PC lock on its own, so other jobs can run in between.
#   bash tools/fuzz_suite.sh <work_dir> [local seeds dir]
set -uo pipefail
cd "$(dirname "$0")/.."
W=$1; SEEDS=${2:-}
PY=${PYTHON:-python}
mkdir -p "$W"
export EGJ_CANARY=1 PYTHONWARNINGS=ignore
for i in $(seq 1 10); do
  $PY -W ignore bench/with_lock.py "exact-gpu-jpeg fuzz part $i/12" "$W/cond_$i.json" -- $PY -W ignore tools/fuzz.py "$W/part_$i" --iters 10000 --seed $((100 + i)) > "$W/log_$i.txt" 2>&1
done
if [ -n "$SEEDS" ]; then for i in 11 12; do
  $PY -W ignore bench/with_lock.py "exact-gpu-jpeg fuzz part $i/12" "$W/cond_$i.json" -- $PY -W ignore tools/fuzz.py "$W/part_$i" --iters 10000 --seed $((100 + i)) --seeds-dir "$SEEDS" > "$W/log_$i.txt" 2>&1
done; fi
unset EGJ_CANARY
$PY -W ignore bench/with_lock.py "exact-gpu-jpeg encoder fuzz" "$W/cond_enc.json" -- $PY -W ignore tools/fuzz_encode.py "$W/encode.json" --iters 3000 --seed 7 > "$W/log_enc.txt" 2>&1
echo "fuzz suite done"
