#!/bin/bash
# One locked decode-alone measurement for a speed-up step: bench/speed_step.sh <tag> "<label>"
# PHOTOS=<your JPEG folder>; photos are read locally, results (no file names) go to bench/results/speed/<tag>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${PHOTOS:?set PHOTOS to a folder of your own camera JPEGs}
$PY -W ignore bench/with_lock.py "decode speed step $1" bench/results/speed/$1.conditions.json -- \
    $PY -W ignore bench/decode_alone.py bench/results/speed/$1.json "$D" --label "$2"
