#!/bin/bash
# One JOZ-CNN run with the TSER-GCN authors' main_cnn.py (only change: input window, plus per-IM logging):
#   run_one_joz.sh <network> <window> <seed>
# Each run has its own directory, because main_cnn.py writes fixed checkpoint and results file names.
# Copy this script next to main_cnn_window.py (the patched GCNTimeseriesRegression clone, see README.md).
# Environment:
#   JOZ_PYTHON  python of the TF environment (tser-env)                         [required]
#   JOZ_DATA    data folder laid out as upstream data/ (inputs_ci.npy, targets.npy, station_coords.npy,
#               othernetwork/{inputs_cw.npy,targets.npy,station_coords.npy})     [default: ./data]
#   JOZ_PREFIX  run name prefix: joz (window study) or joz_perim (10 s per-IM runs) [default: joz]
cd "$(dirname "$0")"
data=$(realpath "${JOZ_DATA:-data}")
[ -d "$data" ] || { echo "data folder not found: $data" >&2; exit 2; }
mkdir -p logs
run=runs/${JOZ_PREFIX:-joz}_$1_w$2_rs$3
mkdir -p "$run/models"
ln -sfn "$data" "$run/data"
[ -s "$run/githubresults.csv" ] && { echo "$(date -Is) joz $1 w$2 rs$3 skip(done)" >> logs/finished.txt; exit 0; }
( cd "$run" && JOZ_WINDOW=$2 TF_FORCE_GPU_ALLOW_GROWTH=true "${JOZ_PYTHON:?set JOZ_PYTHON}" ../../main_cnn_window.py "$1" main "$3" > log.txt 2>&1 )
rc=$?   # capture first: $(date) inside the echo would reset $?
echo "$(date -Is) joz $1 w$2 rs$3 rc=$rc" >> logs/finished.txt
