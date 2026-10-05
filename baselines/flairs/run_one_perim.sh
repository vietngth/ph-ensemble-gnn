#!/bin/bash
# One FLAIRS run with our per-IM logging copy of the authors' eqModelTrain.py (training unchanged): run_one_perim.sh <config> <network> <random_state>
# Copy this script into the GraphNodeAttention clone (see README.md). FLAIRS_PYTHON = python of flairs-env [required].
cd "$(dirname "$0")"
mkdir -p logs results_perim   # eqModelTrain_perim.py writes to results_perim/, which the upstream repo does not have
log=logs/$1_$2_rs$3_perim.log
"${FLAIRS_PYTHON:?set FLAIRS_PYTHON}" eqModelTrain_perim.py with configs/$1.yaml data.network_choice=$2 data.random_state_here=$3 > "$log" 2>&1
rc=$?   # capture first: $(date) inside the echo would reset $?
echo "$(date -Is) perim $1 $2 rs$3 rc=$rc" >> logs/finished.txt
# The authors' script writes fold checkpoints to models/ but never reads them (it predicts with the in-memory model);
# at ~3.4 GB per run they filled the disk, so this run's checkpoints are removed once it has finished.
rm -f models/*.n$2.rs$3.*
