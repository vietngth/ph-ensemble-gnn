#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --requeue
# JOZ-CNN runs: lines "<network> <window> <seed>", LANES at a time on one GPU; finished runs are skipped on requeue.
# Submit from the work directory (the patched GCNTimeseriesRegression clone holding main_cnn_window.py,
# run_one_joz.sh and the run lists), or set JOZ_WORK. In a job array, "{task}" in LIST is replaced by the task id.
# Environment: JOZ_PYTHON (python of tser-env = flairs-env + matplotlib) [required], LIST, LANES, JOZ_PREFIX, JOZ_DATA.
cd "${JOZ_WORK:-$SLURM_SUBMIT_DIR}" || exit 2
LIST=${LIST//\{task\}/${SLURM_ARRAY_TASK_ID:-}}
export JOZ_PYTHON=${JOZ_PYTHON:?set JOZ_PYTHON to the python of tser-env} PYTHONUNBUFFERED=1
echo "=== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} list=$LIST lanes=$LANES"
xargs -P "${LANES:-6}" -L 1 bash run_one_joz.sh < "$LIST"
