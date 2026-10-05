#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
# Run a list of FLAIRS configurations (lines "<config> <network> <random_state>") LANES at a time on one GPU.
# The authors' script skips a configuration whose results file exists, so a requeued job continues where it stopped.
# In a job array, "{task}" in LIST is replaced by the array task id (one list per task).
# Submit from the work directory (the GraphNodeAttention clone holding configs/, the run scripts and the run lists),
# or set FLAIRS_WORK. Environment: FLAIRS_PYTHON (python of flairs-env) [required], LIST, LANES,
# RUNNER (run_one.sh, default, or run_one_perim.sh).
cd "${FLAIRS_WORK:-$SLURM_SUBMIT_DIR}" || exit 2
LIST=${LIST//\{task\}/${SLURM_ARRAY_TASK_ID:-}}
export FLAIRS_PYTHON=${FLAIRS_PYTHON:?set FLAIRS_PYTHON to the python of flairs-env} PYTHONUNBUFFERED=1
echo "=== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} list=$LIST lanes=$LANES"
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
xargs -P "${LANES:-6}" -L 1 bash "${RUNNER:-run_one.sh}" < "$LIST" &
wait $!
