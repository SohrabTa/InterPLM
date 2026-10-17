#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -t 6:00:00
#SBATCH --mem=96G
#SBATCH -c 4
#SBATCH --array=0-0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/pp06a_cpu_%x_%A.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/pp06a_cpu_%x_%A.err

# Roadmap PP-06a: the three CPU stages of one dictionary (normalize, eval, F1) in one job.
#
# All our sessions share one user, and the cpu QOS allows 50 submitted jobs for each user.
# Three chained jobs for each of the 7 dictionaries would hold 21 slots while the encode runs.
# This job holds 1 slot for each dictionary. It runs the three stage scripts one after the other
# with bash, so each stage keeps its own code. The #SBATCH lines of the stage scripts do not
# apply. Their srun steps get this allocation, which is the largest of the three requests:
#   submit_normalize_store.sh  2 h, 64 GB            measured 10 min (crosscoder, job 5750568)
#   submit_eval_store.sh       12 h, 96 GB, 4 CPUs   matmul, one worker for the 208 shards
#   submit_calculate_f1.sh     1 h, 32 GB, 4 CPUs    measured 14 to 32 min (jobs 5773980, 5751771)
# --array=0-0 gives the eval stage SLURM_ARRAY_TASK_ID=0 and SLURM_ARRAY_TASK_COUNT=1, that is
# one worker for all shards. The stage output goes to this job log.
#
# A second run of the same dictionary continues the work. It skips the normalize stage if the
# two outputs of that stage exist, and the eval stage skips each shard whose counts exist.
#
# Usage, from ~/code/InterPLM on the login node, after the two encode jobs A and B:
#   RERUN_DICT=sae0_l7 sbatch -J pp06a_sae0_l7 --dependency=afterok:A:B scripts/submit_pp06a_cpu_chain.sh
# RERUN_DICT is one of PP06A_DICTS in scripts/pp06a_dictionaries.sh.

set -euo pipefail

INTERPLM_DIR="${RERUN_INTERPLM_DIR:-/dss/dsshome1/08/ga25ley2/code/InterPLM}"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"

export RERUN_TARGET=pp06a
export RERUN_SCALE=normalized
export RERUN_DICT

source "${INTERPLM_DIR}/scripts/pp06a_dictionaries.sh"
pp06a_check_dict
HOST_SAE_DIR="${CKPT_DIR}$(pp06a_ckpt "${RERUN_DICT}" | sed 's|^/workspace/model_checkpoints||')"

echo "PP-06a CPU chain for ${RERUN_DICT} (${HOST_SAE_DIR}), job ${SLURM_JOB_ID:-none}"

if [ -e "${HOST_SAE_DIR}/feature_stats/max.npy" ] && [ -e "${HOST_SAE_DIR}/ae_normalized.pt" ]; then
  echo "=== normalize: skipped, ${HOST_SAE_DIR} holds feature_stats/max.npy and ae_normalized.pt ==="
else
  echo "=== normalize ==="
  bash "${INTERPLM_DIR}/scripts/submit_normalize_store.sh"
fi

echo "=== eval ==="
bash "${INTERPLM_DIR}/scripts/submit_eval_store.sh"

echo "=== calculate_f1 ==="
bash "${INTERPLM_DIR}/scripts/submit_calculate_f1.sh"

echo "PP-06a CPU chain for ${RERUN_DICT} finished at $(date)"
