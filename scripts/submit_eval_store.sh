#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -t 12:00:00
#SBATCH --mem=96G
#SBATCH -c 4
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/eval_shard_%A_%a.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/eval_shard_%A_%a.err

# Stage 3 (from the activation store), as a Slurm ARRAY of WORKERS, not of shards.
#
# Submit ten workers, whatever the shard count:
#     RERUN_TARGET=score345 RERUN_SCALE=normalized sbatch --array=0-9 scripts/submit_eval_store.sh
#     RERUN_TARGET=diag67k  RERUN_SCALE=raw        sbatch --array=0-9 scripts/submit_eval_store.sh
#
# Worker i takes shards i, i+10, i+20 ... so ten workers cover 84 or 208 shards
# with no gaps. Each worker builds the venv once and loops, which matters: the
# container import and uv install cost minutes, and one task per shard would pay
# that 84 or 208 times.
#
# Why an array (roadmap PP-02e): the single-job eval took 25 h 16 m on 84 shards
# and would exceed the 48 h cap on 208. Shards are independent and calculate_f1
# sums their counts. Reading the store means no SAE and no GPU, so these are CPU
# tasks.
#
# Why ten workers and not one task per shard: the cpu QOS allows 10 running and
# 50 submitted jobs per user, and Slurm counts array tasks individually. So
# --array=0-83 is rejected outright (QOSMaxSubmitJobPerUserLimit), and a %32
# throttle is a fiction because 10 is the real concurrency cap. Ten workers sit
# exactly at the limit and need one submission. --qos=cpu is required too: our
# default QOS is gpu, and lrz-cpu rejects it with "Invalid qos specification".
#
# Re-running is safe. eval_shard.py skips a shard whose output exists, so a
# resubmit after a walltime kill picks up where the worker stopped.
#
# The venv is shared, because all ten workers mount the same /workspace/InterPLM.
# A worker rebuilds it only when the import check fails, so in the normal case
# nothing writes to it and ten concurrent workers cannot race. Validate with a
# single task (--array=0-0) after any change to requirements, so that at most one
# worker ever builds it.
#
# RERUN_SCALE picks what the thresholds are measured against:
#   normalized  divide by the per-feature max first. Correct. Needs the normalize
#               stage to have run (submit_normalize_store.sh).
#   raw         no division. Reproduces the pre-2026-08-13 behavior, where the
#               JumpReLU gate at theta=3.312 sits above every threshold and the
#               sweep collapses to threshold 0. Kept so the two can be compared.
#
# After the array finishes, combine and report:
#     python -m interplm.analysis.concepts.calculate_f1 --eval_res_dir <out>/valid_counts --eval_set_dir <annots>/valid
#     python -m interplm.analysis.concepts.calculate_f1 --eval_res_dir <out>/test_counts  --eval_set_dir <annots>/test
#     python -m interplm.analysis.concepts.report_metrics --valid_path <out>/valid_counts/concept_f1_scores.csv --test_path <out>/test_counts/concept_f1_scores.csv

set -euo pipefail

# Mount a prebuilt squashfs instead of naming the registry image. Pulling by name
# makes every task extract its own ~18 GB copy into node-local /run/pyxis/<jobid>,
# and Slurm packs many array tasks onto one node: on 2026-08-15 it put 7 tasks on
# cpu-002 and 6 on cpu-006, and 16 of 20 died with "Write failed because No space
# left on device" while building the squashfs. A prebuilt image is mounted
# read-only, so there is nothing to extract, no node-local space is used, and the
# job also starts several minutes sooner.
CONTAINER="${RERUN_CONTAINER:-/dss/dsshome1/08/ga25ley2/nvidia+pytorch+25.12-py3.sqsh}"

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM"
MOUNTS="${MOUNTS},${CROSSCODE_DIR}:/workspace/crosscode"
MOUNTS="${MOUNTS},${CKPT_DIR}:/workspace/model_checkpoints"
MOUNTS="${MOUNTS},${DATA_DIR}:/workspace/data"

RERUN_TARGET="${RERUN_TARGET:-score345}"
RERUN_SCALE="${RERUN_SCALE:-normalized}"

case "${RERUN_TARGET}" in
  score345)
    EVALSET="uniprotkb_modern_score345"
    RUN_TAG="full_uniref"
    LAST_SHARD=207
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref50/jumprelu_global_10990182}"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    RUN_TAG="auxfix_scalediag"
    LAST_SHARD=83
    # With --acts_dir the eval reads only feature_stats/max.npy from this path, so
    # point it at the diagnostic normalization, NOT at the auxfix checkpoint.
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/scalediag_normalize_2519836}"
    ;;
  *)
    echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345 or diag67k." >&2
    exit 2
    ;;
esac

case "${RERUN_SCALE}" in
  normalized) NORM_FLAG="--normalize_features" ;;
  raw)        NORM_FLAG="" ;;
  *) echo "Unknown RERUN_SCALE '${RERUN_SCALE}'. Use normalized or raw." >&2; exit 2 ;;
esac

WORKER="${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. --array=0-9}"
STRIDE="${SLURM_ARRAY_TASK_COUNT:?submit with --array, e.g. --array=0-9}"
SHARDS=$(seq "${WORKER}" "${STRIDE}" "${LAST_SHARD}" | tr '\n' ' ')
ACTS_DIR="/workspace/data/crosscoder_activations/${EVALSET}"
ANNOTS="/workspace/data/eval_dataset/${EVALSET}/processed_annotations"
OUT_ROOT="/workspace/data/crosscoder_eval/${RUN_TAG}/${RERUN_SCALE}/${EVALSET}"

export PYTHONPATH="/workspace/InterPLM"

echo "Worker ${WORKER} of ${STRIDE} | target ${RERUN_TARGET} | scale ${RERUN_SCALE}"
echo "Shards : ${SHARDS}"
echo "Store  : ${ACTS_DIR}"
echo "Output : ${OUT_ROOT}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="${CONTAINER}" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "if .venv/bin/python -c 'import interplm, scipy, crosscode' 2>/dev/null; then \
       echo 'venv: reusing /workspace/InterPLM/.venv'; \
     else \
       echo 'venv: building' && \
       uv venv --python 3.12 && source .venv/bin/activate && \
       uv pip install -r requirements.txt && \
       uv pip install -e /workspace/crosscode && \
       uv pip install -e . ; \
     fi && \
     source .venv/bin/activate && \
     for S in ${SHARDS}; do \
       echo \"=== shard \${S} at \$(date +%H:%M:%S) ===\" && \
       uv run scripts/eval_shard.py \
         --sae_dir ${SAE_DIR} \
         --acts_dir ${ACTS_DIR} \
         --eval_data_root ${ANNOTS} \
         --output_root ${OUT_ROOT} \
         --shard \${S} ${NORM_FLAG} || exit 1; \
     done"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Worker ${WORKER} finished shards ${SHARDS} at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
