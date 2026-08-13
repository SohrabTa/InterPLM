#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH -t 12:00:00
#SBATCH --mem=96G
#SBATCH -c 4
#SBATCH -o logs/eval_shard_%A_%a.out
#SBATCH -e logs/eval_shard_%A_%a.err

# Stage 3 (from the activation store), as a Slurm ARRAY over shards.
#
# Submit with the array range matching the shard count, e.g.
#     RERUN_TARGET=score345 RERUN_SCALE=normalized sbatch --array=0-207%32 scripts/submit_eval_store.sh
#     RERUN_TARGET=diag67k  RERUN_SCALE=raw        sbatch --array=0-83%32  scripts/submit_eval_store.sh
#
# Why an array (roadmap PP-02e): the single-job eval took 25 h 16 m on 84 shards
# and would exceed the 48 h cap on 208. Shards are independent and calculate_f1
# sums their counts. Reading the store means no SAE and no GPU, so these are CPU
# tasks; %32 caps concurrency so the shared filesystem is not hammered.
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
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk4/jumprelu_global_10990182}"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    RUN_TAG="auxfix_scalediag"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836}"
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

SHARD="${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. --array=0-207%32}"
ACTS_DIR="/workspace/data/crosscoder_activations/${EVALSET}"
ANNOTS="/workspace/data/eval_dataset/${EVALSET}/processed_annotations"
OUT_ROOT="/workspace/data/crosscoder_eval/${RUN_TAG}/${RERUN_SCALE}/${EVALSET}"

export PYTHONPATH="/workspace/InterPLM"
mkdir -p logs

echo "Eval shard ${SHARD} | target ${RERUN_TARGET} | scale ${RERUN_SCALE}"
echo "Store  : ${ACTS_DIR}"
echo "Output : ${OUT_ROOT}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && \
     source .venv/bin/activate && \
     uv pip install -r requirements.txt && \
     uv pip install -e /workspace/crosscode && \
     uv pip install -e . && \
     uv run scripts/eval_shard.py \
       --sae_dir ${SAE_DIR} \
       --acts_dir ${ACTS_DIR} \
       --eval_data_root ${ANNOTS} \
       --output_root ${OUT_ROOT} \
       --shard ${SHARD} ${NORM_FLAG}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Shard ${SHARD} finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
