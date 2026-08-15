#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -t 2:00:00
#SBATCH --mem=64G
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/normalize_store_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/normalize_store_%j.err

# Stage 2 (from the activation store): per-feature maxima -> ae_normalized.pt.
#
# The embedding-based normalize re-ran the crosscoder over every shard in 13
# feature chunks and took 58 minutes on a GPU (job 5671079). With the store the
# activations already exist, so this is one pass of CSR column maxima: no PLM, no
# crosscoder forward, NO GPU. Hence the cpu partition.
#
# Writes <sae_dir>/feature_stats/max.npy and <sae_dir>/ae_normalized.pt. Both the
# eval (--normalize_features) and collect read the first of those.
#
# RERUN_TARGET selects which store to read, matching submit_encode.sh.
#
# --qos=cpu is required, not cosmetic. Our default QOS is gpu, and submitting to
# lrz-cpu under it fails at once with "Invalid qos specification". The cpu QOS
# allows 2 days of walltime, 10 running jobs and 50 submitted jobs per user.

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

case "${RERUN_TARGET}" in
  score345)
    EVALSET="uniprotkb_modern_score345"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref50/jumprelu_global_10990182}"
    # New checkpoint directory, created by submit_convert_jumprelu.sh, so writing
    # the normalization into it is the normal convention.
    OUT_DIR="${RERUN_OUT_DIR:-${SAE_DIR}}"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836}"
    # Write OUTSIDE the auxfix checkpoint. Its ae_normalized.pt and
    # feature_stats/max.npy produced the hand-in numbers and must stay unchanged.
    OUT_DIR="${RERUN_OUT_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/scalediag_normalize_2519836}"
    ;;
  *)
    echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345 or diag67k." >&2
    exit 2
    ;;
esac

ACTS_DIR="/workspace/data/crosscoder_activations/${EVALSET}"

# Never write over an existing normalization. The auxfix checkpoint's
# ae_normalized.pt and feature_stats/max.npy back the hand-in numbers.
HOST_OUT="${CKPT_DIR}${OUT_DIR#/workspace/model_checkpoints}"
if [ -e "${HOST_OUT}/feature_stats/max.npy" ] || [ -e "${HOST_OUT}/ae_normalized.pt" ]; then
  echo "ERROR: ${HOST_OUT} already holds a normalization. Refusing to overwrite." >&2
  echo "Set RERUN_OUT_DIR to a fresh directory, or delete the old one deliberately." >&2
  exit 1
fi

export PYTHONPATH="/workspace/InterPLM"

echo "Normalize from store : ${ACTS_DIR}"
echo "Crosscoder (read)    : ${SAE_DIR}"
echo "Outputs (write)      : ${OUT_DIR}"
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
     uv run python -m interplm.sae.normalize \
       --sae_dir ${SAE_DIR} \
       --acts_dir ${ACTS_DIR} \
       --out_dir ${OUT_DIR}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Normalize finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
