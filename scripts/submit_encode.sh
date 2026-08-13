#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -o logs/encode_%j.out
#SBATCH -e logs/encode_%j.err

# Stage 1 (streaming): ProtT5 -> crosscoder -> sparse activation store.
#
# Replaces submit_embed.sh. The 24-layer residual stream never reaches disk, so
# this needs ~16 GB for the score-{3,4,5} set instead of ~6.2 TB. See
# documentation/experiments/02-interplm-eval-pipeline.md, "Streaming eval".
#
# Two configurations, selected by RERUN_TARGET:
#
#   score345   the preprint eval. 208 shards, the full-UniRef50 crosscoder.
#              Convert that checkpoint to JumpReLU first (submit_convert_jumprelu.sh)
#              -- BatchTopK picks the top k*B over the whole batch, so its
#              activations depend on batch composition and are not valid for
#              per-token inference.
#   diag67k    the raw-vs-normalized diagnostic. 84 shards, the AUXFIX crosscoder,
#              because the published 0.335 / 61 concepts / 409 features come from
#              that checkpoint. Do not substitute the full-UniRef50 one.
#
# Resumable: a shard whose acts.npz exists is skipped, and the writes are atomic,
# so a walltime kill costs only the shard in flight.

set -euo pipefail

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"
HF_HOME_HOST="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM"
MOUNTS="${MOUNTS},${CROSSCODE_DIR}:/workspace/crosscode"
MOUNTS="${MOUNTS},${CKPT_DIR}:/workspace/model_checkpoints"
MOUNTS="${MOUNTS},${HF_HOME_HOST}:/workspace/hf_home"
MOUNTS="${MOUNTS},${DATA_DIR}:/workspace/data"

RERUN_TARGET="${RERUN_TARGET:-score345}"

case "${RERUN_TARGET}" in
  score345)
    EVALSET="uniprotkb_modern_score345"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk4/jumprelu_global_10990182}"
    SHARD_RANGE="0 207"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836}"
    SHARD_RANGE="0 83"
    ;;
  *)
    echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345 or diag67k." >&2
    exit 2
    ;;
esac

BATCH_SIZE="${RERUN_BATCH_SIZE:-64}"
OUT_DIR="/workspace/data/crosscoder_activations/${EVALSET}"

export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/InterPLM"

mkdir -p logs

echo "Encode target : ${RERUN_TARGET} (${EVALSET}, shards ${SHARD_RANGE})"
echo "Crosscoder    : ${SAE_DIR}"
echo "Output store  : ${OUT_DIR}"
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
     uv run scripts/encode_activations.py \
       --sae_dir ${SAE_DIR} \
       --checkpoint ae.pt \
       --metadata_dir /workspace/data/eval_dataset/${EVALSET}/processed_annotations \
       --output_dir ${OUT_DIR} \
       --shard_range ${SHARD_RANGE} \
       --batch_size ${BATCH_SIZE} \
       --dtype float32"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Encode finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
