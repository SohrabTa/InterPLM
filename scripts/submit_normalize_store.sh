#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH -t 2:00:00
#SBATCH --mem=64G
#SBATCH -o logs/normalize_store_%j.out
#SBATCH -e logs/normalize_store_%j.err

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
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk4/jumprelu_global_10990182}"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836}"
    ;;
  *)
    echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345 or diag67k." >&2
    exit 2
    ;;
esac

# CAUTION: for diag67k this overwrites the auxfix checkpoint's existing
# feature_stats/max.npy and ae_normalized.pt, which the hand-in numbers were
# measured with. Set RERUN_SAE_DIR to a copy of that directory if you want to
# keep the originals byte-identical.
ACTS_DIR="/workspace/data/crosscoder_activations/${EVALSET}"

export PYTHONPATH="/workspace/InterPLM"
mkdir -p logs

echo "Normalize from store : ${ACTS_DIR}"
echo "Crosscoder           : ${SAE_DIR}"
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
       --acts_dir ${ACTS_DIR}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Normalize finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
