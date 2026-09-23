#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/encode_esm_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/encode_esm_%j.err

# Stage 1 for roadmap PP-10a: ESM-2-650M -> InterPLM's six released SAEs -> six sparse stores.
#
# One ESM-2 forward pass per batch, then layers 1, 9, 18, 24, 30 and 33 each go through their own
# SAE into their own store under
#   /workspace/data/crosscoder_activations/uniprotkb_modern_score345_interplm_esm2_650m/layer_<L>
# See scripts/encode_activations_esm.py for what is copied from InterPLM's code and what is added.
#
# Before the first submit, two things must be on the cluster, because the job runs offline:
#   - model_checkpoints/interplm_esm2_650m/layer_<L>/{ae.pt, config.yaml, SOURCE.json}, made by
#     scripts/prepare_interplm_esm_saes.py
#   - facebook/esm2_t33_650M_UR50D at revision 08e4846e... in hf_home/hub
#
# Split the 208 shards over four jobs that run at the same time, as submit_encode.sh does for the
# crosscoder. A shard is skipped only when all six layers have acts.npz, and the writes are atomic,
# so overlapping ranges and a walltime kill are both safe:
#   RERUN_SHARD_RANGE="0 51"    sbatch scripts/submit_encode_esm.sh
#   RERUN_SHARD_RANGE="52 103"  sbatch scripts/submit_encode_esm.sh
#   RERUN_SHARD_RANGE="104 155" sbatch scripts/submit_encode_esm.sh
#   RERUN_SHARD_RANGE="156 207" sbatch scripts/submit_encode_esm.sh
#
# -c 8: the six layers are saved in parallel threads, and the compressed save is the largest CPU
# cost: about 95 s per shard on one core, from 257 M nonzeros in shard 0 at 1.38 s per 3.75 M
# (measured on the M1, 2026-09-23).
#
# Size: the six stores of shard 0 take 1.2 GB (305,028 residues, M1 run of 2026-09-23), thus about
# 250 GB for the 208 shards of the eval set.

set -euo pipefail

CONTAINER="${RERUN_CONTAINER:-/dss/dsshome1/08/ga25ley2/nvidia+pytorch+25.12-py3.sqsh}"

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

EVALSET="uniprotkb_modern_score345"
SAE_ROOT="/workspace/model_checkpoints/interplm_esm2_650m"
SHARD_RANGE="${RERUN_SHARD_RANGE:-0 207}"
BATCH_SIZE="${RERUN_BATCH_SIZE:-64}"
# The store name differs from the eval-set name. A shard whose acts.npz exists is skipped, so a
# shared name would hand back the crosscoder activations already under that path.
OUT_ROOT="${RERUN_OUT_DIR:-/workspace/data/crosscoder_activations/${EVALSET}_interplm_esm2_650m}"

for L in 1 9 18 24 30 33; do
  if [ ! -f "${CKPT_DIR}/interplm_esm2_650m/layer_${L}/ae.pt" ]; then
    echo "ERROR: ${CKPT_DIR}/interplm_esm2_650m/layer_${L}/ae.pt is missing. Run prepare_interplm_esm_saes.py first." >&2
    exit 1
  fi
done

export HF_HOME="/workspace/hf_home"
export HF_HUB_OFFLINE=1
export PYTHONPATH="/workspace/InterPLM"
export UV_LINK_MODE=copy

echo "Encode : ESM-2-650M, layers 1 9 18 24 30 33 (${EVALSET}, shards ${SHARD_RANGE})"
echo "SAEs   : ${SAE_ROOT}"
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
     uv run scripts/encode_activations_esm.py \
       --metadata_dir /workspace/data/eval_dataset/${EVALSET}/processed_annotations \
       --output_root ${OUT_ROOT} \
       --sae_root ${SAE_ROOT} \
       --shard_range ${SHARD_RANGE} \
       --batch_size ${BATCH_SIZE}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Encode finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
