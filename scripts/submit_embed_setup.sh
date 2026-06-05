#!/bin/bash
# One-time setup: build a SHARED venv on the data filesystem that the embedding
# job array reuses read-only. Done once (not 8x) to avoid blowing the small HOME
# uv-cache quota — the uv cache is also redirected off HOME onto the data FS.
#
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -o logs/embed_setup_%j.out
#SBATCH -e logs/embed_setup_%j.err

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data"
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/InterPLM"
mkdir -p logs

# UV_CACHE_DIR on the data FS (same FS as the venv -> hardlinks work, no HOME quota).
srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "export UV_CACHE_DIR=/workspace/data/uv_cache && \
     VENV=/workspace/data/embed_venv && rm -rf \$VENV && \
     uv venv --python 3.12 \$VENV && \
     uv pip install --python \$VENV/bin/python . && \
     uv pip install --python \$VENV/bin/python -r requirements.txt && \
     \$VENV/bin/python -c 'import torch, interplm; print(\"VENV OK:\", torch.__version__)'"
echo "Setup finished at $(date)"
