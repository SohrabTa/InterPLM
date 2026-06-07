#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 1:05:00
#SBATCH -o logs/normalize_%j.out
#SBATCH -e logs/normalize_%j.err

# Define Paths
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"
HF_HOME="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
SAE_DIR="/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836"

# Mounts: Host:Container
MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${HF_HOME}:/workspace/hf_home,${CKPT_DIR}:/workspace/model_checkpoints,${CROSSCODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data"

# Env
export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/InterPLM"

mkdir -p logs

echo "Starting Crosscoder normalization run on $(hostname) at $(date)"
START_TIME=$(date +%s)

# Use Python 3.12 to satisfy crosscode requirements
srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && \
     source .venv/bin/activate && \
     uv pip install -r requirements.txt && \
     uv pip install -e /workspace/crosscode && \
     uv pip install -e . && \
     uv run interplm/sae/normalize.py \
     --sae_dir ${SAE_DIR} \
     --aa_embds_dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/analysis_embeddings/prott5/layer_crosscoder"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Normalize job finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
