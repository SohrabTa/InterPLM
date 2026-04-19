#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -o logs/embed_%j.out
#SBATCH -e logs/embed_%j.err

# Define Mounts
CODE_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2"

# Mounts: Host:Container
MOUNTS="${CODE_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data"

# Env
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/InterPLM"

mkdir -p logs

echo "Starting training run on $(hostname) at $(date)"

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.11 && uv pip install -e . && uv pip install -r requirements.txt && \
     uv run scripts/embed_annotations.py \
     --input_dir /workspace/data/uniprotkb_modern_score5_5k/processed_annotations/ \
    --output_dir /workspace/data/uniprotkb_modern_score5_5k/analysis_embeddings/prott5/layer_crosscoder \
    --embedder_type prott5 \
    --model_name Rostlab/prot_t5_xl_uniref50 \
    --batch_size 256"