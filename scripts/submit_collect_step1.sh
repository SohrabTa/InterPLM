#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o logs/collect_step1_%j.out
#SBATCH -e logs/collect_step1_%j.err

# LLM-autointerp Phase A only: per-protein activation binning over all 84 shards
# of the 67k eval set, through the AuxK-fixed normalized JumpReLU crosscoder.
# Produces cache/bin_assignments.yaml (11 bins: zero + 10x0.1), which the later
# full collect_step1 run (Phases B-E) reuses. GPU-only, no internet needed.

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

echo "Starting LLM-autointerp Phase A (binning) run on $(hostname) at $(date)"
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
     uv run interplm/llm/collect_step1.py \
     --sae-dir ${SAE_DIR} \
     --embeddings-dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/analysis_embeddings/prott5/layer_crosscoder \
     --metadata-dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/processed_annotations \
     --output-dir /workspace/data/llm_autointerp/full_auxfix \
     --feature-ids live \
     --shards-to-search all \
     --seed 42 \
     --n-per-bin 30 \
     --phase-a-only"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Collect_step1 (Phase A) job finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
