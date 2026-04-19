#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/dashboard_%j.out
#SBATCH -e logs/dashboard_%j.err

# Define Paths
CODE_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2"

# Mounts: Host:Container
MOUNTS="${CODE_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data,${CROSSCODE_DIR}:/workspace/crosscode"

# Env
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/InterPLM"
# We set this so the dashboard cache is saved to the data dir properly, or leave default
# export INTERPLM_DATA="/workspace/data"

mkdir -p logs

echo "Starting InterPLM dashboard generation on $(hostname) at $(date)"

# Use Python 3.12 to satisfy crosscode requirements
srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && \
     source .venv/bin/activate && \
     uv pip install -r requirements.txt && \
     uv pip install -e /workspace/crosscode && \
     uv pip install -e . && \
     uv run scripts/create_dashboard.py \
     --sae_path /workspace/data/checkpoints/crosscoder_l8192_k32_bs512_full_2026-03-12_06-03-41/crashed_epoch_0_step_2519836/ae.pt \
     --embeddings_dir /workspace/data/uniprotkb_modern_score5_5k/analysis_embeddings/prott5/layer_crosscoder \
     --metadata_path /workspace/data/uniprotkb_modern_score5_5k/proteins.tsv.gz \
     --concept_enrichment_path /workspace/InterPLM/results/crosscoder_eval/test_counts/heldout_all_top_pairings.csv \
     --layer crosscoder \
     --dashboard_name prott5_crosscoder \
     --model_name prott5 \
     --model_type Rostlab/prot_t5_xl_uniref50 \
     --shard_range 0 7"

echo "Dashboard generation completed!"
