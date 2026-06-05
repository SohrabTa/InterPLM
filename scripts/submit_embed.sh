#!/bin/bash
# 67k ProtT5 embedding harvest — prepared for a FREE V100 node (H100 partition was
# fully allocated on 2026-06-06). ProtT5 loads in fp16 on CUDA, so it fits a 16 GB
# V100; the binding constraint is T5 attention memory (O(L^2), max_length=2000), so
# batch_size is small. Flip the two commented lines back to run on H100 when free.
#
# PRE-FLIGHT (do this once before submitting the 12 h job, so it can't OOM overnight):
#   salloc -p lrz-v100x2 --gres=gpu:1 -t 0:30:00
#   srun --pty bash   # then run embed_annotations.py on ONE shard with --batch_size 16,
#                      # confirm no OOM + note per-shard time; if OOM, drop batch to 8.
#
#SBATCH -p lrz-v100x2                 # was: lrz-hgx-h100-94x4  (H100, fully booked 2026-06-06)
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00                   # was: 1:00:00  (V100 ~3-6x slower than H100)
#SBATCH -o logs/embed_%j.out
#SBATCH -e logs/embed_%j.err

# Define Paths
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

# Mounts: Host:Container
MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data"

# Env
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/InterPLM"

mkdir -p logs

echo "Starting embedding job on $(hostname) at $(date)"
START_TIME=$(date +%s)

# NOTE: input/output dirs now include the eval_dataset/ segment (data was reorganized
# under data/eval_dataset/; the old data/uniprotkb_modern_score45_67k/ path is gone).
# Output is ~1.2 TB fp16 — dssfs02 had 3.6 TB free on 2026-06-06 (shared chair storage).
srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && uv pip install -e . && uv pip install -r requirements.txt && \
     uv run scripts/embed_annotations.py \
     --input_dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/processed_annotations/ \
    --output_dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/analysis_embeddings/prott5/layer_crosscoder \
    --embedder_type prott5 \
    --model_name Rostlab/prot_t5_xl_uniref50 \
    --batch_size 16"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Embedding job finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
