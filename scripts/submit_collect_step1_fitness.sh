#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o logs/collect_step1_fitness_%j.out
#SBATCH -e logs/collect_step1_fitness_%j.err

# LLM-autointerp Phases B-E for the 454 ProteinGym fitness-tracking features
# (exp03/§5.5 follow-up). REUSES the cached Phase-A binning (job 5673517) via the
# fitness_auxfix/cache/bin_assignments.yaml copy, so no re-scan: --feature-ids @file
# is filtered to the 454 after the cache load (see collect_step1 post-phase-A filter).
# Phase C encodes only the sampled proteins -> minutes on H100. Phase D needs internet
# (compute nodes have it). Output: fitness_auxfix/Per_feature_llm_input.parquet.
# No API cost here; DeepSeek scoring (generate_descriptions) is a separate step.

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"
HF_HOME="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
SAE_DIR="/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836"

MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${HF_HOME}:/workspace/hf_home,${CKPT_DIR}:/workspace/model_checkpoints,${CROSSCODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data"

export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/InterPLM"
export PYTHONUNBUFFERED=1

mkdir -p logs
echo "Starting collect_step1 B-E (454 fitness features) on $(hostname) at $(date)"
START_TIME=$(date +%s)

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
     --output-dir /workspace/data/llm_autointerp/fitness_auxfix \
     --feature-ids @/workspace/InterPLM/fitness_ids.txt \
     --shards-to-search all \
     --negatives random \
     --seed 42 \
     --n-per-bin 30 \
     --n-zero 500"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "collect_step1 B-E (fitness) finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
