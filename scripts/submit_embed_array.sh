#!/bin/bash
# Parallel 67k ProtT5 embedding harvest across the idle V100s (H100 was fully
# booked 2026-06-06). 84 shards split over 8 array tasks (~11 shards each),
# 1 V100 per task -> ~7 h instead of ~53 h on a single GPU.
#
# Isolation notes (important for running 8 tasks concurrently against the same
# bind-mounted repo):
#  - Each task builds its venv in container-local /tmp (UV_PROJECT_ENVIRONMENT),
#    NOT the shared .venv on the mount, so installs don't collide.
#  - Non-editable `uv pip install .` (not `-e .`) avoids a shared egg-info write.
#  - Output is per-shard (shard_N/embeddings.pt), so disjoint slices never clash.
#
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o logs/embed_arr_%A_%a.out
#SBATCH -e logs/embed_arr_%A_%a.err

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data"
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/InterPLM"
mkdir -p logs

# 84 shards / 8 tasks = 11 shards each (last task gets fewer; python clamps the slice).
CHUNK=11
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): shard slice [${START}:${END}] at $(date)"
T0=$(date +%s)

# Uses the SHARED venv prebuilt by submit_embed_setup.sh on the data filesystem
# (/workspace/data/embed_venv). No pip install here: 8 concurrent installs blew the
# small HOME uv-cache quota. Tasks only READ the venv (concurrency-safe) and run.
VENV=/workspace/data/embed_venv
srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "${VENV}/bin/python scripts/embed_annotations.py \
     --input_dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/processed_annotations/ \
     --output_dir /workspace/data/eval_dataset/uniprotkb_modern_score45_67k/analysis_embeddings/prott5/layer_crosscoder \
     --embedder_type prott5 \
     --model_name Rostlab/prot_t5_xl_uniref50 \
     --batch_size 16 \
     --shard_start ${START} --shard_end ${END}"
T1=$(date +%s)
echo "Task ${SLURM_ARRAY_TASK_ID} done in $(( (T1-T0)/60 )) min at $(date)"
