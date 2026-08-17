#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/encode_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/encode_%j.err

# Stage 1 (streaming): ProtT5 -> crosscoder -> sparse activation store.
#
# Replaces submit_embed.sh. The 24-layer residual stream never reaches disk, so
# this needs ~16 GB for the score-{3,4,5} set instead of ~6.2 TB. See
# documentation/experiments/02-interplm-eval-pipeline.md, "Streaming eval".
#
# Two configurations, selected by RERUN_TARGET:
#
#   score345   the preprint eval. 208 shards, the full-UniRef50 crosscoder.
#              Convert that checkpoint to JumpReLU first (submit_convert_jumprelu.sh)
#              -- BatchTopK picks the top k*B over the whole batch, so its
#              activations depend on batch composition and are not valid for
#              per-token inference.
#   diag67k    the raw-vs-normalized diagnostic. 84 shards, the AUXFIX crosscoder,
#              because the published 0.335 / 61 concepts / 409 features come from
#              that checkpoint. Do not substitute the full-UniRef50 one.
#
# Resumable: a shard whose acts.npz exists is skipped, and the writes are atomic,
# so a walltime kill costs only the shard in flight.
#
# Walltime: ask for what the job needs, not for headroom. MEASURED on job 5749642
# (2026-08-15, 84 diag67k shards, batch 64, one H100):
#   per shard   46.9s min, 48.4s median, 69.7s max, at ~236k residues/shard
#   shard work  69.0 min      env build + model load  ~5.5 min      total 1h14m
# That is 4,790 residues/s. The old embed job 5616125 did the same set at 8,840
# residues/s, so the crosscoder encode plus the CSR build and compression roughly
# doubles the cost of the ProtT5 forward. Do not assume the store is free.
#
# score345 scales by residues, and its shards are bigger (~301k against ~236k),
# so ~62s/shard x 208 = ~3.6h + build. That does NOT fit the 4h above with any
# margin, so split it in two and run them at the same time:
#   RERUN_SHARD_RANGE="0 103"   sbatch --time=3:00:00 scripts/submit_encode.sh
#   RERUN_SHARD_RANGE="104 207" sbatch --time=3:00:00 scripts/submit_encode.sh
#   diag67k                   :  RERUN_TARGET=diag67k sbatch --time=2:00:00 scripts/submit_encode.sh
# Short jobs also start sooner: in the 2026-08-14 queue a 6h request was scheduled
# 27h later than a 2h one, because 116 of the 180 jobs ahead held 2-day
# reservations and only short jobs backfilled into the gaps. A walltime kill stays
# cheap either way, because finished shards are skipped on resubmit.

set -euo pipefail

# Mount the prebuilt image instead of pulling by name: pulling makes each task
# extract ~18 GB into node-local /run, which cost 16 of 20 workers on 2026-08-15
# (see submit_eval_store.sh) and costs every job several minutes even when it fits.
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

RERUN_TARGET="${RERUN_TARGET:-score345}"

# EVALSET names the eval set whose shards and annotations are read. STORE_NAME
# names the activation store written. They are the same for the first two targets
# and MUST differ for fulluniref67k, which re-encodes an eval set that already has
# a store built by a different crosscoder.
case "${RERUN_TARGET}" in
  score345)
    EVALSET="uniprotkb_modern_score345"
    STORE_NAME="${EVALSET}"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref50/jumprelu_global_10990182}"
    SHARD_RANGE="0 207"
    ;;
  diag67k)
    EVALSET="uniprotkb_modern_score45_67k"
    STORE_NAME="${EVALSET}"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/jumprelu_global_2519836}"
    SHARD_RANGE="0 83"
    ;;
  fulluniref67k)
    # The controlled comparison: same eval set as the auxfix run (67k {4,5}, 219
    # concepts, same split), different crosscoder. score345 changed the model AND
    # the eval set at once, so it cannot say what the retrain alone bought.
    #
    # The store name MUST differ from the eval-set name here. A shard whose
    # acts.npz exists is skipped, so writing to the default path would silently
    # "finish" in seconds and hand back the AUXFIX activations already sitting
    # there, labelled as this run.
    EVALSET="uniprotkb_modern_score45_67k"
    STORE_NAME="uniprotkb_modern_score45_67k_fulluniref"
    SAE_DIR="${RERUN_SAE_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref50/jumprelu_global_10990182}"
    SHARD_RANGE="0 83"
    ;;
  *)
    echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345, diag67k or fulluniref67k." >&2
    exit 2
    ;;
esac

# Override to split a target across concurrent jobs, e.g. RERUN_SHARD_RANGE="0 103".
# Safe to overlap ranges: a shard whose acts.npz exists is skipped.
SHARD_RANGE="${RERUN_SHARD_RANGE:-${SHARD_RANGE}}"

BATCH_SIZE="${RERUN_BATCH_SIZE:-64}"
# Override for probes. The resume logic skips a shard whose acts.npz exists and
# meta.json records sae_dir and checkpoint but NOT theta, so a shard encoded
# under one threshold would be silently kept if the checkpoint were re-converted
# in place. Write throwaway runs somewhere else rather than into the real store.
OUT_DIR="${RERUN_OUT_DIR:-/workspace/data/crosscoder_activations/${STORE_NAME}}"

export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/InterPLM"


echo "Encode target : ${RERUN_TARGET} (${EVALSET}, shards ${SHARD_RANGE})"
echo "Crosscoder    : ${SAE_DIR}"
echo "Output store  : ${OUT_DIR}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="${CONTAINER}" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && \
     source .venv/bin/activate && \
     uv pip install -r requirements.txt && \
     uv pip install -e /workspace/crosscode && \
     uv pip install -e . && \
     uv run scripts/encode_activations.py \
       --sae_dir ${SAE_DIR} \
       --checkpoint ae.pt \
       --metadata_dir /workspace/data/eval_dataset/${EVALSET}/processed_annotations \
       --output_dir ${OUT_DIR} \
       --shard_range ${SHARD_RANGE} \
       --batch_size ${BATCH_SIZE} \
       --dtype float32"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Encode finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
