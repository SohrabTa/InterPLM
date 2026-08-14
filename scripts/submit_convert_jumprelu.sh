#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/convert_jumprelu_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/convert_jumprelu_%j.err

# Stage 0b: convert the full-UniRef50 BatchTopK crosscoder to a JumpReLU gate.
#
# This must run BEFORE the encode, not after. BatchTopK keeps the top k*B
# pre-activations across the whole token batch, so a token's active set depends
# on which other tokens share its batch. That is wrong for per-token inference
# and it would also break the batch-invariance the streaming encode relies on.
# JumpReLU gates each token against a fixed threshold, so the activations a shard
# gets do not depend on how the shard was batched.
#
# Method: Bussmann, Leask & Nanda 2024 (arXiv:2412.06410), global mode -- one
# scalar theta, the mean over batches of the smallest selected pre-activation.
# The encoder, decoder and biases do not change. Rationale and the rejected
# per-latent variant: references/notes/jumprelu-threshold-conversion.md.
#
# The auxfix checkpoint already has jumprelu_global_2519836, so this is only for
# the full-UniRef50 model (roadmap PP-01 -> PP-02).

set -euo pipefail

# Environment recipe: build the venv from InterPLM and run the prott5 script by
# path from /workspace/scc. This is what every working prott5 job does
# (submit_pooled.sh, submit_full_feat.sh, ...), and the reasons are not cosmetic.
# crosscode/pyproject.toml declares `interplm = { path = "../InterPLM" }`, so
# installing crosscode without InterPLM mounted fails with "Distribution not
# found at: file:///workspace/InterPLM" (job 5749590). And the prott5 project
# itself requires Python >=3.13 while these containers build a 3.12 venv, so its
# pyproject must not be installed at all -- only its scripts are used.
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
SCC_DIR="/dss/dsshome1/08/ga25ley2/code/sparse-crosscoders-prott5"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
CKPT_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"
HF_HOME_HOST="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM"
MOUNTS="${MOUNTS},${SCC_DIR}:/workspace/scc"
MOUNTS="${MOUNTS},${CROSSCODE_DIR}:/workspace/crosscode"
MOUNTS="${MOUNTS},${CKPT_DIR}:/workspace/model_checkpoints"
MOUNTS="${MOUNTS},${HF_HOME_HOST}:/workspace/hf_home"
MOUNTS="${MOUNTS},${DATA_DIR}:/workspace/data"

# The training checkpoint to convert, and the corpus theta is estimated on.
CC_DIR="${RERUN_CC_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk4/final_epoch_0_step_10990182}"
# Estimate theta on the same corpus the model was trained on, so the threshold
# matches the activation distribution the model actually saw.
FASTA="${RERUN_FASTA:-/workspace/data/external/uniprot/release-2019_01/uniref/uniref50/chunks_512/chunk_00.fasta}"
OUT_DIR="${RERUN_OUT_DIR:-/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk4/jumprelu_global_10990182}"

export HF_HOME="/workspace/hf_home"

# The converter reads <crosscoder_dir>/config.yaml, but a training checkpoint is
# saved with the name model_cfg.yaml. They are the same file: in the auxfix
# checkpoint config.yaml is a byte-identical copy. Make the copy if it is absent.
HOST_CC="${CKPT_DIR}${CC_DIR#/workspace/model_checkpoints}"
if [ ! -f "${HOST_CC}/config.yaml" ]; then
  if [ -f "${HOST_CC}/model_cfg.yaml" ]; then
    cp "${HOST_CC}/model_cfg.yaml" "${HOST_CC}/config.yaml"
    echo "Copied model_cfg.yaml -> config.yaml in ${HOST_CC}"
  else
    echo "ERROR: neither config.yaml nor model_cfg.yaml in ${HOST_CC}" >&2
    exit 1
  fi
fi

echo "Converting  : ${CC_DIR}"
echo "Theta corpus: ${FASTA}"
echo "Output      : ${OUT_DIR}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "uv venv --python 3.12 && \
     source .venv/bin/activate && \
     uv pip install -r requirements.txt && \
     uv pip install -e /workspace/crosscode && \
     uv pip install -e . && \
     uv pip install scipy pyarrow && \
     python /workspace/scc/convert_batchtopk_to_jumprelu.py \
       --crosscoder_dir ${CC_DIR} \
       --checkpoint model.pt \
       --fasta ${FASTA} \
       --mode global \
       --k 32 \
       --batch_tokens 512 \
       --max_batches 600 \
       --out_dir ${OUT_DIR} \
       --write_converted"

# The converter writes the model to <out_dir>/converted/<checkpoint> and leaves the
# calibration activations in <out_dir>/_calib_acts. Assemble the directory the way
# the auxfix jumprelu_global_2519836 directory is laid out, because that is what
# load_sae and every downstream stage expect:
#   ae.pt, model.pt (byte-identical), config.yaml, jumprelu_threshold.pt
HOST_OUT="${CKPT_DIR}${OUT_DIR#/workspace/model_checkpoints}"
if [ -f "${HOST_OUT}/converted/model.pt" ]; then
  cp "${HOST_OUT}/converted/model.pt"    "${HOST_OUT}/ae.pt"
  cp "${HOST_OUT}/converted/model.pt"    "${HOST_OUT}/model.pt"
  cp "${HOST_OUT}/converted/config.yaml" "${HOST_OUT}/config.yaml"
  echo "Assembled ${HOST_OUT}: ae.pt, model.pt, config.yaml, jumprelu_threshold.pt"
  rm -rf "${HOST_OUT}/converted"
else
  echo "ERROR: ${HOST_OUT}/converted/model.pt is missing; conversion did not complete." >&2
  exit 1
fi

# The calibration step embeds ~600 x 512 tokens of ProtT5 to estimate theta and
# writes them to disk (~30 GB). Delete them: the whole point of the streaming eval
# is that ProtT5 embeddings are not kept.
rm -rf "${HOST_OUT}/_calib_acts"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Conversion finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "Check the reported mean_L0_check is near k=32 before you trust the threshold."
echo "The auxfix conversion recorded mean_L0_check 31.90 with theta 3.312, n_batches 600."
