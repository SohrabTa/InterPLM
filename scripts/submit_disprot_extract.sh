#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -t 00:30:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/disprot_extract_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/disprot_extract_%j.err

# Cut the DisProt proteins out of the score-345 activation store (roadmap PP-03).
#
# Submit:
#     sbatch scripts/submit_disprot_extract.sh
#
# The store is 208 shards and 8.8 GB and lives only on the cluster. The 1,552
# DisProt proteins are 435,719 residues, about 0.7% of it. This job reads the
# store once and writes a subset of roughly 120 MB, which we then copy to the
# Mac. Every later step of the DisProt sweep runs locally off that file, so no
# ProtT5 pass and no re-encode is needed. See
# documentation/experiments/07-disprot-disorder.md in the paper repo.
#
# CPU, not GPU: the job reads sparse matrices and slices rows. No model is
# loaded. --qos=cpu is required, because our default QOS is gpu and lrz-cpu
# rejects it with "Invalid qos specification".
#
# No venv build: the script imports only numpy and scipy, which the container
# image already has. That is why this job runs python3 directly instead of the
# uv venv dance that submit_eval_store.sh needs for interplm and crosscode.
#
# The request is deliberately small, so Slurm can backfill it. The job reads one
# shard at a time (about 85 MB of CSR) and accumulates about 120 MB of output, so
# 8 GB and one core are enough. A larger ask only makes it wait: on 2026-09-05 a
# 4-core 64 GB request was scheduled three days out, because 5 of the 12 lrz-cpu
# nodes were in maintenance. Walltime 30 min is still generous, because each
# shard is one npz read and a row slice, with no model in memory.
#
# Inputs (both must exist before you submit):
#   data/disprot_store_protein_map.tsv   from disprot_build_protein_map.py, on the Mac
#   data/crosscoder_activations/uniprotkb_modern_score345/   the store, 208 shards

set -euo pipefail

CONTAINER="${DISPROT_CONTAINER:-/dss/dsshome1/08/ga25ley2/nvidia+pytorch+25.12-py3.sqsh}"

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM,${DATA_DIR}:/workspace/data"

EVALSET="uniprotkb_modern_score345"
ACTS_DIR="/workspace/data/crosscoder_activations/${EVALSET}"
MAP="/workspace/data/disprot_store_protein_map.tsv"
OUT_DIR="/workspace/data/disprot_activations"

echo "Store  : ${ACTS_DIR}"
echo "Map    : ${MAP}"
echo "Output : ${OUT_DIR}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="${CONTAINER}" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     python3 scripts/disprot_extract_activations.py \
       --acts_dir "${ACTS_DIR}" \
       --map "${MAP}" \
       --out_dir "${OUT_DIR}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
