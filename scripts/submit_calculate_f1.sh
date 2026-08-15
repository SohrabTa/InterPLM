#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -t 1:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/calculate_f1_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/interplm/calculate_f1_%j.err

# Stages 5b and 5c of repos/InterPLM/README.md section 3, the two steps that come
# after the per-shard counting that submit_eval_store.sh does.
#
#   calculate_f1   sums the per-shard tp/fp/tp_per_domain and turns them into a
#                  per-concept F1 table. Runs once per split.
#                  -> <split>_counts/concept_f1_scores.csv
#   report_metrics picks the best (feature, concept) pairing on the VALIDATION
#                  split and reports how those pairs score on TEST, so the reported
#                  number is not the one the pairing was chosen on.
#                  -> test_counts/{heldout_top_pairings.csv, heldout_all_top_pairings.csv}
#
# Entry points are tapify(combine_metrics_across_shards) in
# interplm/analysis/concepts/calculate_f1.py and tapify(report_metrics) in
# interplm/analysis/concepts/report_metrics.py. Both take the default thresholds
# [0, 0.15, 0.5, 0.6, 0.8]; report_metrics defaults --top_threshold 0.5.
#
# Minutes, not hours: this only reads the per-shard count arrays, no activations.
#
#   RERUN_TARGET=diag67k RERUN_SCALE=raw        sbatch scripts/submit_calculate_f1.sh
#   RERUN_TARGET=diag67k RERUN_SCALE=normalized sbatch scripts/submit_calculate_f1.sh

set -euo pipefail

# Prebuilt image, same reason as submit_eval_store.sh: pulling by name makes each
# task extract ~18 GB into node-local /run and that filled the node on 2026-08-15.
CONTAINER="${RERUN_CONTAINER:-/dss/dsshome1/08/ga25ley2/nvidia+pytorch+25.12-py3.sqsh}"

INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

MOUNTS="${INTERPLM_DIR}:/workspace/InterPLM"
MOUNTS="${MOUNTS},${CROSSCODE_DIR}:/workspace/crosscode"
MOUNTS="${MOUNTS},${DATA_DIR}:/workspace/data"

RERUN_TARGET="${RERUN_TARGET:-score345}"
RERUN_SCALE="${RERUN_SCALE:-normalized}"

case "${RERUN_TARGET}" in
  score345) EVALSET="uniprotkb_modern_score345";     RUN_TAG="full_uniref" ;;
  diag67k)  EVALSET="uniprotkb_modern_score45_67k";  RUN_TAG="auxfix_scalediag" ;;
  *) echo "Unknown RERUN_TARGET '${RERUN_TARGET}'. Use score345 or diag67k." >&2; exit 2 ;;
esac

case "${RERUN_SCALE}" in
  raw|normalized) ;;
  *) echo "Unknown RERUN_SCALE '${RERUN_SCALE}'. Use raw or normalized." >&2; exit 2 ;;
esac

ANNOTS="/workspace/data/eval_dataset/${EVALSET}/processed_annotations"
OUT_ROOT="/workspace/data/crosscoder_eval/${RUN_TAG}/${RERUN_SCALE}/${EVALSET}"

# Refuse to run on an incomplete eval: calculate_f1 sums whatever shards it finds
# and would silently report an F1 computed on a subset.
HOST_OUT="${DATA_DIR}${OUT_ROOT#/workspace/data}"
for split in valid test; do
  have=$(ls "${HOST_OUT}/${split}_counts"/shard_*_counts.npz 2>/dev/null | wc -l)
  want=$(ls -d "${DATA_DIR}${ANNOTS#/workspace/data}/${split}"/shard_* 2>/dev/null | wc -l)
  if [ "${want}" -gt 0 ] && [ "${have}" -ne "${want}" ]; then
    echo "ERROR: ${split} has ${have} count files but the eval set defines ${want} shards." >&2
    echo "Finish submit_eval_store.sh before combining, or the F1 is computed on a subset." >&2
    exit 1
  fi
  echo "${split}: ${have} shard count files"
done

export PYTHONPATH="/workspace/InterPLM"

echo "Target : ${RERUN_TARGET} | scale ${RERUN_SCALE}"
echo "Counts : ${OUT_ROOT}"
echo "Starting on $(hostname) at $(date)"
START_TIME=$(date +%s)

srun --container-image="${CONTAINER}" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/InterPLM" \
     bash -c "if .venv/bin/python -c 'import interplm, scipy, crosscode' 2>/dev/null; then \
       echo 'venv: reusing /workspace/InterPLM/.venv'; \
     else \
       echo 'venv: building' && \
       uv venv --python 3.12 && source .venv/bin/activate && \
       uv pip install -r requirements.txt && \
       uv pip install -e /workspace/crosscode && \
       uv pip install -e . ; \
     fi && \
     source .venv/bin/activate && \
     for SPLIT in valid test; do \
       echo \"=== calculate_f1 \${SPLIT} ===\" && \
       python -m interplm.analysis.concepts.calculate_f1 \
         --eval_res_dir ${OUT_ROOT}/\${SPLIT}_counts \
         --eval_set_dir ${ANNOTS}/\${SPLIT} || exit 1; \
     done && \
     echo '=== report_metrics (pairs chosen on valid, scored on test) ===' && \
     python -m interplm.analysis.concepts.report_metrics \
       --valid_path ${OUT_ROOT}/valid_counts/concept_f1_scores.csv \
       --test_path ${OUT_ROOT}/test_counts/concept_f1_scores.csv"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at $(date)"
echo "Total duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
