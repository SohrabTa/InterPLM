#!/bin/bash
# Run the full-UniRef50 random-init baseline (roadmap PP-01a) through the four
# eval stages on the score-{3,4,5} set, as one afterok chain.
#
# The baseline is the matched null for the preprint headline (0.479 avg test-F1,
# 187 of 408 concepts, 1020 features). It trained on the same corpus and the same
# 10,990,430-step budget as the preprint crosscoder, with the ProtT5 weights
# shuffled. Everything here is the score345 pipeline with RERUN_TARGET
# baselineuniref345, which differs only in the crosscoder and the store name.
#
# Stage 0b (convert BatchTopK -> JumpReLU) is NOT part of this chain. Run it
# first and pass its job id, because the encode reads its output:
#
#   RERUN_CC_DIR=/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_baseline_uniref_chunk4/final_epoch_0_step_10990182 \
#   RERUN_OUT_DIR=/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_baseline_uniref_chunk4/jumprelu_global_10990182 \
#     sbatch --export=ALL,RERUN_CC_DIR,RERUN_OUT_DIR scripts/submit_convert_jumprelu.sh
#
# Then, from the InterPLM repo root on the login node:
#
#   bash scripts/run_baseline345_chain.sh <convert_job_id>
#   bash scripts/run_baseline345_chain.sh            # no dependency, convert already done
#
# Two encode jobs, not one: 208 shards at ~62 s/shard is ~3.6 h, which does not
# fit the 4 h cap with margin (see submit_encode.sh). They split the shard range
# and run at the same time. Overlapping ranges would be safe anyway, because a
# shard whose acts.npz exists is skipped.
#
# Forty eval workers, not fifty: the cpu QOS allows 50 SUBMITTED jobs per user
# and Slurm counts array tasks individually, so 50 workers plus normalize plus
# calculate_f1 would be 52 and the array would be rejected.

set -euo pipefail

CONVERT_JOB="${1:-}"
TARGET=baselineuniref345
SCALE=normalized

dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || echo ""; }

# The RERUN_* variables are set in the environment rather than inside --export,
# because a shard range holds a space and --export splits its value on commas.
# --export=ALL carries the whole submit-time environment, which covers them.
export RERUN_TARGET="${TARGET}"

enc_a=$(RERUN_SHARD_RANGE="0 103" sbatch --parsable $(dep "${CONVERT_JOB}") \
  --time=3:00:00 -J bl345_encode_a --export=ALL scripts/submit_encode.sh)
echo "encode shards 0-103   : ${enc_a}"

enc_b=$(RERUN_SHARD_RANGE="104 207" sbatch --parsable $(dep "${CONVERT_JOB}") \
  --time=3:00:00 -J bl345_encode_b --export=ALL scripts/submit_encode.sh)
echo "encode shards 104-207 : ${enc_b}"

norm=$(sbatch --parsable --dependency=afterok:${enc_a}:${enc_b} -J bl345_normalize \
  --export=ALL scripts/submit_normalize_store.sh)
echo "normalize             : ${norm}"

export RERUN_SCALE="${SCALE}"

evl=$(sbatch --parsable --dependency=afterok:${norm} --array=0-39 -J bl345_eval \
  --export=ALL scripts/submit_eval_store.sh)
echo "eval array (40)       : ${evl}"

f1=$(sbatch --parsable --dependency=afterok:${evl} -J bl345_f1 \
  --export=ALL scripts/submit_calculate_f1.sh)
echo "calculate_f1          : ${f1}"

echo
echo "Chain submitted. Read the answer off:"
echo "  data/crosscoder_eval/baseline_uniref_on345/${SCALE}/uniprotkb_modern_score345/test_counts/heldout_top_pairings.csv"
echo "Compare it to the full-UniRef50 model's 0.479 F1 / 187 concepts / 1020 features."
