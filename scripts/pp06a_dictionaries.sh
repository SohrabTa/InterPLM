# The dictionaries of roadmap PP-06a and their paths, in one place for the four eval stages.
# Sourced by submit_encode.sh, submit_normalize_store.sh, submit_eval_store.sh and
# submit_calculate_f1.sh for RERUN_TARGET=pp06a. The paths are container paths.
#
# cc_chunk0      the full-UniRef50 crosscoder at step PP06A_STEP of chunk 0, the exact pair of the SAEs
# sae<p>_l<L>    the per-layer SAEs of crosscode/training_config_layerwise_sae.yaml (p = 0..5)
#
# Each dictionary has a JumpReLU dir jumprelu_global_${PP06A_STEP} next to its BatchTopK checkpoint
# (convert_batchtopk_to_jumprelu.py, with --layer for an SAE). Exp 12 has the commands.
#
# PP06A_STEP is the pair step. The SAE job 5848994 reaches its time limit before step 2,199,187
# (the end of chunk 0), and it cannot resume. Thus the pair is the last save before the limit,
# step 1,889,919, which the crosscoder also saved (exp 12, 2026-10-15).

PP06A_DICTS="cc_chunk0 sae0_l7 sae1_l13 sae2_l16 sae3_l16 sae4_l22 sae5_l24"
PP06A_SAE_RUN="/workspace/model_checkpoints/sae_l8192_k32_bs512_layerwise_uniref_chunk0"
PP06A_STEP=1889919

# The JumpReLU checkpoint dir of one dictionary.
pp06a_ckpt() {
  case "$1" in
    cc_chunk0) echo "/workspace/model_checkpoints/crosscoder_l8192_k32_bs512_full_uniref_chunk0/jumprelu_global_${PP06A_STEP}" ;;
    sae[0-5]_l[0-9]*)
      local p="${1%%_*}" layer="${1##*_l}"
      echo "${PP06A_SAE_RUN}/${p}_encoder.blocks.${layer}.hook_resid_post/jumprelu_global_${PP06A_STEP}" ;;
    *) echo "Unknown PP-06a dictionary '$1'. Use one of: ${PP06A_DICTS}" >&2; return 2 ;;
  esac
}

# The store name of one dictionary, under /workspace/data/crosscoder_activations/.
pp06a_store() { echo "uniprotkb_modern_score345_pp06a/$1"; }

# The RUN_TAG of one dictionary, under /workspace/data/crosscoder_eval/.
pp06a_run_tag() { echo "pp06a/$1"; }

# Stop unless RERUN_DICT names a PP-06a dictionary.
pp06a_check_dict() {
  case " ${PP06A_DICTS} " in
    *" ${RERUN_DICT:-} "*) ;;
    *) echo "RERUN_TARGET=pp06a needs RERUN_DICT in: ${PP06A_DICTS}, got '${RERUN_DICT:-}'." >&2; exit 2 ;;
  esac
}
