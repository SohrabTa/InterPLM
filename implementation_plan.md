# Implementation Plan - Sparse AUPRC Evaluation for Crosscoder

This document outlines the plan to refactor the InterPLM evaluation pipeline to calculate AUPRC (Average Precision) by exploiting the sparsity of JumpReLU activations ($L_0 < 30$).

## User Review Required

> [!IMPORTANT]
> **Dashboard Compatibility**: To ensure compatibility with existing InterPLM dashboard scripts, the output CSV will map AUPRC values to the `f1` and `f1_per_domain` columns. We will set `threshold_pct` to a sentinel value (e.g., `-1`) to indicate these are AUPRC-based scores.

> [!TIP]
> **MacBook Pro M1 (Apple Silicon) Setup**:
> - Use `torch.device("mps")` for GPU acceleration on Apple Silicon, but fall back to `"cpu"` for operations not yet supported on Metal.
> - Use `uv run` for environment management.
> - Monitor RAM usage as laptop memory is more constrained than the LRZ nodes.

## Proposed Changes

### InterPLM Analysis Refactor

#### [NEW] [calculate_auprc.py](file:///home/sohrab/code/interp_repos/InterPLM/interplm/analysis/concepts/calculate_auprc.py)
Create a new script to perform the sparse-aware AUPRC calculation and format the results for dashboard compatibility.
- **Phase 1: Sparse Activation Collection**
    - Load the Crosscoder via `CrosscoderDictionaryWrapper`.
    - Iterate through embeddings shards.
    - For each batch, get activations: `feats = model.encode(batch)`.
    - Extract non-zero values and store them in a feature-indexed dictionary: `feature_to_token_hits[feature_id]`.
- **Phase 2: Sparse Label Collection**
    - Load Swiss-Prot ground-truth labels from `eval_set` `npz` files.
    - Store mapping: `concept_to_token_hits[concept_id]`.
- **Phase 3: Vectorized AUPRC Computation**
    - For each (feature, concept) pair:
        - Construct a temporary sparse list of scores and binary labels.
        - Compute AUPRC using a `sparse_average_precision` utility.
    - Parallelize this over concepts/features using `multiprocessing`.
- **Phase 4: Dashboard Formatting**
    - Construct a `pd.DataFrame` with columns: `concept`, `feature`, `threshold_pct`, `precision`, `recall`, `f1`, `f1_per_domain`.
    - Map AUPRC to `f1` and `f1_per_domain`.

### Crosscoder Integration

#### [MODIFY] [crosscoder_dictionary.py](file:///home/sohrab/code/interp_repos/crosscode/crosscode/interplm_adapter/crosscoder_dictionary.py)
Ensure the `encode` method correctly returns non-zero activations for the `calculate_auprc.py` script.

## Verification Plan

### Automated Tests
- **Sparse AUPRC Accuracy**: Verify `sparse_average_precision` against `sklearn.metrics.average_precision_score`.
- **Memory Profiling**: Verify that the hit-tracking lists stay within reasonable bounds.
- **Local End-to-End Test**: Run the script with mock data on the M1 MacBook.

### Manual Verification
- Deploy to LRZ cluster after local verification.
- Verify that `concept_f1_scores.csv` loads correctly in the InterPLM dashboard.
- Ensure the results accurately correlate Crosscoder features with Swiss-Prot biological concepts.
