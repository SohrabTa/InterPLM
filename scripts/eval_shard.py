#!/usr/bin/env python
"""Run the concept-activation comparison for ONE shard, off the activation store.

Why a per-shard entry point exists
----------------------------------
`run_eval_pipeline.py` walks every shard in one process. That took 25 h 16 m for
the 84-shard {4,5} set (job 5673420) and would take ~100 h for the 208-shard
{3,4,5} set, over the 48 h LRZ cap. The shards are independent and
`calculate_f1.combine_metrics_across_shards` sums the per-shard counts, so the
fix is to split the job, not to change the algorithm (roadmap PP-02e).

Reading the activation store makes this cheap to schedule: no SAE is loaded and
no GPU is needed, so these are CPU array tasks.

The shard's split (valid or test) is looked up rather than passed, so a Slurm
array index maps straight to a shard number with no off-by-one bookkeeping.

Example (one array task):
    python scripts/eval_shard.py \
        --sae_dir <ckpt>/jumprelu_global_10990182 \
        --acts_dir /workspace/data/crosscoder_activations/<evalset> \
        --eval_data_root /workspace/data/eval_dataset/<evalset>/processed_annotations \
        --output_root /workspace/data/crosscoder_eval/<run>/<evalset> \
        --shard 17 --normalize_features
"""

# NOTE: no `from __future__ import annotations` here. tapify resolves each
# annotation with get_type_hints against a throwaway SimpleNamespace, so under
# PEP 563 the string "Path" has no module globals to resolve against and the
# CLI dies with NameError before it parses a single argument. Every other
# tapify script in this repo omits it for the same reason.
import json
import sys
from pathlib import Path
from typing import List, Optional

from interplm.analysis.concepts.compare_activations import analyze_concepts


def find_split(eval_data_root: Path, shard: int) -> str:
    """Return the split ('valid' or 'test') that owns this shard."""
    for split in ("valid", "test"):
        meta_path = eval_data_root / split / "metadata.json"
        if not meta_path.exists():
            continue
        if shard in json.loads(meta_path.read_text())["shard_source"]:
            return split
    raise ValueError(f"Shard {shard} is in neither the valid nor the test split")


def eval_shard(
    sae_dir: Path,
    eval_data_root: Path,
    output_root: Path,
    shard: int,
    acts_dir: Optional[Path] = None,
    aa_embds_dir: Optional[Path] = None,
    normalize_features: bool = False,
    thresholds: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
    is_sparse: bool = True,
    skip_existing: bool = True,
):
    """
    Args:
        sae_dir: crosscoder directory. With acts_dir it is read only for
            feature_stats/max.npy, and only when normalize_features is set.
        eval_data_root: processed_annotations dir holding valid/ and test/.
        output_root: results root; counts land in <output_root>/<split>_counts/.
        shard: shard index to process.
        acts_dir: sparse activation store (scripts/encode_activations.py).
        aa_embds_dir: legacy embedding dir, used only when acts_dir is unset.
        normalize_features: divide by the per-feature rescale factor before
            thresholding. Correct, and what every eval from 2026-08-13 uses. Runs
            before that date left this off and so compared on the raw scale,
            which made all five thresholds identical.
        skip_existing: leave a shard alone if its counts file is already there,
            so a re-submitted array does not redo finished work.
    """
    if acts_dir is None and aa_embds_dir is None:
        raise ValueError("Specify --acts_dir (preferred) or --aa_embds_dir")

    split = find_split(eval_data_root, shard)
    counts_dir = output_root / f"{split}_counts"
    out_file = counts_dir / f"shard_{shard}_counts.npz"

    if skip_existing and out_file.exists():
        print(f"shard {shard} ({split}): {out_file} exists, skipping")
        return

    scale = "normalized" if normalize_features else "raw"
    print(f"shard {shard} -> split '{split}', scale '{scale}', out {counts_dir}")

    analyze_concepts(
        sae_dir=sae_dir,
        aa_embds_dir=aa_embds_dir if aa_embds_dir is not None else Path("."),
        eval_set_dir=eval_data_root / split,
        output_dir=counts_dir,
        threshold_percents=thresholds,
        shard=shard,
        is_sparse=is_sparse,
        acts_dir=acts_dir,
        normalize_features=normalize_features,
    )
    print(f"shard {shard} done -> {out_file}")


if __name__ == "__main__":
    from tap import tapify

    sys.exit(tapify(eval_shard))
