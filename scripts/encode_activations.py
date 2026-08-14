#!/usr/bin/env python
"""Stage 1 (streaming): ProtT5 -> crosscoder -> sparse activation store.

Replaces `embed_annotations.py` for the crosscoder eval. Instead of writing the
24-layer ProtT5 residual stream to disk (96 KiB/residue, ~5.6 TiB for the
score-{3,4,5} eval set), each ProtT5 batch is encoded through the crosscoder
immediately and only the sparse latents are kept (~256 B/residue at the measured
mean L0 of ~32 of 8192). The embeddings never leave GPU/host memory.

Downstream, `compare_activations` and `collect_feature_activations` read this
store instead of embeddings, which also removes the ~87 redundant full-dictionary
encodes per shard those stages currently do (their feature-chunk loops call
`encode_feat_subset`, which encodes all latents and then slices).

Activations are stored RAW (un-rescaled) -- see interplm/analysis/activation_store.py
for why. Normalization is a divide at read time.

Batching note: sequences are fed in shard order in fixed-size batches with
`padding="longest"` within each batch, exactly as `embed_annotations.py` does, so
a given `--batch_size` reproduces that path's tokenization and padding.

Reads:
    <metadata_dir>/shard_<i>/protein_data.tsv   (Entry, Sequence, Length)
Writes:
    <output_dir>/shard_<i>/{acts.npz, meta.json}

Example:
    python scripts/encode_activations.py \
        --sae_dir  .../jumprelu_global_2519836 \
        --metadata_dir .../processed_annotations \
        --output_dir  .../crosscoder_activations \
        --shard_range 0 207 --batch_size 32
"""

# NOTE: no `from __future__ import annotations` here. tapify resolves each
# annotation with get_type_hints against a throwaway SimpleNamespace, so under
# PEP 563 the string "Path" has no module globals to resolve against and the
# CLI dies with NameError before it parses a single argument. Every other
# tapify script in this repo omits it for the same reason.
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from interplm.analysis.activation_store import ShardActivationWriter, shard_dir
from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
from interplm.sae.inference import load_sae
from interplm.utils import get_device

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def encode_activations(
    sae_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    shards: Optional[List[int]] = None,
    shard_range: Optional[List[int]] = None,
    checkpoint: str = "ae.pt",
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 32,
    dtype: str = "float32",
    overwrite: bool = False,
):
    """
    Args:
        sae_dir: crosscoder directory (config.yaml + the checkpoint).
        metadata_dir: processed_annotations dir holding shard_<i>/protein_data.tsv.
        output_dir: where the sparse activation store is written.
        shards: explicit shard indices. Use shard_range for a contiguous span.
        shard_range: [start, end] inclusive.
        checkpoint: weights file inside sae_dir. Defaults to the un-normalized
            `ae.pt`, because the store holds raw activations and the normalize
            stage runs *after* this one (it reads the store, not embeddings).
        batch_size: ProtT5 sequences per forward pass.
        dtype: ProtT5 precision. float32 matches the crosscoder's training
            harvest and every existing eval artifact; see insights.md 2026-07-04.
        overwrite: re-encode shards that already have a complete acts.npz.
    """
    if shards is not None and shard_range is not None:
        raise ValueError("Cannot specify both shards and shard_range")
    if shard_range is not None:
        shards = list(range(shard_range[0], shard_range[1] + 1))
    if shards is None:
        raise ValueError("Specify --shards or --shard_range")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Device      : {device}")
    print(f"ProtT5 dtype: {dtype}")
    print(f"Crosscoder  : {sae_dir}/{checkpoint}")
    print(f"Batch size  : {batch_size}")
    print(f"Shards      : {len(shards)} ({shards[0]}..{shards[-1]})")

    sae = load_sae(model_dir=sae_dir, model_name=checkpoint, device=device)
    embedder = ProtT5CrosscoderEmbedder(
        model_name=model_name, device=device, dtype=DTYPES[dtype]
    )

    provenance = {
        "sae_dir": str(sae_dir),
        "checkpoint": checkpoint,
        "plm": model_name,
        "plm_dtype": dtype,
        "batch_size": batch_size,
    }

    for shard in shards:
        out = shard_dir(output_dir, shard)
        if (out / "acts.npz").exists() and not overwrite:
            print(f"shard {shard}: already encoded, skipping")
            continue

        df = pd.read_csv(metadata_dir / f"shard_{shard}" / "protein_data.tsv", sep="\t")
        sequences = df["Sequence"].tolist()
        entries = df["Entry"].tolist()

        writer = ShardActivationWriter(n_latents=sae.dict_size)
        for entry, seq in zip(entries, sequences):
            writer.add_protein(entry, len(seq))

        t0 = time.time()
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            # Same call embed_annotations.py makes, one batch at a time, so the
            # padding and tokenization match that path exactly.
            embds = embedder.extract_embeddings(batch, batch_size=batch_size)
            with torch.no_grad():
                latents = sae.encode(embds.to(device), normalize_features=False)
            writer.add(latents)
            del embds, latents

        meta = writer.finalize(out, provenance)
        elapsed = time.time() - t0
        print(
            f"shard {shard}: {len(sequences)} proteins, {meta['n_residues']:,} residues, "
            f"mean L0 {meta['mean_l0']:.2f}, nnz {meta['nnz']:,}, {elapsed:.1f}s"
        )

    print(f"\nDone. Activation store at {output_dir}")


if __name__ == "__main__":
    from tap import tapify

    tapify(encode_activations)
