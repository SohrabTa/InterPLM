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

Several dictionaries from one ProtT5 pass (roadmap PP-06a): pass --extra_sae_dirs and
--extra_output_dirs. Each dictionary writes its own store. A dictionary whose config.yaml has
`layer: L` (a one-layer SAE, written by convert_batchtopk_to_jumprelu.py --layer) reads only the
hookpoint encoder.blocks.L, that is index L-1 of the 24 layers. A dictionary without `layer` reads
all 24 layers, as before. The checkpoints have the training normalization folded into their weights,
so each dictionary reads the raw ProtT5 activations. Without the extra options, the script does
what it did before.

Reads:
    <metadata_dir>/shard_<i>/protein_data.tsv   (Entry, Sequence, Length)
Writes:
    <output_dir>/shard_<i>/{acts.npz, meta.json}, and the same for each extra output dir

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
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import yaml

from interplm.analysis.activation_store import ShardActivationWriter, shard_dir
from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
from interplm.sae.inference import load_sae
from interplm.utils import get_device

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TF32_ENV_VARS = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


def _print_tf32_state(label: str) -> None:
    print(
        f"TF32 {label}: cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
    )


def _disable_tf32() -> None:
    """Make every fp32 matmul and convolution full fp32, for ProtT5 and the crosscoder.

    The NGC PyTorch image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which makes the default
    float32 matmul precision "high" (TF32). Roadmap PP-02i.
    """
    print(f"torch {torch.__version__}")
    for name in TF32_ENV_VARS:
        print(f"env {name}={os.environ.get(name, '<unset>')}")
    _print_tf32_state("before")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _print_tf32_state("after ")


def _load_dictionary(sae_dir: Path, checkpoint: str, device: str, n_layers: int):
    """Load one dictionary and the slice of the 24 ProtT5 layers that it reads.

    Returns (dictionary, layer). layer is None for a dictionary that reads all layers, else the
    1-based hookpoint number from `layer` in its config.yaml.
    """
    config = yaml.safe_load((Path(sae_dir) / "config.yaml").read_text())
    layer = config.get("layer")
    sae = load_sae(model_dir=sae_dir, model_name=checkpoint, device=device)
    n_hookpoints = sae.crosscoder.n_hookpoints
    expected = n_layers if layer is None else 1
    if n_hookpoints != expected:
        raise ValueError(
            f"{sae_dir}: n_hookpoints {n_hookpoints} but config layer={layer} needs {expected}"
        )
    if layer is not None and not 1 <= layer <= n_layers:
        raise ValueError(f"{sae_dir}: layer {layer} is not in 1..{n_layers}")
    return sae, layer


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
    disable_tf32: bool = False,
    extra_sae_dirs: Optional[List[Path]] = None,
    extra_output_dirs: Optional[List[Path]] = None,
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
        disable_tf32: run all fp32 matmuls without TF32, and log the TF32
            settings before and after the change. Off by default, which keeps
            the settings of the container (TF32 on in the NGC image). Write
            such a run to its own output_dir.
        extra_sae_dirs: more dictionaries to encode from the same ProtT5 pass,
            each with the same checkpoint name. See the module docstring for
            the layer slice.
        extra_output_dirs: one store dir for each extra_sae_dirs entry, in order.
    """
    if shards is not None and shard_range is not None:
        raise ValueError("Cannot specify both shards and shard_range")
    if shard_range is not None:
        shards = list(range(shard_range[0], shard_range[1] + 1))
    if shards is None:
        raise ValueError("Specify --shards or --shard_range")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    extra_sae_dirs = extra_sae_dirs or []
    extra_output_dirs = extra_output_dirs or []
    if len(extra_sae_dirs) != len(extra_output_dirs):
        raise ValueError("extra_sae_dirs and extra_output_dirs need the same length")
    sae_dirs = [Path(sae_dir), *map(Path, extra_sae_dirs)]
    output_dirs = [Path(output_dir), *map(Path, extra_output_dirs)]
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("each dictionary needs its own output dir")

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Device      : {device}")
    print(f"ProtT5 dtype: {dtype}")
    for d, o in zip(sae_dirs, output_dirs):
        print(f"Crosscoder  : {d}/{checkpoint} -> {o}")
    print(f"Batch size  : {batch_size}")
    print(f"Shards      : {len(shards)} ({shards[0]}..{shards[-1]})")
    if disable_tf32:
        _disable_tf32()

    embedder = ProtT5CrosscoderEmbedder(
        model_name=model_name, device=device, dtype=DTYPES[dtype]
    )
    dictionaries = []
    for d, o in zip(sae_dirs, output_dirs):
        sae, layer = _load_dictionary(d, checkpoint, device, embedder.num_hookpoints)
        provenance = {
            "sae_dir": str(d),
            "checkpoint": checkpoint,
            "plm": model_name,
            "plm_dtype": dtype,
            "batch_size": batch_size,
        }
        if layer is not None:
            provenance["layer"] = layer
        if disable_tf32:
            provenance["tf32"] = "disabled"
        dictionaries.append((sae, layer, o, provenance))

    for shard in shards:
        pending = [
            entry
            for entry in dictionaries
            if overwrite or not (shard_dir(entry[2], shard) / "acts.npz").exists()
        ]
        if not pending:
            print(f"shard {shard}: already encoded, skipping")
            continue

        df = pd.read_csv(metadata_dir / f"shard_{shard}" / "protein_data.tsv", sep="\t")
        sequences = df["Sequence"].tolist()
        entries = df["Entry"].tolist()

        writers = []
        for sae, _, _, _ in pending:
            writer = ShardActivationWriter(n_latents=sae.dict_size)
            for entry, seq in zip(entries, sequences):
                writer.add_protein(entry, len(seq))
            writers.append(writer)

        t0 = time.time()
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            # Same call embed_annotations.py makes, one batch at a time, so the
            # padding and tokenization match that path exactly.
            embds = embedder.extract_embeddings(batch, batch_size=batch_size).to(device)
            for (sae, layer, _, _), writer in zip(pending, writers):
                x = embds if layer is None else embds[:, :, layer - 1 : layer, :]
                with torch.no_grad():
                    latents = sae.encode(x, normalize_features=False)
                writer.add(latents)
                del latents
            del embds

        elapsed = time.time() - t0
        for (_, _, out_dir, provenance), writer in zip(pending, writers):
            meta = writer.finalize(shard_dir(out_dir, shard), provenance)
            print(
                f"shard {shard} -> {out_dir}: {len(sequences)} proteins, "
                f"{meta['n_residues']:,} residues, mean L0 {meta['mean_l0']:.2f}, "
                f"nnz {meta['nnz']:,}, {elapsed:.1f}s"
            )

    print(f"\nDone. Activation stores at {[str(o) for o in output_dirs]}")


if __name__ == "__main__":
    from tap import tapify

    tapify(encode_activations)
