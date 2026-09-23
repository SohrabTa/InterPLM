#!/usr/bin/env python
"""Stage 1 for InterPLM's released ESM-2-650M SAEs: ESM-2 -> six SAEs -> six sparse stores.

Roadmap PP-10a. Our eval set and our eval, applied to InterPLM's own dictionaries, so that the
drop from domain-level to residue-level pairing can be put beside ours
(documentation/experiments/02-interplm-eval-pipeline.md, section PP-10).

One ESM-2 forward pass per batch gives all hidden states. Each requested layer goes through its
own SAE, and each layer gets its own activation store, in the same format as
scripts/encode_activations.py writes for the crosscoder. The downstream stages (normalize, eval,
calculate_f1) then run on each store with no change.

What is the same as InterPLM's own code:
  - The model and tokenizer load with the calls of ESM.load_model (interplm/embedders/esm.py),
    and the tokenizer call is the one in ESM.extract_embeddings_multiple_layers: padding=True,
    truncation=True, max_length=1024.
  - Layer L is hidden_states[L] of the HF EsmModel output, with the CLS and EOS positions removed,
    as in that same function.
  - The SAE input goes through _normalize_input_and_get_norms before encode, as
    ReLUSAE.encode_feat_subset does (interplm/sae/dictionary.py). That is the path InterPLM's eval
    uses. For these legacy SAEs normalize_to_sqrt_d is recorded in each shard's meta.json.

What is added:
  - The HF revisions are fixed and recorded, so the weights cannot change under a re-run.
  - Each protein must give exactly len(sequence) + 2 tokens. If not, the script stops. The ESM
    embedder slices by string length and would misalign residues and labels without a message.
  - The nonzero entries are found on the GPU and only those move to the CPU
    (ShardActivationWriter.add_csr). The stored values are the same as a dense copy gives.
  - Each shard records the SAE's fraction of variance unexplained (fvu) on that shard. A wrong
    layer index or input scale shows there as a high value.

Activations are stored RAW (ae_unnormalized.pt). The normalize stage then computes the per-feature
maximum on this eval set, exactly as it does for the crosscoder, so the thresholds mean the same
thing for the two dictionaries.

Reads:
    <metadata_dir>/shard_<i>/protein_data.tsv   (Entry, Sequence, Length)
    HF facebook/esm2_t33_650M_UR50D @ ESM_REVISION
    HF Elana/InterPLM-esm2-650m @ SAE_REVISION, layer_<L>/ae_unnormalized.pt, or the same file
    copied to <sae_root>/layer_<L>/ae.pt (scripts/prepare_interplm_esm_saes.py), which is what the
    normalize stage reads too
Writes:
    <output_root>/layer_<L>/shard_<i>/{acts.npz, meta.json}

Example:
    python scripts/encode_activations_esm.py \
        --metadata_dir .../uniprotkb_modern_score345/processed_annotations \
        --output_root  .../crosscoder_activations/uniprotkb_modern_score345_interplm_esm2_650m \
        --shard_range 0 207 --batch_size 64
"""

# NOTE: no `from __future__ import annotations` here, for the tapify reason recorded in
# scripts/encode_activations.py.
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy import sparse
from transformers import AutoTokenizer, EsmModel

from interplm.analysis.activation_store import ShardActivationWriter, shard_dir
from interplm.sae.dictionary import ReLUSAE
from interplm.utils import get_device

ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
SAE_REPO = "Elana/InterPLM-esm2-650m"
SAE_REVISION = "5121c4c7f3ad0b5fbe0f3b9a457969192bb9912f"
SAE_LAYERS = [1, 9, 18, 24, 30, 33]  # interplm/sae/inference.py INTERPLM_HF_LAYERS


def to_csr(latents: torch.Tensor) -> sparse.csr_matrix:
    """Dense [n, F] tensor on any device -> scipy CSR on the CPU, moving only the nonzeros."""
    rows, cols = torch.nonzero(latents, as_tuple=True)
    values = latents[rows, cols]
    n, n_features = latents.shape
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(torch.bincount(rows, minlength=n).cpu().numpy(), out=indptr[1:])
    return sparse.csr_matrix(
        (values.cpu().numpy().astype(np.float32), cols.cpu().numpy(), indptr),
        shape=(n, n_features),
    )


def encode_activations_esm(
    metadata_dir: Path,
    output_root: Path,
    shards: Optional[List[int]] = None,
    shard_range: Optional[List[int]] = None,
    layers: List[int] = SAE_LAYERS,
    sae_root: Optional[Path] = None,
    batch_size: int = 64,
    max_proteins: Optional[int] = None,
    overwrite: bool = False,
):
    """
    Args:
        metadata_dir: processed_annotations dir holding shard_<i>/protein_data.tsv.
        output_root: one store per layer is written to <output_root>/layer_<L>.
        shards: explicit shard indices. Use shard_range for a contiguous span.
        shard_range: [start, end] inclusive.
        layers: ESM-2 layers to encode. Each must have a released SAE.
        sae_root: read <sae_root>/layer_<L>/ae.pt instead of the HF download. The files must be
            the SAE_REVISION ae_unnormalized.pt; each shard records the md5 it read.
        batch_size: sequences per ESM-2 forward pass.
        max_proteins: encode only the first N proteins of each shard. For probes only: the eval
            refuses a store whose row count differs from the labels, so never point this at a
            real store.
        overwrite: re-encode a layer's shard that already has acts.npz.
    """
    if shards is not None and shard_range is not None:
        raise ValueError("Cannot specify both shards and shard_range")
    if shard_range is not None:
        shards = list(range(shard_range[0], shard_range[1] + 1))
    if shards is None:
        raise ValueError("Specify --shards or --shard_range")
    for layer in layers:
        if layer not in SAE_LAYERS:
            raise ValueError(f"No released SAE for layer {layer}; options {SAE_LAYERS}")

    output_root = Path(output_root)
    device = get_device()
    print(f"Device     : {device}")
    print(f"ESM-2      : {ESM_MODEL} @ {ESM_REVISION}")
    print(f"SAEs       : {SAE_REPO} @ {SAE_REVISION}, layers {layers}")
    print(f"Batch size : {batch_size}")
    print(f"Shards     : {len(shards)} ({shards[0]}..{shards[-1]})")
    if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
        print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL, revision=ESM_REVISION, clean_up_tokenization_spaces=True
    )
    model = EsmModel.from_pretrained(
        ESM_MODEL, revision=ESM_REVISION, add_pooling_layer=False
    ).to(device)
    model.eval()

    saes, sae_files, sae_md5 = {}, {}, {}
    for layer in layers:
        if sae_root is not None:
            path = Path(sae_root) / f"layer_{layer}" / "ae.pt"
        else:
            path = hf_hub_download(
                repo_id=SAE_REPO,
                filename=f"layer_{layer}/ae_unnormalized.pt",
                revision=SAE_REVISION,
            )
        sae = ReLUSAE.from_pretrained(path, device=device)
        sae.eval()
        saes[layer] = sae
        sae_files[layer] = str(path)
        sae_md5[layer] = hashlib.md5(Path(path).read_bytes()).hexdigest()
        print(
            f"layer {layer:2d}: dict_size {sae.dict_size}, activation_dim {sae.activation_dim}, "
            f"normalize_to_sqrt_d {bool(sae.normalize_to_sqrt_d)}, md5 {sae_md5[layer]}"
        )

    for shard in shards:
        todo = [
            layer
            for layer in layers
            if overwrite or not (shard_dir(output_root / f"layer_{layer}", shard) / "acts.npz").exists()
        ]
        if not todo:
            print(f"shard {shard}: all layers encoded, skipping")
            continue

        df = pd.read_csv(metadata_dir / f"shard_{shard}" / "protein_data.tsv", sep="\t")
        if max_proteins is not None:
            df = df.iloc[:max_proteins]
        sequences = df["Sequence"].tolist()
        entries = df["Entry"].tolist()

        writers = {layer: ShardActivationWriter(n_latents=saes[layer].dict_size) for layer in todo}
        for writer in writers.values():
            for entry, seq in zip(entries, sequences):
                writer.add_protein(entry, len(seq))
        # float64 sums for the fraction of variance unexplained, per layer
        sq_err = {layer: 0.0 for layer in todo}
        sum_x = {layer: None for layer in todo}
        sum_x2 = {layer: 0.0 for layer in todo}
        n_rows = 0

        t0 = time.time()
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
            )
            lengths = torch.tensor([len(s) for s in batch])
            n_tokens = inputs["attention_mask"].sum(dim=1)
            if not torch.equal(n_tokens, lengths + 2):
                bad = [(entries[i + j], int(lengths[j]), int(n_tokens[j])) for j in range(len(batch)) if n_tokens[j] != lengths[j] + 2]
                raise RuntimeError(f"shard {shard}: token count is not length + 2 for {bad[:5]}")
            # Residue positions: 1 .. len (drop CLS at 0, EOS at len + 1, and padding)
            positions = torch.arange(inputs["input_ids"].shape[1])[None, :]
            residue_mask = (positions >= 1) & (positions <= lengths[:, None])

            inputs = {k: v.to(device) for k, v in inputs.items()}
            residue_mask = residue_mask.to(device)
            with torch.no_grad():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
                for layer in todo:
                    x = hidden_states[layer][residue_mask]  # [n_residues, 1280], sequence order
                    x_in, _ = saes[layer]._normalize_input_and_get_norms(x)
                    latents = saes[layer].encode(x_in)
                    x_hat = saes[layer].decode(latents)
                    sq_err[layer] += float(((x_in - x_hat) ** 2).sum(dtype=torch.float64))
                    s = x_in.sum(dim=0, dtype=torch.float64)
                    sum_x[layer] = s if sum_x[layer] is None else sum_x[layer] + s
                    sum_x2[layer] += float((x_in.double() ** 2).sum())
                    writers[layer].add_csr(to_csr(latents))
                    del x, x_in, latents, x_hat
                n_rows += int(residue_mask.sum())
            del hidden_states, inputs

        elapsed = time.time() - t0
        provenances = {}
        for layer in todo:
            total_var = sum_x2[layer] - float((sum_x[layer] ** 2).sum()) / n_rows
            provenance = {
                "plm": ESM_MODEL,
                "plm_revision": ESM_REVISION,
                "plm_dtype": "float32",
                "layer": layer,
                "sae_repo": SAE_REPO,
                "sae_revision": SAE_REVISION,
                "checkpoint": f"layer_{layer}/ae_unnormalized.pt",
                "sae_file": sae_files[layer],
                "sae_md5": sae_md5[layer],
                "normalize_to_sqrt_d": bool(saes[layer].normalize_to_sqrt_d),
                "batch_size": batch_size,
                "fvu": sq_err[layer] / total_var,
            }
            provenances[layer] = provenance

        # The compressed save costs about 95 s per shard for the six layers on one core: shard 0
        # has 257 M nonzeros, and the M1 saved 3.75 M in 1.38 s (2026-09-23). zlib releases the GIL while it
        # compresses, so one thread per layer runs the saves at the same time.
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            futures = {
                layer: pool.submit(
                    writers[layer].finalize,
                    shard_dir(output_root / f"layer_{layer}", shard),
                    provenances[layer],
                )
                for layer in todo
            }
            metas = {layer: f.result() for layer, f in futures.items()}
        for layer in todo:
            meta = metas[layer]
            print(
                f"shard {shard} layer {layer:2d}: {len(sequences)} proteins, "
                f"{meta['n_residues']:,} residues, mean L0 {meta['mean_l0']:.1f}, "
                f"nnz {meta['nnz']:,}, fvu {provenances[layer]['fvu']:.4f}"
            )
        del writers
        print(
            f"shard {shard}: {elapsed:.1f}s encode, {time.time() - t1:.1f}s save, "
            f"{n_rows:,} residues",
            flush=True,
        )

    print(f"\nDone. Activation stores under {output_root}")


if __name__ == "__main__":
    from tap import tapify

    tapify(encode_activations_esm)
