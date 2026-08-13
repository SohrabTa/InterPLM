"""Sparse per-shard store of crosscoder feature activations.

Why this exists
---------------
The InterPLM analysis stages (normalize, compare_activations, collect, dashboard)
all read stored PLM embeddings and re-encode them through the SAE/crosscoder.
For our ProtT5 crosscoder those embeddings are the residual stream of all 24
encoder layers, i.e. ``[n_residues, 1, 24, 1024]`` float32 = 96 KiB per residue.
The score-{3,4,5} eval set is 62.7 M residues, so storing them costs ~5.6 TiB.

The activations themselves are sparse: the JumpReLU inference gate is a hard
``(preact > threshold) * preact`` (crosscode/models/activations/jumprelu.py), so
sub-threshold latents are *exact* zeros and CSR storage is lossless. At the
measured mean L0 of ~32 of 8192 latents, one residue costs ~256 bytes instead of
98,304 -- roughly 380x less.

So we encode once, store the sparse activations, and let every downstream stage
read them. That also removes the repeated encoding: because
``encode_feat_subset`` computes the full dictionary and then slices columns,
the feature-chunk loops in normalize (13 chunks), compare_activations (33) and
collect (41) currently re-encode the same shard ~87 times.

Scale convention
----------------
Activations are stored **raw** (un-rescaled), exactly as ``encode`` returns them.
That matches how the stages differ today: compare_activations reads raw values
(it never passes ``normalize_features``), while collect divides by the
per-feature ``activation_rescale_factor``. Storing raw and dividing at read time
reproduces both without a second encode pass.

Layout
------
    <acts_dir>/shard_<i>/acts.npz   scipy CSR [n_residues, n_latents], float32
    <acts_dir>/shard_<i>/meta.json  protein_ids, boundaries, provenance

Written atomically (temp file + os.replace) so an interrupted job never leaves a
half-written shard behind (see insights.md 2026-07-19).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import sparse

SCHEMA_VERSION = 1


def shard_dir(acts_dir: Path, shard_idx: int) -> Path:
    return Path(acts_dir) / f"shard_{shard_idx}"


class ShardActivationWriter:
    """Accumulate per-batch activations for one shard, then write it out.

    Each ``add`` takes a dense ``[n_residues_in_batch, n_latents]`` tensor (one
    ProtT5 batch worth of residues) and immediately converts it to CSR, so the
    dense block is freed straight away and peak memory stays bounded by the
    batch, not the shard.
    """

    def __init__(self, n_latents: int):
        self.n_latents = n_latents
        self._blocks: list[sparse.csr_matrix] = []
        self._protein_ids: list[str] = []
        self._boundaries: list[tuple[int, int]] = []
        self._n_residues = 0

    def add(self, latents: torch.Tensor) -> None:
        arr = latents.detach().to("cpu", torch.float32).numpy()
        if arr.shape[1] != self.n_latents:
            raise ValueError(
                f"expected {self.n_latents} latents, got {arr.shape[1]}"
            )
        self._blocks.append(sparse.csr_matrix(arr))
        self._n_residues += arr.shape[0]

    def add_protein(self, protein_id: str, length: int) -> None:
        start = (
            self._boundaries[-1][1] if self._boundaries else 0
        )
        self._boundaries.append((start, start + length))
        self._protein_ids.append(str(protein_id))

    def finalize(self, out_dir: Path, provenance: dict) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        acts = (
            sparse.vstack(self._blocks, format="csr")
            if self._blocks
            else sparse.csr_matrix((0, self.n_latents), dtype=np.float32)
        )
        if acts.shape[0] != self._n_residues:
            raise RuntimeError(
                f"row count mismatch: {acts.shape[0]} vs {self._n_residues}"
            )
        expected = self._boundaries[-1][1] if self._boundaries else 0
        if expected != self._n_residues:
            raise RuntimeError(
                f"boundaries cover {expected} residues but stored {self._n_residues}"
            )

        meta = {
            "schema_version": SCHEMA_VERSION,
            "n_residues": int(self._n_residues),
            "n_latents": int(self.n_latents),
            "nnz": int(acts.nnz),
            "mean_l0": float(acts.nnz / self._n_residues) if self._n_residues else 0.0,
            "protein_ids": self._protein_ids,
            "boundaries": [[int(a), int(b)] for a, b in self._boundaries],
            "scale": "raw",
            **provenance,
        }

        _atomic_write(out_dir / "acts.npz", lambda p: sparse.save_npz(p, acts))
        _atomic_write(
            out_dir / "meta.json",
            lambda p: Path(p).write_text(json.dumps(meta) + "\n"),
        )
        return meta


def _atomic_write(path: Path, write_fn) -> None:
    """Write via a temp file in the same directory, then rename into place."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # scipy appends .npz if the name lacks it, so hand it a matching suffix.
    if path.suffix == ".npz":
        tmp = path.with_name(path.name + ".tmp.npz")
    write_fn(tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_shard_activations(
    acts_dir: Path, shard_idx: int
) -> tuple[sparse.csr_matrix, dict]:
    """Load one shard's raw activations and metadata."""
    d = shard_dir(acts_dir, shard_idx)
    acts_path = d / "acts.npz"
    meta_path = d / "meta.json"
    if not acts_path.exists():
        raise FileNotFoundError(f"No activation shard at {acts_path}")
    acts = sparse.load_npz(acts_path).tocsr()
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{meta_path}: schema_version {meta.get('schema_version')} "
            f"!= {SCHEMA_VERSION}"
        )
    if acts.shape[0] != meta["n_residues"]:
        raise ValueError(
            f"{acts_path}: {acts.shape[0]} rows but meta says {meta['n_residues']}"
        )
    return acts, meta


def available_shards(acts_dir: Path) -> list[int]:
    acts_dir = Path(acts_dir)
    out = []
    for d in acts_dir.glob("shard_*"):
        if (d / "acts.npz").exists():
            try:
                out.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def protein_ids_per_residue(meta: dict) -> list[str]:
    """Expand the per-protein boundaries into one protein id per residue."""
    ids: list[str] = []
    for pid, (start, end) in zip(meta["protein_ids"], meta["boundaries"]):
        ids.extend([pid] * (end - start))
    return ids


def feature_subset(
    acts: sparse.csr_matrix,
    feat_list: Iterable[int] | None = None,
    rescale: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Slice feature columns, optionally dividing by the per-feature rescale.

    ``rescale`` is the full-length ``activation_rescale_factor``; it is clamped
    the same way CrosscoderDictionaryWrapper clamps it, so dead features (max 0)
    give 0 rather than NaN.
    """
    if feat_list is None:
        out = acts
        cols = None
    else:
        cols = np.asarray(list(feat_list), dtype=np.int64)
        out = acts[:, cols]
    if rescale is not None:
        div = np.asarray(rescale, dtype=np.float32)
        if cols is not None:
            div = div[cols]
        div = np.clip(div, 1e-12, None)
        out = out.multiply(sparse.csr_matrix(1.0 / div))
        out = sparse.csr_matrix(out)
    return out
