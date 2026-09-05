#!/usr/bin/env python3
"""Cut the DisProt proteins out of the score-345 activation store into one small file.

Why this step exists
--------------------
The sweep needs crosscoder activations for 1,552 proteins (435,719 residues). Those rows already
exist in `data/crosscoder_activations/uniprotkb_modern_score345/`, which is 208 shards and 8.8 GB and
lives only on the cluster. The DisProt rows are 0.7% of it. So we read the store once on the cluster,
keep the rows we want, and copy a file of about 120 MB to the Mac. Every later step then runs locally
in seconds, and no ProtT5 run and no re-encode is needed.

The store keys its rows by the accession in the processed eval shards, which disagrees with the
DisProt accession for 145 proteins. This script does not resolve that. It reads
`disprot_store_protein_map.tsv`, which `disprot_build_protein_map.py` already resolved by sequence.

Provenance is checked, not assumed. Three separate Slurm jobs wrote this store (`5750408`, `5750409`,
`5750567`), so the script makes sure that every shard it reads reports the same crosscoder, the same
checkpoint and the same scale. A mixed store would silently blend two activation scales.

Inputs:
  - <acts_dir>/shard_<i>/{acts.npz, meta.json}   the activation store (208 shards)
  - <map>                                        disprot_store_protein_map.tsv
Outputs, in <out_dir>:
  - disprot_acts.npz    CSR [n_residues, n_latents] float32, raw scale, as the store holds it
  - disprot_index.tsv   one row per protein: disprot_acc, store_acc, shard, split, start, end
  - disprot_meta.json   provenance, residue and protein counts, and the source store path
Deterministic; no RNG, no seed needed. Proteins appear in the order of the map file, so the row
ranges in disprot_index.tsv are stable across runs.

Example (cluster):
    python3 disprot_extract_activations.py \
        --acts_dir /workspace/data/crosscoder_activations/uniprotkb_modern_score345 \
        --map /workspace/data/disprot_store_protein_map.tsv \
        --out_dir /workspace/data/disprot_activations
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

# Fields that must agree across every shard. A disagreement means the store mixes two encodes.
PROVENANCE_FIELDS = ("sae_dir", "checkpoint", "scale", "n_latents")


def read_map(path: Path) -> dict:
    """shard -> list of (disprot_acc, store_acc, split), in map-file order."""
    wanted = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            wanted[int(row["shard"])].append(
                (row["disprot_acc"], row["store_acc"], row["split"])
            )
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--acts_dir", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    wanted = read_map(args.map)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blocks, index, provenance = [], [], None
    n_rows = 0

    for shard in sorted(wanted):
        shard_dir = args.acts_dir / f"shard_{shard}"
        meta = json.loads((shard_dir / "meta.json").read_text())

        current = {field: meta.get(field) for field in PROVENANCE_FIELDS}
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise SystemExit(
                f"shard_{shard} provenance {current} differs from {provenance}. "
                "The store mixes two encodes, so the rows are not comparable."
            )

        position = {acc: i for i, acc in enumerate(meta["protein_ids"])}
        boundaries = meta["boundaries"]

        matrix = None
        for disprot_acc, store_acc, split in wanted[shard]:
            if store_acc not in position:
                raise SystemExit(f"{store_acc} is not in shard_{shard}; the map is out of date")
            if matrix is None:
                matrix = sparse.load_npz(shard_dir / "acts.npz").tocsr()
            start, end = boundaries[position[store_acc]]
            blocks.append(matrix[start:end])
            index.append((disprot_acc, store_acc, shard, split, n_rows, n_rows + (end - start)))
            n_rows += end - start

        print(f"shard_{shard}: {len(wanted[shard])} proteins, {n_rows:,} rows so far", flush=True)

    stacked = sparse.vstack(blocks).tocsr()
    stacked.sort_indices()
    sparse.save_npz(args.out_dir / "disprot_acts.npz", stacked, compressed=True)

    with open(args.out_dir / "disprot_index.tsv", "w") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["disprot_acc", "store_acc", "shard", "split", "start", "end"])
        writer.writerows(index)

    out_meta = {
        "n_proteins": len(index),
        "n_residues": int(stacked.shape[0]),
        "n_latents": int(stacked.shape[1]),
        "nnz": int(stacked.nnz),
        "mean_l0": float(stacked.nnz / stacked.shape[0]),
        "source_store": str(args.acts_dir),
        "shards_read": len(wanted),
        "provenance": provenance,
    }
    (args.out_dir / "disprot_meta.json").write_text(json.dumps(out_meta, indent=2))

    print(
        f"\n{out_meta['n_proteins']:,} proteins, {out_meta['n_residues']:,} residues, "
        f"mean L0 {out_meta['mean_l0']:.2f}, {out_meta['nnz']:,} non-zeros"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
