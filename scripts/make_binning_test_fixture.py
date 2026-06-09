"""Build a small real-ProtT5 embedding fixture for testing the LLM-autointerp
binning (collect_step1.py Phase A) locally on the M1 without the 1.8 TB cluster
shards.

Picks a deterministic set of Swiss-Prot sequences from the 67k eval set's
proteins.tsv, embeds them with the SAME embedder the evals use
(ProtT5CrosscoderEmbedder, all 24 residual streams), and writes a single-shard
fixture in the layout load_shard_embeddings / find_max_examples_per_feat expect:

    <out>/embeddings/shard_0/embeddings.pt   # (N_tokens, 1, 24, 1024) fp16
    <out>/metadata/shard_0/protein_data.tsv  # Entry, Length, Sequence (token-aligned)

The embedder slices each sequence to exactly len(seq) tokens (EOS stripped), so
Length == per-protein token count and the find_max_examples_per_feat alignment
assert holds. Pass the two dirs to collect_step1.py as --embeddings-dir /
--metadata-dir.

Inputs : data/eval_dataset/uniprotkb_modern_score45_67k/proteins.tsv
Outputs: <out>/embeddings/shard_0/embeddings.pt, <out>/metadata/shard_0/protein_data.tsv
Repro  : deterministic — selects the first --n proteins (file order) within the
         length window; no RNG. Run from repos/InterPLM with its venv active.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--proteins-tsv",
        type=Path,
        default=Path(
            "../../data/eval_dataset/uniprotkb_modern_score45_67k/proteins.tsv"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../../data/test_fixtures/prott5_binning"),
    )
    ap.add_argument("--n", type=int, default=40, help="number of proteins to embed")
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.proteins_tsv, sep="\t", usecols=["Entry", "Length", "Sequence"])
    df = df[(df["Length"] >= args.min_len) & (df["Length"] <= args.max_len)]
    df = df.dropna(subset=["Sequence"]).head(args.n).reset_index(drop=True)
    if len(df) < args.n:
        print(f"WARNING: only {len(df)} proteins in [{args.min_len},{args.max_len}] aa")

    entries = df["Entry"].astype(str).tolist()
    seqs = df["Sequence"].astype(str).tolist()
    lengths = [len(s) for s in seqs]  # token count == len(seq) (EOS stripped)
    print(f"Selected {len(seqs)} proteins; total tokens = {sum(lengths)}")

    from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder

    embedder = ProtT5CrosscoderEmbedder()
    emb = embedder.extract_embeddings(seqs, batch_size=args.batch_size)  # (N,1,24,1024)
    assert emb.shape[0] == sum(lengths), (
        f"token count mismatch: emb {emb.shape[0]} vs sum(len) {sum(lengths)}"
    )
    print(f"Embedded tensor: {tuple(emb.shape)} dtype before store {emb.dtype}")

    emb_dir = args.out_dir / "embeddings" / "shard_0"
    meta_dir = args.out_dir / "metadata" / "shard_0"
    emb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Store fp32 to match the cluster analysis_embeddings shards: the crosscoder
    # weights are fp32 and get_sae_feats_in_batches does not cast the input, so a
    # fp16 shard would fail the encode einsum with a Half-vs-float dtype error.
    torch.save(emb.to(torch.float32).contiguous(), emb_dir / "embeddings.pt")
    pd.DataFrame({"Entry": entries, "Length": lengths, "Sequence": seqs}).to_csv(
        meta_dir / "protein_data.tsv", sep="\t", index=False
    )
    mb = (emb.numel() * 4) / 1e6
    print(
        f"Wrote {emb_dir/'embeddings.pt'} (~{mb:.0f} MB fp16) and "
        f"{meta_dir/'protein_data.tsv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
