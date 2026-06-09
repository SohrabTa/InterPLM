"""Step 1 of the LLM annotation pipeline: produce a self-contained parquet
of per-feature LLM inputs.

For every "live" crosscoder feature, this script:

  A. Scans the cached ProtT5 embedding shards through the crosscoder once,
     producing normalized [0,1] activations and assigning each protein to
     one of 11 bins (zero + 10 bins of width 0.1) for that feature.

  B. Applies the InterPLM paper's sampling rule: 2 proteins per non-top bin,
     10 from (0.9, 1.0], top up (0.8, 0.9] to reach 24 in the top two bins
     combined, 10 random zero-activation negatives. Features with fewer than
     20 proteins in the top three bins combined are excluded. The selected
     proteins are split 50/50 into train and eval halves.

  C. Re-scans the shards to extract per-residue activation traces (only
     residues with normalized activation > --trace-threshold) for every
     selected (feature, protein) pair.

  D. Fetches richer Swiss-Prot metadata (name, organism, function, keywords,
     GO terms, features) from the UniProt REST API for every unique selected
     protein. Cached locally so a re-run is cheap.

  E. Writes a single parquet file with one row per (feature, protein) sample,
     plus a side-car JSON with per-feature aggregate stats.

The output is fully self-contained: the Step-2 LLM caller doesn't need ProtT5,
the crosscoder, or the embedding shards.

Designed to run on the H100 cluster against /workspace/data, but every phase
can also be invoked locally with --shards-to-search 0 --feature-ids 1322 for
smoke-testing on a single feature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths assume the script is launched with `crosscode/` and `InterPLM/` on
# sys.path; on the cluster `uv run` from /workspace/crosscode handles that.
# ---------------------------------------------------------------------------

UNIPROT_FIELDS = ",".join(
    [
        "accession",
        "protein_name",
        "organism_name",
        "length",
        "cc_function",
        "cc_subcellular_location",
        "keyword",
        "go_p",
        "go_f",
        "go_c",
        "ft_domain",
        "ft_motif",
        "ft_act_site",
        "ft_binding",
        "ft_region",
    ]
)

# Paper's bin scheme: 10 bins of width 0.1 on the [0,1] normalized scale.
BIN_EDGES: list[tuple[float, float]] = [
    (round(i * 0.1, 1), round((i + 1) * 0.1, 1)) for i in range(10)
]

# Sampling rule constants — exactly the InterPLM paper recipe.
N_PER_NON_TOP_BIN = 2
N_TOP_BIN = 10
N_TOP_TWO_TARGET = 24
N_ZERO_NEGATIVES = 10
MIN_PROTEINS_IN_TOP_THREE = 20


# ---------------------------------------------------------------------------
# Phase A: bin-and-count scan over all shards
# ---------------------------------------------------------------------------


def phase_a_bin_scan(
    sae_dir: Path,
    embeddings_dir: Path,
    metadata_dir: Path,
    shards_to_search: list[int],
    feature_ids: list[int],
    feature_chunk_size: int,
    cache_path: Path,
    n_per_bin_to_sample: int = 30,
    n_zero_to_sample: int = 500,
) -> dict[int, dict[tuple[float, float], list[str]]]:
    """Return {feature_id: {(lo, hi): [protein_ids]}}.

    Wraps the existing find_max_examples_per_feat with 11 bins (zero + 10
    width-0.1 bins). Output is cached to disk so phase A can be skipped on
    re-runs.
    """
    if cache_path.exists():
        print(f"Phase A: loading cached bin assignments from {cache_path}")
        with cache_path.open() as f:
            raw = yaml.unsafe_load(f)
        # YAML can't key dicts on tuples; we serialize bins as "lo,hi" strings.
        return {
            int(fid): {
                tuple(map(float, k.split(","))): list(v)
                for k, v in bins.items()
            }
            for fid, bins in raw.items()
        }

    from interplm.analysis.per_protein_tracking import find_max_examples_per_feat
    from interplm.sae.inference import load_sae

    sae = load_sae(sae_dir, model_name="ae_normalized.pt")
    sae_full = _restrict_to_feature_subset(sae, feature_ids)

    bin_thresholds = [(0.0, 0.0)] + BIN_EDGES  # paper bins + explicit zero bucket
    print(
        f"Phase A: scanning {len(shards_to_search)} shards × "
        f"{len(feature_ids)} features in chunks of {feature_chunk_size}..."
    )
    t0 = time.time()
    results = find_max_examples_per_feat(
        sae=sae_full,
        aa_embeds_dir=embeddings_dir,
        aa_metadata_dir=metadata_dir,
        shards_to_search=shards_to_search,
        feature_chunk_size=feature_chunk_size,
        lower_quantile_thresholds=bin_thresholds,
        n_top_proteins_to_track=10,
        # Per-bin output cap: Phase B's recipe wants up to 24 across the top two
        # bins, so the per-bin sample must exceed 10 (n_top only sizes the heaps,
        # not these bin lists). Defaults to 30.
        n_per_bin_to_sample=n_per_bin_to_sample,
        # Zero bin gets a larger reservoir pool so Phase B can later draw close
        # (k-mer-matched) negatives from a composition-representative set.
        n_zero_to_sample=n_zero_to_sample,
        activation_threshold=0.0,
    )
    print(f"Phase A: scan took {time.time() - t0:.1f}s")

    # find_max_examples_per_feat tracks ALL features by their ABSOLUTE index
    # (_restrict_to_feature_subset is a no-op), so results["lower_quantile"] is
    # keyed {absolute_feat_id: {bin: [protein_ids]}}. Index it by feat_id, not by
    # position in feature_ids — otherwise feature live[i] gets feature i's bins.
    bin_assignments: dict[int, dict[tuple[float, float], list[str]]] = {}
    for feat_id in feature_ids:
        bin_assignments[feat_id] = {
            k: list(v) for k, v in results["lower_quantile"][feat_id].items()
        }

    # Cache (with tuple keys serialized as "lo,hi")
    serializable = {
        fid: {f"{lo},{hi}": list(prots) for (lo, hi), prots in bins.items()}
        for fid, bins in bin_assignments.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        yaml.dump(serializable, f, default_flow_style=False)
    print(f"Phase A: cached bin assignments to {cache_path}")
    return bin_assignments


def _restrict_to_feature_subset(sae, feature_ids: list[int]):
    """No-op for now: find_max_examples_per_feat already chunks across all
    features and the per-feature work is dominated by encoding, not by the
    feature dimension. Future optimization could slice the encoder columns
    to only the requested feature IDs when we're processing a small subset.
    """
    return sae


# ---------------------------------------------------------------------------
# Phase B: apply paper sampling rule per feature
# ---------------------------------------------------------------------------


def phase_b_apply_paper_rule(
    bin_assignments: dict[int, dict[tuple[float, float], list[str]]],
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    """Return {feature_id: [{"entry", "bin", "split"}, ...]}.

    Drops features with fewer than MIN_PROTEINS_IN_TOP_THREE proteins across
    the top three (0.7, 1.0] bins combined.
    """
    rng = np.random.default_rng(seed)
    sampled: dict[int, list[dict[str, Any]]] = {}
    n_dropped_sparse = 0

    for feat_id, bins in bin_assignments.items():
        top_three = (
            len(bins.get((0.7, 0.8), []))
            + len(bins.get((0.8, 0.9), []))
            + len(bins.get((0.9, 1.0), []))
        )
        if top_three < MIN_PROTEINS_IN_TOP_THREE:
            n_dropped_sparse += 1
            continue

        picks: list[dict[str, Any]] = []
        used: set[str] = set()

        # Top bin: up to N_TOP_BIN
        top = list(bins.get((0.9, 1.0), []))
        rng.shuffle(top)
        for entry in top[:N_TOP_BIN]:
            picks.append({"entry": entry, "bin": 10})
            used.add(entry)

        # Second-top bin (0.8, 0.9]: at least 2, top up to N_TOP_TWO_TARGET
        # combined with the top bin.
        second = [e for e in bins.get((0.8, 0.9), []) if e not in used]
        rng.shuffle(second)
        n_second = min(len(second), max(N_PER_NON_TOP_BIN, N_TOP_TWO_TARGET - len(picks)))
        for entry in second[:n_second]:
            picks.append({"entry": entry, "bin": 9})
            used.add(entry)

        # Other non-top bins: 2 each
        for bin_idx_within_bin_edges, (lo, hi) in enumerate(BIN_EDGES[:-2]):
            bin_int = bin_idx_within_bin_edges + 1
            avail = [e for e in bins.get((lo, hi), []) if e not in used]
            rng.shuffle(avail)
            for entry in avail[:N_PER_NON_TOP_BIN]:
                picks.append({"entry": entry, "bin": bin_int})
                used.add(entry)

        # Zero-activation negatives
        zeros = [e for e in bins.get((0.0, 0.0), []) if e not in used]
        rng.shuffle(zeros)
        for entry in zeros[:N_ZERO_NEGATIVES]:
            picks.append({"entry": entry, "bin": 0})
            used.add(entry)

        # 50/50 train/eval shuffle
        rng.shuffle(picks)
        mid = len(picks) // 2
        for p in picks[:mid]:
            p["split"] = "train"
        for p in picks[mid:]:
            p["split"] = "eval"

        sampled[feat_id] = picks

    print(
        f"Phase B: selected examples for {len(sampled)} features; "
        f"dropped {n_dropped_sparse} sparse features (<{MIN_PROTEINS_IN_TOP_THREE} "
        f"in top 3 bins)"
    )
    return sampled


# ---------------------------------------------------------------------------
# Phase C: re-scan shards to extract per-residue traces for sampled pairs
# ---------------------------------------------------------------------------


def phase_c_extract_traces(
    sae_dir: Path,
    embeddings_dir: Path,
    metadata_dir: Path,
    shards_to_search: list[int],
    sampled: dict[int, list[dict[str, Any]]],
    trace_threshold: float,
    feature_chunk_size: int,
    cache_path: Path,
) -> dict[tuple[int, str], list[tuple[int, str, float]]]:
    """Return {(feature_id, protein_entry): [(position, residue, activation), ...]}.

    Only positions with normalized activation > trace_threshold are stored.
    """
    if cache_path.exists():
        print(f"Phase C: loading cached traces from {cache_path}")
        df = pd.read_parquet(cache_path)
        out: dict[tuple[int, str], list[tuple[int, str, float]]] = defaultdict(list)
        for _, row in df.iterrows():
            out[(int(row["feature_id"]), row["protein_entry"])].append(
                (int(row["position"]), row["residue"], float(row["activation"]))
            )
        return dict(out)

    # Build the set of (feature, protein) pairs we need.
    pairs_per_protein: dict[str, set[int]] = defaultdict(set)
    for feat_id, picks in sampled.items():
        for p in picks:
            pairs_per_protein[p["entry"]].add(feat_id)
    needed_proteins = set(pairs_per_protein.keys())
    needed_features = sorted({f for fs in pairs_per_protein.values() for f in fs})
    print(
        f"Phase C: extracting traces for {len(needed_proteins)} proteins × "
        f"{len(needed_features)} features"
    )

    from interplm.data_processing.embedding_loader import load_shard_embeddings
    from interplm.sae.inference import load_sae

    sae = load_sae(sae_dir, model_name="ae_normalized.pt")
    device = next(sae.parameters()).device

    traces: dict[tuple[int, str], list[tuple[int, str, float]]] = {}

    # We chunk features for memory. With ~5k features and chunk=512, ~10 chunks.
    for shard in shards_to_search:
        shard_data = load_shard_embeddings(
            embeddings_dir, shard, device=str(device), return_tensor_only=False
        )
        if isinstance(shard_data, dict) and "protein_ids" in shard_data:
            embeddings = shard_data["embeddings"]
            boundaries = shard_data["boundaries"]
            protein_ids = list(shard_data["protein_ids"])
        else:
            # Fallback: rebuild from metadata TSV
            tsv = pd.read_csv(
                metadata_dir / f"shard_{shard}" / "protein_data.tsv", sep="\t"
            )
            embeddings = (
                shard_data if isinstance(shard_data, torch.Tensor) else shard_data["embeddings"]
            )
            boundaries = []
            cursor = 0
            protein_ids = []
            for _, row in tsv.iterrows():
                protein_ids.append(str(row["Entry"]))
                boundaries.append((cursor, cursor + int(row["Length"])))
                cursor += int(row["Length"])

        # Only process proteins we actually need
        relevant_idx = [
            i for i, pid in enumerate(protein_ids) if pid in needed_proteins
        ]
        if not relevant_idx:
            continue
        print(f"  shard {shard}: {len(relevant_idx)} relevant proteins")

        for prot_local_idx in relevant_idx:
            pid = protein_ids[prot_local_idx]
            start, end = boundaries[prot_local_idx]
            wanted_feats = sorted(pairs_per_protein[pid])
            if not wanted_feats:
                continue
            seq_embed = embeddings[start:end].to(device)  # [L, 1, 24, 1024]
            # Chunk features through encode_feat_subset to bound peak memory.
            for chunk_start in range(0, len(wanted_feats), feature_chunk_size):
                chunk = wanted_feats[chunk_start : chunk_start + feature_chunk_size]
                with torch.no_grad():
                    feats = sae.encode_feat_subset(
                        seq_embed, chunk, normalize_features=True
                    )  # [L, len(chunk)] in [0,1]
                feats_np = feats.cpu().numpy()
                # We don't have the sequence string here from embeddings alone.
                # The Phase E joiner will fill in residue letters from the
                # processed_annotations TSV. For now, store position +
                # activation only.
                for col, fid in enumerate(chunk):
                    above = np.flatnonzero(feats_np[:, col] > trace_threshold)
                    traces[(fid, pid)] = [
                        (int(i), "?", float(feats_np[i, col])) for i in above
                    ]

    # Cache as parquet
    rows = []
    for (fid, pid), trace in traces.items():
        for pos, aa, act in trace:
            rows.append(
                {
                    "feature_id": fid,
                    "protein_entry": pid,
                    "position": pos,
                    "residue": aa,
                    "activation": act,
                }
            )
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Phase C: cached {len(rows)} trace rows to {cache_path}")
    return traces


# ---------------------------------------------------------------------------
# Phase D: UniProt REST fetch (cached on disk)
# ---------------------------------------------------------------------------


def phase_d_fetch_uniprot(
    entries: set[str], cache_path: Path, sleep_s: float = 0.1
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)
        print(f"Phase D: loaded {len(cache)} cached UniProt records from {cache_path}")
    todo = sorted(entries - set(cache.keys()))
    print(f"Phase D: fetching {len(todo)} new UniProt records")

    for i, entry in enumerate(todo):
        url = (
            f"https://rest.uniprot.org/uniprotkb/{entry}.json?fields={UNIPROT_FIELDS}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            cache[entry] = resp.json()
        except Exception as e:
            print(f"  ! {entry}: {e}")
            cache[entry] = {}
        if (i + 1) % 100 == 0:
            with cache_path.open("w") as f:
                json.dump(cache, f)
            print(f"  fetched {i + 1}/{len(todo)}, cache flushed")
        time.sleep(sleep_s)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f)
    print(f"Phase D: wrote {len(cache)} UniProt records to {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Phase E: write final parquet
# ---------------------------------------------------------------------------


def phase_e_write_parquet(
    output_path: Path,
    sampled: dict[int, list[dict[str, Any]]],
    traces: dict[tuple[int, str], list[tuple[int, str, float]]],
    uniprot_cache: dict[str, dict[str, Any]],
    sequences: dict[str, str],
) -> None:
    rows = []
    for feat_id, picks in sampled.items():
        for p in picks:
            entry = p["entry"]
            trace = traces.get((feat_id, entry), [])
            seq = sequences.get(entry, "")
            # Fill in residue letters now that we know the sequence.
            trace_with_letters = [
                {
                    "position": pos,
                    "residue": seq[pos] if seq and 0 <= pos < len(seq) else aa,
                    "activation": act,
                }
                for (pos, aa, act) in trace
            ]
            rows.append(
                {
                    "feature_id": feat_id,
                    "protein_entry": entry,
                    "bin": p["bin"],
                    "split": p["split"],
                    "n_activated_residues": len(trace_with_letters),
                    "max_activation_observed": (
                        max((r["activation"] for r in trace_with_letters), default=0.0)
                    ),
                    "activated_residues_json": json.dumps(trace_with_letters),
                    "uniprot_meta_json": json.dumps(uniprot_cache.get(entry, {})),
                }
            )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Phase E: wrote {len(df):,} rows to {output_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def gather_sequences(
    metadata_dir: Path, shards_to_search: list[int], needed_entries: set[str]
) -> dict[str, str]:
    """Build {entry: sequence} from processed_annotations TSVs for the
    proteins we need to fill in residue letters in traces."""
    seqs: dict[str, str] = {}
    for shard in shards_to_search:
        tsv_path = metadata_dir / f"shard_{shard}" / "protein_data.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t", usecols=["Entry", "Sequence"])
        df = df[df["Entry"].isin(needed_entries)]
        for _, row in df.iterrows():
            seqs[str(row["Entry"])] = str(row["Sequence"])
    return seqs


def parse_feature_ids(arg: str, sae_dir: Path) -> list[int]:
    if arg.lower() == "live":
        from interplm.sae.inference import load_sae

        sae = load_sae(sae_dir, model_name="ae_normalized.pt", device="cpu")
        live = (sae.activation_rescale_factor > 0).nonzero().squeeze().tolist()
        print(f"--feature-ids live → {len(live)} live features")
        return [int(x) for x in live]
    return [int(x) for x in arg.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shards-to-search",
        type=str,
        default="all",
        help="Comma-separated shard indices or 'all' (default).",
    )
    parser.add_argument(
        "--feature-ids",
        type=str,
        default="live",
        help="Comma-separated feature ids, or 'live' for all features with "
        "rescale_factor > 0 (default).",
    )
    parser.add_argument("--feature-chunk-size", type=int, default=256)
    parser.add_argument("--trace-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=30,
        help="Max proteins sampled per activation bin in Phase A (default 30; must "
        "exceed 10 so Phase B can reach its 24-in-top-two recipe target).",
    )
    parser.add_argument(
        "--n-zero",
        type=int,
        default=500,
        help="Size of the zero-activation pool per feature (default 500), filled by "
        "reservoir sampling across all shards so close (composition-matched) negatives "
        "can be drawn from it later. Larger than --n-per-bin because it is a candidate "
        "pool to match within, not a final selection.",
    )
    parser.add_argument(
        "--phase-a-only",
        action="store_true",
        help="Run only Phase A (bin scan + cache) and exit. Use to pre-produce the "
        "binning on a GPU/compute node; a later full run reuses the cache and runs "
        "Phases B-E (Phase D needs internet, so run that where UniProt is reachable).",
    )
    parser.add_argument("--skip-uniprot", action="store_true")
    args = parser.parse_args()

    # Seed for reproducible per-bin sampling (PerProteinActivationTracker.get_results
    # samples via the global np.random) and the Phase B train/eval split rng.
    np.random.seed(args.seed)
    print(f"Seeded np.random with seed={args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve shard list
    if args.shards_to_search == "all":
        shards = sorted(
            int(p.name.split("_")[1])
            for p in args.embeddings_dir.iterdir()
            if p.name.startswith("shard_")
        )
    else:
        shards = [int(s) for s in args.shards_to_search.split(",") if s.strip()]
    print(f"Using shards: {shards}")

    feature_ids = parse_feature_ids(args.feature_ids, args.sae_dir)

    # Phase A: bin scan
    bin_assignments = phase_a_bin_scan(
        sae_dir=args.sae_dir,
        embeddings_dir=args.embeddings_dir,
        metadata_dir=args.metadata_dir,
        shards_to_search=shards,
        feature_ids=feature_ids,
        feature_chunk_size=args.feature_chunk_size,
        cache_path=cache_dir / "bin_assignments.yaml",
        n_per_bin_to_sample=args.n_per_bin,
        n_zero_to_sample=args.n_zero,
    )

    if args.phase_a_only:
        n_bins_populated = sum(
            1 for bins in bin_assignments.values() for v in bins.values() if len(v) > 0
        )
        print(
            f"--phase-a-only: wrote bin assignments for {len(bin_assignments)} features "
            f"({n_bins_populated} non-empty (feature, bin) cells) to "
            f"{cache_dir / 'bin_assignments.yaml'}. Stopping before Phase B."
        )
        return 0

    # Phase B: paper rule
    sampled = phase_b_apply_paper_rule(bin_assignments, seed=args.seed)
    with (cache_dir / "sampled.json").open("w") as f:
        json.dump({str(k): v for k, v in sampled.items()}, f)

    # Phase C: trace extraction
    traces = phase_c_extract_traces(
        sae_dir=args.sae_dir,
        embeddings_dir=args.embeddings_dir,
        metadata_dir=args.metadata_dir,
        shards_to_search=shards,
        sampled=sampled,
        trace_threshold=args.trace_threshold,
        feature_chunk_size=args.feature_chunk_size,
        cache_path=cache_dir / "traces.parquet",
    )

    # Fetch sequences from processed_annotations TSVs for residue-letter lookup
    needed_entries = {p["entry"] for picks in sampled.values() for p in picks}
    sequences = gather_sequences(args.metadata_dir, shards, needed_entries)
    print(f"Got sequences for {len(sequences)}/{len(needed_entries)} entries")

    # Phase D: UniProt metadata
    if args.skip_uniprot:
        uniprot_cache: dict[str, dict[str, Any]] = {}
    else:
        uniprot_cache = phase_d_fetch_uniprot(
            needed_entries, cache_path=cache_dir / "uniprot.json"
        )

    # Phase E: final parquet
    phase_e_write_parquet(
        output_path=args.output_dir / "Per_feature_llm_input.parquet",
        sampled=sampled,
        traces=traces,
        uniprot_cache=uniprot_cache,
        sequences=sequences,
    )

    # Quick stats
    df = pd.read_parquet(args.output_dir / "Per_feature_llm_input.parquet")
    print(
        f"\nSummary:\n"
        f"  features in output: {df['feature_id'].nunique()}\n"
        f"  total (feature, protein) rows: {len(df)}\n"
        f"  mean activated residues per row: "
        f"{df['n_activated_residues'].mean():.1f}\n"
        f"  rows per split: {df['split'].value_counts().to_dict()}\n"
        f"  rows per bin: {df['bin'].value_counts().sort_index().to_dict()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
