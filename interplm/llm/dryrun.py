"""Local single-feature dry-run of the LLM annotation pipeline.

Walks one crosscoder feature (default: 1322) through the full InterPLM-style
annotation pipeline on the MacBook, end-to-end:

  1. Sample N random short Swiss-Prot proteins from the local TSV.
  2. Embed them with ProtT5 (all 24 encoder residual streams) on MPS.
  3. Run the crosscoder, extract activations for the chosen feature.
  4. Apply per-feature [0,1] normalization manually (see NORMALIZATION below).
  5. Bin proteins into 10 activation bins of width 0.1; sample per the
     InterPLM paper recipe with whatever proteins land where.
  6. Fetch richer metadata (Function [CC], Keywords, GO terms, organism,
     subcellular location) from the UniProt REST API for each sampled
     protein.
  7. Build a description prompt and call Claude Sonnet 4.6 with prompt
     caching on the static system block.
  8. Build a prediction prompt over the held-out half, ask Claude to predict
     each protein's activation bin.
  9. Compute Pearson r between predicted and measured bins.
 10. Report tokens / latency / cost per call and project to ~5,000 features.

NORMALIZATION
-------------
`CrosscoderDictionaryWrapper` currently ignores `normalize_features=True`
and does not register `activation_rescale_factor` as a buffer, so the
rescale tensor in `ae_normalized.pt` is silently dropped on load. We work
around that here by loading the state-dict separately and dividing the
raw feature activation by `activation_rescale_factor[feature_id]`.

USAGE
-----
    uv run python repos/InterPLM/interplm/llm/dryrun.py \
        --feature-id 1322 \
        --n-proteins 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv

REPO_ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
INTERPLM_ROOT = REPO_ROOT / "repos" / "InterPLM"
CROSSCODE_ROOT = REPO_ROOT / "repos" / "crosscode"
ENV_PATH = REPO_ROOT / "repos" / "sparse-crosscoders-prott5" / ".env"
CHECKPOINT_DIR = (
    REPO_ROOT
    / "model_checkpoints"
    / "crosscoder_l8192_k32_bs512_full_2026-03-12_06-03-41"
    / "crashed_epoch_0_step_2519836"
)
PROTEINS_TSV = (
    REPO_ROOT
    / "data"
    / "eval_dataset"
    / "uniprotkb_modern_score45_67k"
    / "proteins.tsv"
)

# Top-10 max-activating proteins per feature, harvested from the 67k
# dashboard cache. Used to seed the dry-run with known high-activation
# examples since feature 1322 is highly specific and won't be hit by
# random short-protein sampling.
KNOWN_TOP_ACTIVATORS: dict[int, list[str]] = {
    1322: [
        "Q9LPQ3",
        "Q9FX43",
        "Q9JJ78",
        "Q96KB5",
        "G5EDT6",
        "Q8RXG3",
        "P9WI75",
        "D3ZBE5",
        "G5EFM9",
        "O80397",
    ],
}

# Sonnet 4.6 pricing as of 2026-05 — update if you re-run later.
PRICE_INPUT_PER_M = 3.0
PRICE_OUTPUT_PER_M = 15.0
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30

MODEL_NAME = "claude-sonnet-4-6"

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

# Paper's bin scheme: 10 bins of width 0.1.
BIN_EDGES = [(round(i * 0.1, 1), round((i + 1) * 0.1, 1)) for i in range(10)]


for p in (INTERPLM_ROOT, CROSSCODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProteinRecord:
    entry: str
    sequence: str
    length: int
    local_meta: dict[str, Any]  # raw row from proteins.tsv
    per_residue_acts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_act: float = 0.0
    bin: tuple[float, float] | None = None
    uniprot_meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Step 1: sample proteins
# ---------------------------------------------------------------------------


def sample_proteins(
    n: int,
    max_length: int,
    seed: int,
    seed_entries: list[str] | None = None,
) -> list[ProteinRecord]:
    df = pd.read_csv(PROTEINS_TSV, sep="\t")
    df_in_range = df[df["Length"] <= max_length]
    sampled = df_in_range.sample(n=n, random_state=seed)

    if seed_entries:
        # Force-include seed entries even if they exceed max_length, since they
        # are known high activators for the target feature.
        forced = df[df["Entry"].isin(seed_entries)]
        missing = set(seed_entries) - set(forced["Entry"])
        if missing:
            print(f"  ! seed entries not in TSV (skipped): {sorted(missing)}")
        sampled = pd.concat([forced, sampled]).drop_duplicates(subset=["Entry"])

    sampled = sampled.reset_index(drop=True)
    records: list[ProteinRecord] = []
    for _, row in sampled.iterrows():
        rec = ProteinRecord(
            entry=str(row["Entry"]),
            sequence=str(row["Sequence"]),
            length=int(row["Length"]),
            local_meta=row.to_dict(),
        )
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Step 2-4: embed + encode + normalize
# ---------------------------------------------------------------------------


def get_local_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_and_encode(
    records: list[ProteinRecord],
    feature_id: int,
    device: str,
    batch_size: int,
) -> None:
    """In-place: populates per_residue_acts and max_act on each record.

    Normalization is now handled inside ``CrosscoderDictionaryWrapper``:
    ``encode_feat_subset(..., normalize_features=True)`` returns activations
    in the [0,1] range, with dead features safely producing 0.
    """

    from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
    from interplm.sae.inference import load_sae

    print(f"Loading ProtT5 + crosscoder on device={device}...")
    embedder = ProtT5CrosscoderEmbedder(device=device)
    wrapper = load_sae(
        CHECKPOINT_DIR, model_name="ae_normalized.pt", device=device
    )
    rescale = float(wrapper.activation_rescale_factor[feature_id].item())
    if rescale <= 0:
        raise RuntimeError(
            f"Feature {feature_id} has rescale factor {rescale} — likely dead. "
            "Pick a different feature."
        )
    print(f"Rescale factor for feature {feature_id}: {rescale:.4f}")

    sequences = [r.sequence for r in records]
    print(f"Embedding {len(sequences)} sequences (batch_size={batch_size})...")
    t0 = time.time()
    out = embedder.extract_embeddings_with_boundaries(
        sequences, batch_size=batch_size
    )
    print(f"  ProtT5 forward took {time.time() - t0:.1f}s")

    embeddings = out["embeddings"].to(device)  # [total_tokens, 1, 24, 1024]
    boundaries = out["boundaries"]

    print(f"Encoding {embeddings.shape[0]} tokens through crosscoder...")
    t0 = time.time()
    feats = wrapper.encode_feat_subset(
        embeddings, [feature_id], normalize_features=True
    )  # [T, 1] already in [0,1]
    feats = np.clip(feats.squeeze(-1).cpu().float().numpy(), 0.0, 1.0)
    print(f"  Crosscoder forward took {time.time() - t0:.1f}s")

    for rec, (start, end) in zip(records, boundaries):
        per_res = feats[start:end]
        rec.per_residue_acts = per_res
        rec.max_act = float(per_res.max()) if per_res.size else 0.0


# ---------------------------------------------------------------------------
# Step 5: bin + sample per paper rule
# ---------------------------------------------------------------------------


def assign_bin(value: float) -> tuple[float, float]:
    if value <= 0.0:
        return (0.0, 0.0)
    for lo, hi in BIN_EDGES:
        if lo < value <= hi:
            return (lo, hi)
    return (0.9, 1.0)  # exact 1.0


def select_examples(
    records: list[ProteinRecord], rng: random.Random
) -> tuple[list[ProteinRecord], list[ProteinRecord]]:
    """Apply the InterPLM paper sampling rule with whatever proteins we have.

    With only ~50 proteins, most bins will be empty. We do best-effort:
    per non-top bin take up to 2; top bin (0.9,1.0) take up to 10; top up
    (0.8,0.9) to reach 24 total in the top two combined; add up to 10 zero
    proteins. Split 50/50 train/eval.
    """

    by_bin: dict[tuple[float, float], list[ProteinRecord]] = defaultdict(list)
    for rec in records:
        rec.bin = assign_bin(rec.max_act)
        by_bin[rec.bin].append(rec)

    print("\nBin distribution (max-activation):")
    print(f"  zero (0.0): {len(by_bin[(0.0, 0.0)])}")
    for lo, hi in BIN_EDGES:
        print(f"  ({lo:.1f}, {hi:.1f}]: {len(by_bin[(lo, hi)])}")

    selected: list[ProteinRecord] = []

    # Non-top bins: 2 each
    for lo, hi in BIN_EDGES[:-2]:
        pool = by_bin[(lo, hi)]
        selected.extend(rng.sample(pool, min(2, len(pool))))

    # Top two bins: aim for 24 combined, with (0.9,1.0) getting up to 10
    top_pool = by_bin[(0.9, 1.0)]
    second_pool = by_bin[(0.8, 0.9)]
    n_top = min(10, len(top_pool))
    selected.extend(rng.sample(top_pool, n_top))
    n_second_target = max(0, 24 - n_top - 2)  # 2 was already added for (0.8,0.9)
    extras_needed = max(0, n_second_target)
    second_already = min(2, len(second_pool))
    extra_available = max(0, len(second_pool) - second_already)
    second_extras = min(extras_needed, extra_available)
    if second_extras > 0:
        already = [
            r
            for r in selected
            if r.bin == (0.8, 0.9)
        ]
        remaining = [r for r in second_pool if r not in already]
        selected.extend(rng.sample(remaining, second_extras))

    # Zero-activation negatives
    zeros = by_bin[(0.0, 0.0)]
    selected.extend(rng.sample(zeros, min(10, len(zeros))))

    # Deduplicate while preserving order
    seen = set()
    deduped: list[ProteinRecord] = []
    for r in selected:
        if r.entry in seen:
            continue
        seen.add(r.entry)
        deduped.append(r)

    rng.shuffle(deduped)
    mid = len(deduped) // 2
    train = deduped[:mid]
    eval_ = deduped[mid:]
    print(
        f"\nSelected {len(deduped)} proteins for LLM "
        f"(train={len(train)}, eval={len(eval_)})"
    )
    return train, eval_


# ---------------------------------------------------------------------------
# Step 6: UniProt REST fetch
# ---------------------------------------------------------------------------


def fetch_uniprot_metadata(records: list[ProteinRecord]) -> None:
    print(f"\nFetching UniProt metadata for {len(records)} proteins...")
    for rec in records:
        url = (
            f"https://rest.uniprot.org/uniprotkb/{rec.entry}.json"
            f"?fields={UNIPROT_FIELDS}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            rec.uniprot_meta = resp.json()
        except Exception as e:
            print(f"  ! {rec.entry}: {e}")
            rec.uniprot_meta = {}
        time.sleep(0.1)  # courtesy to UniProt API


def summarize_uniprot_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce UniProt JSON to the fields we want in the prompt."""
    if not meta:
        return {}
    out: dict[str, Any] = {}

    pd_section = meta.get("proteinDescription", {})
    rec_name = pd_section.get("recommendedName", {}).get("fullName", {}).get(
        "value"
    )
    if rec_name:
        out["recommended_name"] = rec_name

    organism = meta.get("organism", {}).get("scientificName")
    if organism:
        out["organism"] = organism

    keywords = [k.get("name") for k in meta.get("keywords", []) if k.get("name")]
    if keywords:
        out["keywords"] = keywords

    comments = meta.get("comments", [])
    for c in comments:
        if c.get("commentType") == "FUNCTION":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            if texts:
                out["function"] = " ".join(texts)
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            locs = []
            for loc in c.get("subcellularLocations", []):
                v = loc.get("location", {}).get("value")
                if v:
                    locs.append(v)
            if locs:
                out["subcellular_location"] = locs

    refs = meta.get("uniProtKBCrossReferences", [])
    gos = []
    for r in refs:
        if r.get("database") == "GO":
            term = next(
                (p.get("value") for p in r.get("properties", []) if p.get("key") == "GoTerm"),
                None,
            )
            if term:
                gos.append(term)
    if gos:
        out["go_terms"] = gos[:30]

    features = meta.get("features", [])
    feat_summary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in features:
        t = f.get("type")
        loc = f.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        desc = f.get("description") or ""
        if t and start is not None and end is not None:
            feat_summary[t].append({"start": start, "end": end, "desc": desc})
    if feat_summary:
        out["features"] = dict(feat_summary)

    return out


# ---------------------------------------------------------------------------
# Step 7-8: prompts
# ---------------------------------------------------------------------------


from interplm.llm.prompts import (
    SYSTEM_DESCRIPTION,
    SYSTEM_PREDICTION,
    parse_json_response as _shared_parse_json_response,
)


def bin_to_int(bin_: tuple[float, float] | None) -> int:
    if bin_ is None or bin_ == (0.0, 0.0):
        return 0
    return int(round(bin_[0] * 10)) + 1


def build_protein_example(rec: ProteinRecord, include_activation: bool) -> dict[str, Any]:
    meta = summarize_uniprot_meta(rec.uniprot_meta)
    payload: dict[str, Any] = {
        "entry": rec.entry,
        "length": rec.length,
        "metadata": meta,
    }
    if include_activation:
        payload["activation_bin"] = bin_to_int(rec.bin)
        payload["max_activation"] = round(rec.max_act, 3)
        activated = [
            {
                "position": int(i + 1),  # 1-indexed for readability
                "residue": rec.sequence[i],
                "activation": round(float(rec.per_residue_acts[i]), 3),
            }
            for i in range(rec.length)
            if rec.per_residue_acts[i] > 0.1
        ]
        payload["activated_residues"] = activated
    return payload


# ---------------------------------------------------------------------------
# Step 9: Claude calls
# ---------------------------------------------------------------------------


def call_anthropic(
    client: Any,
    system_text: str,
    user_payload: dict[str, Any],
    max_tokens: int,
    assistant_prefill: str | None = None,
) -> tuple[str, dict[str, Any], float]:
    t0 = time.time()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        }
    ]
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )
    dt = time.time() - t0
    text = "".join(block.text for block in resp.content if block.type == "text")
    if assistant_prefill:
        text = assistant_prefill + text
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            resp.usage, "cache_creation_input_tokens", 0
        )
        or 0,
        "cache_read_input_tokens": getattr(
            resp.usage, "cache_read_input_tokens", 0
        )
        or 0,
    }
    return text, usage, dt


parse_json_response = _shared_parse_json_response


def cost_for_usage(usage: dict[str, Any]) -> float:
    return (
        usage["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_M
        + usage["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
        + usage["cache_creation_input_tokens"] / 1_000_000 * PRICE_CACHE_WRITE_PER_M
        + usage["cache_read_input_tokens"] / 1_000_000 * PRICE_CACHE_READ_PER_M
    )


# ---------------------------------------------------------------------------
# Step 10: Pearson
# ---------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-id", type=int, default=1322)
    parser.add_argument("--n-proteins", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-entries",
        type=str,
        default="",
        help="Comma-separated UniProt IDs to force-include. Defaults to "
        "KNOWN_TOP_ACTIVATORS[feature_id] if available.",
    )
    parser.add_argument(
        "--n-projected-features",
        type=int,
        default=5000,
        help="Live-feature count for production cost projection.",
    )
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    # python-dotenv silently skips files with no trailing newline; manual fallback.
    if not os.environ.get("ANTHROPIC_API_KEY") and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines() or [ENV_PATH.read_text()]:
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"ERROR: ANTHROPIC_API_KEY not set (looked at {ENV_PATH})")
        return 1

    import anthropic

    rng = random.Random(args.seed)
    device = get_local_device()
    print(f"Device: {device}, feature_id={args.feature_id}")

    print("\n=== Step 1: sample proteins ===")
    if args.seed_entries.strip():
        seed_entries = [s.strip() for s in args.seed_entries.split(",") if s.strip()]
    else:
        seed_entries = KNOWN_TOP_ACTIVATORS.get(args.feature_id, [])
    if seed_entries:
        print(f"Force-including {len(seed_entries)} known activators")
    records = sample_proteins(
        args.n_proteins, args.max_length, args.seed, seed_entries=seed_entries
    )
    print(
        f"Sampled {len(records)} proteins, mean length "
        f"{np.mean([r.length for r in records]):.1f}"
    )

    print("\n=== Step 2-4: embed + encode + normalize ===")
    embed_and_encode(records, args.feature_id, device, args.batch_size)
    n_nonzero = sum(1 for r in records if r.max_act > 0)
    print(
        f"\nFeature {args.feature_id}: fired on {n_nonzero}/{len(records)} proteins"
    )
    if n_nonzero == 0:
        print("Feature is dead on this sample — aborting.")
        return 2

    print("\n=== Step 5: bin + sample per paper rule ===")
    train, eval_ = select_examples(records, rng)
    if not train or not eval_:
        print("Not enough proteins for train+eval split — aborting.")
        return 3

    print("\n=== Step 6: fetch UniProt metadata ===")
    fetch_uniprot_metadata(train + eval_)

    print("\n=== Step 7: build description prompt ===")
    description_user = {
        "feature_id": args.feature_id,
        "n_training_proteins": len(train),
        "proteins": [build_protein_example(r, include_activation=True) for r in train],
    }
    prompt_text = json.dumps(description_user, ensure_ascii=False)
    print(f"User prompt length: {len(prompt_text):,} chars")

    print("\n=== Step 8: description call ===")
    client = anthropic.Anthropic()
    desc_text, desc_usage, desc_dt = call_anthropic(
        client, SYSTEM_DESCRIPTION, description_user, max_tokens=1500
    )
    print(f"Latency: {desc_dt:.1f}s")
    print(f"Usage: {desc_usage}")
    try:
        desc_obj = parse_json_response(desc_text)
        print("\n--- Description ---")
        print(desc_obj.get("description", "(missing)"))
        print("\n--- Summary ---")
        print(desc_obj.get("summary", "(missing)"))
    except Exception as e:
        print(f"Failed to parse description JSON: {e}")
        print(f"Raw: {desc_text[:500]}")
        return 4

    print("\n=== Step 9: prediction call ===")
    prediction_user = {
        "feature_description": desc_obj.get("description", ""),
        "feature_summary": desc_obj.get("summary", ""),
        "n_proteins": len(eval_),
        "proteins": [build_protein_example(r, include_activation=False) for r in eval_],
    }
    pred_text, pred_usage, pred_dt = call_anthropic(
        client, SYSTEM_PREDICTION, prediction_user, max_tokens=1500
    )
    print(f"Latency: {pred_dt:.1f}s")
    print(f"Usage: {pred_usage}")
    try:
        preds = parse_json_response(pred_text)
        if not isinstance(preds, list):
            raise ValueError("expected JSON array")
    except Exception as e:
        print(f"Failed to parse prediction JSON: {e}")
        print(f"Raw: {pred_text[:500]}")
        return 5

    by_entry = {p["entry"]: p["predicted_bin"] for p in preds}
    actual_bins, predicted_bins = [], []
    for r in eval_:
        if r.entry not in by_entry:
            print(f"  ! No prediction for {r.entry}")
            continue
        actual_bins.append(bin_to_int(r.bin))
        predicted_bins.append(int(by_entry[r.entry]))
    print("\nPredictions:")
    for r in eval_:
        if r.entry in by_entry:
            print(
                f"  {r.entry}: predicted={by_entry[r.entry]:>2}  "
                f"measured={bin_to_int(r.bin):>2}  max_act={r.max_act:.3f}"
            )

    r_val = pearson(predicted_bins, actual_bins)
    print(f"\nPearson r (predicted vs measured bin): {r_val:.3f}")
    print(f"(Paper's median across 1,240 features: 0.72)")

    # ----- cost / projection -----
    total_usage = {k: desc_usage[k] + pred_usage[k] for k in desc_usage}
    total_cost = cost_for_usage(total_usage)
    total_latency = desc_dt + pred_dt
    print("\n=== Cost report ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Total tokens this feature: {total_usage}")
    print(f"Total cost this feature: ${total_cost:.4f}")
    print(f"Total wall-time this feature: {total_latency:.1f}s")

    # Projection: assume same per-feature tokens, but with cache savings on
    # the system block. In production the system block is cached on the first
    # call, then re-read for the rest of the features × 2 calls.
    sys_tokens_estimate = max(
        total_usage["cache_creation_input_tokens"],
        # rough estimate if no caching happened on this single run:
        len(SYSTEM_DESCRIPTION) // 4 + len(SYSTEM_PREDICTION) // 4,
    )
    variable_input = max(
        0,
        total_usage["input_tokens"] - sys_tokens_estimate,
    )
    output_tokens = total_usage["output_tokens"]
    n = args.n_projected_features
    projected_cost = (
        sys_tokens_estimate / 1_000_000 * PRICE_CACHE_WRITE_PER_M  # write once
        + (n - 1) * sys_tokens_estimate / 1_000_000 * PRICE_CACHE_READ_PER_M  # read n-1 times for description
        + n * sys_tokens_estimate / 1_000_000 * PRICE_CACHE_READ_PER_M  # read n times for prediction
        + n * variable_input / 1_000_000 * PRICE_INPUT_PER_M
        + n * output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    )
    print(
        f"\nProjection to {n} features (with system-block prompt caching):"
    )
    print(f"  Estimated cost: ${projected_cost:.2f}")
    print(
        f"  Estimated wall-time @ 50 parallel calls: "
        f"{n * total_latency / 50 / 60:.1f} min"
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
