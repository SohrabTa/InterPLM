"""Pilot: multi-provider autointerp generate-then-score on the ProtT5 crosscoder.

Goal of this pilot
------------------
Validate that the InterPLM-style "describe a feature from training proteins,
then predict held-out proteins' activation bins" loop produces a meaningful
Pearson r on OUR crosscoder, and compare LLM providers on quality vs cost.

It reuses the data-prep machinery from ``dryrun.py`` (sample proteins, embed
with ProtT5 + crosscoder on MPS, bin per the InterPLM rule, fetch UniProt
metadata, build prompts) but:
  - loads ProtT5 + the crosscoder ONCE and loops over many features,
  - seeds each feature with its known top activators from the dashboard cache
    (``Per_feature_max_examples.yaml``) so specific features actually fire,
  - calls multiple providers (Anthropic + any OpenAI-compatible endpoint:
    OpenAI, DeepSeek) per feature,
  - adds a mismatched-label null (describe feature A, predict feature B's
    held-out bins -> r should be ~0),
  - writes a tidy results CSV + prints a summary table with measured tokens/cost.

Inputs read:
  - model_checkpoints/.../crashed_epoch_0_step_2519836/ (full crosscoder)
  - data/eval_dataset/uniprotkb_modern_score45_67k/proteins.tsv (sequences+meta)
  - data/dashboard_cache/.../Per_feature_max_examples.yaml (seed activators)
  - UniProt REST API (richer per-protein metadata)
  - repos/sparse-crosscoders-prott5/.env (ANTHROPIC/OPENAI/DEEPSEEK keys)

Outputs written:
  - <out>/pilot_results.csv  (one row per feature x provider, + mismatched rows)
  - prints a summary table

Usage:
  uv run --with openai python interplm/llm/pilot_autointerp.py --check
  uv run --with openai python interplm/llm/pilot_autointerp.py \
      --features 1322,1339,7966 --providers deepseek,openai,anthropic \
      --n-neg 250 --out /tmp/pilot
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

# Reuse everything data-side from the dry-run module.
from interplm.llm import dryrun as dr
from interplm.llm.prompts import (
    SYSTEM_DESCRIPTION,
    SYSTEM_PREDICTION,
    parse_json_response,
)

REPO_ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
ENV_PATH = REPO_ROOT / "repos" / "sparse-crosscoders-prott5" / ".env"
MAX_EXAMPLES_YAML = (
    REPO_ROOT
    / "data"
    / "dashboard_cache"
    / "uniprotkb_modern_score45_67k"
    / "dashboard_cache"
    / "prott5_crosscoder"
    / "layer_crosscoder"
    / "Per_feature_max_examples.yaml"
)

# ---------------------------------------------------------------------------
# Provider registry. Prices are $/1M tokens, verified 2026-06-05 (see
# documentation/experiments/06; update if you re-run later).
# ---------------------------------------------------------------------------


@dataclass
class Provider:
    name: str
    kind: str  # "anthropic" | "openai"
    model: str
    price_in: float
    price_out: float
    env_key: str
    base_url: str | None = None
    use_max_completion_tokens: bool = False  # reasoning models (gpt-5.x)
    send_temperature: bool = True
    out_token_budget: int = 2000
    json_mode: bool = False  # send response_format={"type":"json_object"} on object-returning calls


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        kind="anthropic",
        model="claude-opus-4-6",
        price_in=5.0,
        price_out=25.0,
        env_key="ANTHROPIC_API_KEY",
        out_token_budget=1500,
    ),
    "openai": Provider(
        name="openai",
        kind="openai",
        model="gpt-5.4",
        price_in=2.5,
        price_out=15.0,
        env_key="OPENAI_API_KEY",
        base_url=None,  # default OpenAI endpoint
        use_max_completion_tokens=True,
        send_temperature=False,
        out_token_budget=8000,  # reasoning tokens count toward completion
    ),
    "deepseek": Provider(
        name="deepseek",
        kind="openai",
        model="deepseek-v4-pro",
        price_in=0.435,
        price_out=0.87,
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        use_max_completion_tokens=False,
        send_temperature=True,
        # Diagnostic (diag_deepseek.py, 2026-06-05): the empty-content failure is
        # finish_reason="length" — reasoning tokens exhaust max_tokens before any
        # answer. Fix is budget, NOT json mode (json mode left 1/4 failing and made
        # reasoning longer) and NOT temperature (irrelevant). Reasoning ran up to
        # ~8000 tokens, so budget 16000 + escalate-on-empty retry.
        out_token_budget=16000,
        json_mode=False,
    ),
}


# ---------------------------------------------------------------------------
# Env / clients
# ---------------------------------------------------------------------------


def load_env() -> None:
    load_dotenv(ENV_PATH)
    # python-dotenv skips files without trailing newline; manual fallback.
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
                if line.startswith(k + "=") and not os.environ.get(k):
                    os.environ[k] = line.split("=", 1)[1]


def make_client(p: Provider) -> Any:
    key = os.environ.get(p.env_key)
    if not key:
        raise RuntimeError(f"{p.env_key} not set (looked at {ENV_PATH})")
    if p.kind == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=key)
    else:
        from openai import OpenAI

        return OpenAI(api_key=key, base_url=p.base_url)


# ---------------------------------------------------------------------------
# Provider calls -> (text, usage{input_tokens,output_tokens}, latency_s)
# ---------------------------------------------------------------------------


def call_provider(
    client: Any,
    p: Provider,
    system_text: str,
    user_payload: dict[str, Any],
    force_json: bool = False,
    max_attempts: int = 3,
) -> tuple[str, dict[str, int], float]:
    """One LLM call -> (text, usage, latency_s).

    force_json: request response_format=json_object (object-returning calls
    only, i.e. the description; the prediction returns a JSON array so json
    mode must NOT be used there). Reasoning models (e.g. DeepSeek V4-Pro) can
    return an empty `content` when the call is spent on hidden reasoning; we
    retry up to max_attempts on empty output.
    """
    t0 = time.time()
    user_text = json.dumps(user_payload, ensure_ascii=False)
    if p.kind == "anthropic":
        resp = client.messages.create(
            model=p.model,
            max_tokens=p.out_token_budget,
            temperature=0.0,  # deterministic labels (was defaulting to 1.0)
            system=[{"type": "text", "text": system_text}],
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": int(resp.usage.input_tokens),
            "output_tokens": int(resp.usage.output_tokens),
        }
        return text, usage, time.time() - t0

    # OpenAI-compatible (OpenAI / DeepSeek). Empty content on reasoning models is
    # finish_reason="length" (reasoning ate the budget); the only reliable fix is
    # a larger budget, so retry escalates max_tokens (temperature/json mode don't
    # help — see diag_deepseek.py). Budget capped at 32000.
    text = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    budget = p.out_token_budget
    for attempt in range(max_attempts):
        kwargs: dict[str, Any] = {
            "model": p.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        }
        if p.use_max_completion_tokens:
            kwargs["max_completion_tokens"] = budget
        else:
            kwargs["max_tokens"] = budget
        if p.send_temperature:
            kwargs["temperature"] = 0.0
        if force_json and p.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage["input_tokens"] += int(resp.usage.prompt_tokens)
        usage["output_tokens"] += int(resp.usage.completion_tokens)
        finish = resp.choices[0].finish_reason
        if text.strip():
            break
        if finish == "length":
            budget = min(budget * 2, 32000)  # reasoning starved the answer; give more room
    return text, usage, time.time() - t0


def cost_usd(p: Provider, usage: dict[str, int]) -> float:
    return (
        usage["input_tokens"] / 1e6 * p.price_in
        + usage["output_tokens"] / 1e6 * p.price_out
    )


# ---------------------------------------------------------------------------
# Seed activators from the dashboard cache
# ---------------------------------------------------------------------------


def load_yaml_max_examples(path: str | Path) -> dict[int, list[str]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    out: dict[int, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list) and v:
            out[int(k)] = [str(x) for x in v]
    return out


def load_max_examples() -> dict[int, list[str]]:
    return load_yaml_max_examples(MAX_EXAMPLES_YAML)


# ---------------------------------------------------------------------------
# Encode a single feature over an already-loaded embedder + crosscoder
# ---------------------------------------------------------------------------


def encode_feature_on_records(
    embedder: Any,
    wrapper: Any,
    records: list[dr.ProteinRecord],
    feature_id: int,
    device: str,
    batch_size: int,
) -> bool:
    """In-place: set per_residue_acts + max_act on each record for this feature.
    Returns False if the feature is dead (rescale factor <= 0)."""
    rescale = float(wrapper.activation_rescale_factor[feature_id].item())
    if rescale <= 0:
        return False
    sequences = [r.sequence for r in records]
    out = embedder.extract_embeddings_with_boundaries(sequences, batch_size=batch_size)
    embeddings = out["embeddings"].to(device)
    boundaries = out["boundaries"]
    feats_t = wrapper.encode_feat_subset(embeddings, [feature_id], normalize_features=True)
    feats = np.clip(feats_t.squeeze(-1).cpu().float().numpy(), 0.0, 1.0)
    for rec, (start, end) in zip(records, boundaries):
        per_res = feats[start:end]
        rec.per_residue_acts = per_res
        rec.max_act = float(per_res.max()) if per_res.size else 0.0
    # Free GPU/MPS memory between features (per-feature encode over a large pool
    # accumulates otherwise and OOMs MPS by the 4th feature).
    import gc
    import torch
    del embeddings, feats_t, out
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _kmer_vec(seq: str, k: int = 3) -> dict[str, float]:
    """L2-normalized k-mer frequency vector of a sequence (sparse dict)."""
    from collections import Counter
    c = Counter(seq[i:i + k] for i in range(max(0, len(seq) - k + 1)))
    norm = (sum(v * v for v in c.values())) ** 0.5 or 1.0
    return {kmer: v / norm for kmer, v in c.items()}


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(kmer, 0.0) for kmer, v in a.items())


def select_with_negatives(records, rng, mode: str, n_each: int = 12):
    """Pick positives (firing proteins) + negatives, split 50/50 into train/eval.

    mode="random": negatives are random zero-activation proteins (the original
      InterPLM recipe; lets a feature win by detecting residue composition).
    mode="close":  negatives are the zero-activation proteins most k-mer-similar
      to the positives (composition-matched hard negatives, Ma et al. 2026
      style). A feature can no longer score above chance merely by detecting
      amino-acid composition, since the negatives share it — this is what
      separates learned concept features from residue-identity features.
    """
    for r in records:
        r.bin = dr.assign_bin(r.max_act)
    pos = [r for r in records if r.max_act > 0]
    zeros = [r for r in records if r.max_act == 0]
    rng.shuffle(pos)
    pos = pos[:n_each]
    if not pos or not zeros:
        return [], []
    k = min(len(pos), len(zeros))  # balance positives and negatives
    if mode == "close":
        pos_vecs = [_kmer_vec(r.sequence) for r in pos]
        scored = []
        for r in zeros:
            v = _kmer_vec(r.sequence)
            scored.append((max(_cosine_sparse(v, pv) for pv in pos_vecs), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        negs = [r for _, r in scored[:k]]
    else:
        negs = rng.sample(zeros, k)
    sel = pos + negs
    rng.shuffle(sel)
    mid = len(sel) // 2
    print(f"  selected {len(pos)} pos + {len(negs)} neg ({mode}) -> "
          f"train={mid}, eval={len(sel)-mid}")
    return sel[:mid], sel[mid:]


def build_description_payload(feature_id: int, train: list[dr.ProteinRecord]) -> dict:
    return {
        "feature_id": feature_id,
        "n_training_proteins": len(train),
        "proteins": [dr.build_protein_example(r, include_activation=True) for r in train],
    }


def build_prediction_payload(desc_obj: dict, eval_: list[dr.ProteinRecord]) -> dict:
    return {
        "feature_description": desc_obj.get("description", ""),
        "feature_summary": desc_obj.get("summary", ""),
        "n_proteins": len(eval_),
        "proteins": [dr.build_protein_example(r, include_activation=False) for r in eval_],
    }


def score_predictions(eval_: list[dr.ProteinRecord], pred_text: str) -> tuple[float, int]:
    """Return (pearson_r, n_matched). NaN r if unparseable / no variance."""
    try:
        preds = parse_json_response(pred_text)
        if not isinstance(preds, list):
            return float("nan"), 0
    except Exception:
        return float("nan"), 0
    by_entry = {}
    for p in preds:
        try:
            by_entry[p["entry"]] = int(p["predicted_bin"])
        except Exception:
            continue
    actual, predicted = [], []
    for r in eval_:
        if r.entry in by_entry:
            actual.append(dr.bin_to_int(r.bin))
            predicted.append(by_entry[r.entry])
    # A constant predictor (e.g. a mismatched label that predicts "no
    # activation" everywhere) has zero correlation, not an undefined one:
    # report 0.0 so the null aggregates cleanly. Only a degenerate eval set
    # (no variance in measured bins) is a true NaN.
    if len(predicted) >= 2 and len(set(predicted)) == 1 and len(set(actual)) > 1:
        return 0.0, len(actual)
    return dr.pearson(predicted, actual), len(actual)


# ---------------------------------------------------------------------------
# Connectivity / model-id check
# ---------------------------------------------------------------------------


def run_check(provider_names: list[str]) -> int:
    load_env()
    for name in provider_names:
        p = PROVIDERS[name]
        print(f"\n=== {name} (configured model: {p.model}) ===")
        try:
            client = make_client(p)
        except Exception as e:
            print(f"  CLIENT ERROR: {e}")
            continue
        try:
            models = client.models.list()
            ids = [m.id for m in models.data]
            print(f"  {len(ids)} models available.")
            # show models matching the configured family
            stem = p.model.split("-")[0]
            hits = [m for m in ids if stem in m or p.model in m]
            print(f"  matching '{stem}': {sorted(hits)[:25]}")
            print(f"  configured model present: {p.model in ids}")
        except Exception as e:
            print(f"  models.list() failed ({e}); trying a 1-token ping...")
            try:
                txt, usage, dt = call_provider(
                    client, replace(p, out_token_budget=16),
                    "Reply with the single token OK.",
                    {"ping": True},
                )
                print(f"  ping ok in {dt:.1f}s, usage={usage}, text={txt[:40]!r}")
            except Exception as e2:
                print(f"  PING FAILED: {e2}")
    return 0


# ---------------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------------

# Default pilot feature set (full crosscoder), annotated with exp-04 F1 + role.
DEFAULT_FEATURES = {
    1322: "kinase domain (F1 0.99, concentrated)",
    4785: "peptidase S1 (F1 0.94, concentrated)",
    5531: "globin (F1 0.95, concentrated)",
    1339: "disulfide bond (F1 0.76, concentrated/structural)",
    7966: "N-glycosylation (F1 0.76, concentrated)",
    3285: "Cys identity (F1 0.75, causal in steering)",
    6251: "zinc finger (F1 0.69, mid)",
    4189: "SH3 (F1 0.62, mid)",
    4437: "helix (F1 0.25, distributed -> expect low r)",
    4227: "beta strand (F1 0.25, distributed -> expect low r)",
}

# Baseline (random-init ProtT5) crosscoder — the null. Any live features work;
# their descriptions should NOT predict held-out activations (low r).
BASELINE_DIR = (
    REPO_ROOT
    / "model_checkpoints"
    / "crosscoder_l8192_k32_bs512_baseline_2026-05-09_11-50-43"
    / "final_epoch_0_step_2519836"
)


def process_feature(embedder, wrapper, max_examples, fid, label, tag, device,
                    args, rng, clients, provider_names, rows, feature_state) -> None:
    """Run the describe->predict->Pearson loop for one feature across providers.
    Appends result rows; stores train/eval/desc under feature_state[f'{tag}:{fid}']."""
    print(f"\n{'='*70}\n[{tag}] FEATURE {fid} — {label}")
    seed_entries = max_examples.get(fid, [])
    records = dr.sample_proteins(args.n_neg, args.max_length, args.seed, seed_entries=seed_entries)
    ok = encode_feature_on_records(embedder, wrapper, records, fid, device, args.batch_size)
    if not ok:
        print("  feature is dead (rescale<=0) — skipping.")
        return
    n_fired = sum(1 for r in records if r.max_act > 0)
    print(f"  fired on {n_fired}/{len(records)} pool proteins")
    train, eval_ = select_with_negatives(records, rng, args.negatives)
    if len(train) < 3 or len(eval_) < 3:
        print("  not enough spread for train/eval — skipping.")
        return
    if len(eval_) > args.max_eval:
        eval_ = eval_[: args.max_eval]
    dr.fetch_uniprot_metadata(train + eval_)
    desc_payload = build_description_payload(fid, train)
    key = f"{tag}:{fid}"
    feature_state[key] = {"train": train, "eval": eval_, "desc": {}}
    for n in provider_names:
        p = PROVIDERS[n]
        client = clients[n]
        try:
            dtext, dusage, ddt = call_provider(client, p, SYSTEM_DESCRIPTION, desc_payload, force_json=True)
            desc_obj = parse_json_response(dtext)
            if not isinstance(desc_obj, dict):
                raise ValueError("description not a JSON object")
        except Exception as e:
            print(f"  [{n}] description FAILED: {e}")
            rows.append({"crosscoder": tag, "feature": fid, "label": label, "provider": n,
                         "stage": "description", "error": str(e)[:200]})
            continue
        feature_state[key]["desc"][n] = desc_obj
        pred_payload = build_prediction_payload(desc_obj, eval_)
        try:
            ptext, pusage, pdt = call_provider(client, p, SYSTEM_PREDICTION, pred_payload)
        except Exception as e:
            print(f"  [{n}] prediction FAILED: {e}")
            rows.append({"crosscoder": tag, "feature": fid, "label": label, "provider": n,
                         "stage": "prediction", "error": str(e)[:200]})
            continue
        r_val, n_matched = score_predictions(eval_, ptext)
        tin = dusage["input_tokens"] + pusage["input_tokens"]
        tout = dusage["output_tokens"] + pusage["output_tokens"]
        c = cost_usd(p, {"input_tokens": tin, "output_tokens": tout})
        print(f"  [{n}] r={r_val:.3f}  matched={n_matched}/{len(eval_)}  "
              f"in={tin} out={tout}  ${c:.4f}  ({ddt+pdt:.1f}s)  "
              f"summary={desc_obj.get('summary','')[:70]!r}")
        rows.append({"crosscoder": tag, "feature": fid, "label": label, "provider": n,
                     "stage": "ok", "pearson_r": round(r_val, 4), "n_eval": len(eval_),
                     "n_matched": n_matched, "in_tokens": tin, "out_tokens": tout,
                     "cost_usd": round(c, 5), "latency_s": round(ddt + pdt, 1),
                     "summary": desc_obj.get("summary", "")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="List models / verify auth, then exit.")
    ap.add_argument("--features", type=str, default="", help="Comma-separated feature ids. Default: built-in pilot set.")
    ap.add_argument("--providers", type=str, default="deepseek,anthropic",
                    help="deepseek = production choice; anthropic (Opus) kept as a quality reference.")
    ap.add_argument("--n-neg", type=int, default=250, help="Candidate negative/spread proteins per feature (pool to draw negatives from).")
    ap.add_argument("--negatives", choices=["close", "random"], default="close",
                    help="close = composition-matched hard negatives (neutralizes the residue-composition shortcut); random = original recipe.")
    ap.add_argument("--max-length", type=int, default=500)
    ap.add_argument("--max-eval", type=int, default=25, help="Cap eval proteins sent to the LLM (token control).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mismatch-pairs", type=str, default="", help="e.g. '1322:4785,1339:4437' describe A -> predict B.")
    ap.add_argument("--baseline-features", type=str, default="15,22,96,157,161",
                    help="Random-init baseline crosscoder feature ids to run as the null. Empty to skip.")
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--out", type=Path, default=Path("/tmp/pilot_autointerp"))
    args = ap.parse_args()

    provider_names = [s.strip() for s in args.providers.split(",") if s.strip()]
    for n in provider_names:
        if n not in PROVIDERS:
            print(f"Unknown provider {n!r}. Known: {list(PROVIDERS)}")
            return 1

    if args.check:
        return run_check(provider_names)

    load_env()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.features.strip():
        features = {int(x): "(cli)" for x in args.features.split(",") if x.strip()}
    else:
        features = dict(DEFAULT_FEATURES)

    seeds_by_feature = load_max_examples()
    clients = {n: make_client(PROVIDERS[n]) for n in provider_names}

    device = dr.get_local_device()
    print(f"Device: {device}")
    print("Loading ProtT5 + crosscoder (once)...")
    from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
    from interplm.sae.inference import load_sae

    embedder = ProtT5CrosscoderEmbedder(device=device)
    wrapper = load_sae(dr.CHECKPOINT_DIR, model_name="ae_normalized.pt", device=device)

    rows: list[dict[str, Any]] = []
    # Cache built (train, eval, desc per provider) keyed "tag:fid" so the
    # mismatch null can reuse full-crosscoder descriptions.
    feature_state: dict[str, dict[str, Any]] = {}

    # ---- full crosscoder: real features ----
    for fid, label in features.items():
        process_feature(embedder, wrapper, seeds_by_feature, fid, label, "full",
                        device, args, rng, clients, provider_names, rows, feature_state)

    # ---- mismatched-label null: describe A, predict B's held-out bins ----
    full_fids = [int(k.split(":")[1]) for k in feature_state if k.startswith("full:")]
    pairs: list[tuple[int, int]] = []
    if args.mismatch_pairs.strip():
        for tok in args.mismatch_pairs.split(","):
            a, b = tok.split(":")
            pairs.append((int(a), int(b)))
    else:
        for i, a in enumerate(full_fids):
            b = full_fids[(i + 1) % len(full_fids)]
            if a != b:
                pairs.append((a, b))

    print(f"\n{'='*70}\nMISMATCHED-LABEL NULL ({len(pairs)} pairs)")
    for a, b in pairs:
        ka, kb = f"full:{a}", f"full:{b}"
        if ka not in feature_state or kb not in feature_state:
            continue
        eval_b = feature_state[kb]["eval"]
        for n in provider_names:
            desc_a = feature_state[ka]["desc"].get(n)
            if not desc_a:
                continue
            p = PROVIDERS[n]
            pred_payload = build_prediction_payload(desc_a, eval_b)
            try:
                ptext, pusage, pdt = call_provider(clients[n], p, SYSTEM_PREDICTION, pred_payload)
            except Exception as e:
                print(f"  [{n}] {a}->{b} prediction FAILED: {e}")
                continue
            r_val, n_matched = score_predictions(eval_b, ptext)
            print(f"  [{n}] describe f{a} -> predict f{b}: r={r_val:.3f} (matched {n_matched}/{len(eval_b)})")
            rows.append({"crosscoder": "full", "feature": f"{a}->{b}", "label": "mismatch_null",
                         "provider": n, "stage": "mismatch", "pearson_r": round(r_val, 4),
                         "n_eval": len(eval_b), "n_matched": n_matched})

    # ---- baseline (random-init) crosscoder null ----
    base_ids = [int(x) for x in args.baseline_features.split(",") if x.strip()]
    if base_ids:
        print(f"\n{'='*70}\nBASELINE (random-init) CROSSCODER NULL — {len(base_ids)} features")
        from interplm.sae.inference import load_sae as _load_sae
        base_wrapper = _load_sae(args.baseline_dir, model_name="ae_normalized.pt", device=device)
        base_max = load_yaml_max_examples(args.baseline_dir / "Per_feature_max_examples.yaml")
        for fid in base_ids:
            process_feature(embedder, base_wrapper, base_max, fid, "baseline-crosscoder",
                            "baseline", device, args, rng, clients, provider_names, rows, feature_state)

    # ---- write + summarize ----
    import csv
    out_csv = args.out / "pilot_results.csv"
    cols = ["crosscoder", "feature", "label", "provider", "stage", "pearson_r", "n_eval",
            "n_matched", "in_tokens", "out_tokens", "cost_usd", "latency_s", "summary", "error"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"\nWrote {out_csv}")

    def _median_r(provider, stage, crosscoder=None):
        vals = [r["pearson_r"] for r in rows
                if r.get("provider") == provider and r.get("stage") == stage
                and (crosscoder is None or r.get("crosscoder") == crosscoder)
                and isinstance(r.get("pearson_r"), float) and not np.isnan(r["pearson_r"])]
        return (np.median(vals), len(vals)) if vals else (float("nan"), 0)

    print("\n=== SUMMARY (median Pearson r) ===")
    print(f"  {'provider':<10} {'full-real':>12} {'mismatch':>12} {'baseline':>12}  {'$/feat (full)':>14}")
    for n in provider_names:
        rr, nr = _median_r(n, "ok", "full")
        mr, nm = _median_r(n, "mismatch")
        br, nb = _median_r(n, "ok", "baseline")
        costs = [r["cost_usd"] for r in rows if r.get("provider") == n and r.get("stage") == "ok"
                 and r.get("crosscoder") == "full"]
        cpf = np.mean(costs) if costs else float("nan")
        print(f"  {n:<10} {rr:>9.3f}(n{nr}) {mr:>9.3f}(n{nm}) {br:>9.3f}(n{nb})  ${cpf:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
