"""Step 2 of the LLM annotation pipeline: generate descriptions + predictions
from the Step-1 parquet using a local LLM (or any OpenAI-compatible endpoint).

For every feature in the input parquet, we:
  1. Build a description prompt from the "train" rows (proteins + activation
     bins + activated residues + UniProt metadata).
  2. Send it to the configured LLM; parse the JSON {description, summary}.
  3. Build a prediction prompt from the "eval" rows (same proteins minus
     activation info) + the description from step 2.
  4. Send it; parse the JSON list of predicted bins.
  5. Compute Pearson r between predicted and measured bins.

Results are written as a parquet keyed by (endpoint_name, feature_id) so
multiple LLMs can be compared side-by-side from one run.

Designed for vLLM (OpenAI-compatible /v1/chat/completions). Works against
any number of endpoints concurrently by passing
``--endpoints "llama=http://host:8000/v1,deepseek=http://host:8001/v1"``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interplm.llm.prompts import (
    SYSTEM_DESCRIPTION,
    SYSTEM_PREDICTION,
    parse_json_response,
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def summarize_uniprot(meta_json: str) -> dict[str, Any]:
    """Mirror of interplm.llm.dryrun.summarize_uniprot_meta, but reads from
    the JSON-blob column in the Step-1 parquet."""
    try:
        meta = json.loads(meta_json) if meta_json else {}
    except json.JSONDecodeError:
        meta = {}
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

    for c in meta.get("comments", []):
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

    gos = []
    for r in meta.get("uniProtKBCrossReferences", []):
        if r.get("database") == "GO":
            term = next(
                (
                    p.get("value")
                    for p in r.get("properties", [])
                    if p.get("key") == "GoTerm"
                ),
                None,
            )
            if term:
                gos.append(term)
    if gos:
        out["go_terms"] = gos[:30]

    features = meta.get("features", [])
    by_type: dict[str, list[dict[str, Any]]] = {}
    for f in features:
        t = f.get("type")
        loc = f.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        desc = f.get("description") or ""
        if t and start is not None and end is not None:
            by_type.setdefault(t, []).append(
                {"start": start, "end": end, "desc": desc}
            )
    if by_type:
        out["features"] = by_type
    return out


def build_protein_payload(row: pd.Series, include_activation: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entry": row["protein_entry"],
        "metadata": summarize_uniprot(row["uniprot_meta_json"]),
    }
    if include_activation:
        payload["activation_bin"] = int(row["bin"])
        payload["max_activation"] = round(float(row["max_activation_observed"]), 3)
        residues = json.loads(row["activated_residues_json"]) if row[
            "activated_residues_json"
        ] else []
        # Sort by position for readability
        residues.sort(key=lambda r: r["position"])
        payload["activated_residues"] = [
            {
                "position": int(r["position"]) + 1,  # 1-indexed
                "residue": r["residue"],
                "activation": round(float(r["activation"]), 3),
            }
            for r in residues
        ]
    return payload


def build_description_payload(rows: pd.DataFrame) -> dict[str, Any]:
    train = rows[rows["split"] == "train"]
    return {
        "feature_id": int(train.iloc[0]["feature_id"]),
        "n_training_proteins": len(train),
        "proteins": [build_protein_payload(r, True) for _, r in train.iterrows()],
    }


def build_prediction_payload(
    rows: pd.DataFrame, description: str, summary: str
) -> dict[str, Any]:
    eval_ = rows[rows["split"] == "eval"]
    return {
        "feature_description": description,
        "feature_summary": summary,
        "n_proteins": len(eval_),
        "proteins": [build_protein_payload(r, False) for _, r in eval_.iterrows()],
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible client (async)
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    text: str
    usage: dict[str, int]
    latency_s: float
    error: str | None = None


@dataclass
class FeatureResult:
    endpoint_name: str
    feature_id: int
    description: str
    summary: str
    description_usage: dict[str, int]
    description_latency: float
    prediction_text: str
    prediction_usage: dict[str, int]
    prediction_latency: float
    pearson_r: float
    n_eval_with_prediction: int
    predicted_bins: list[int] = field(default_factory=list)
    measured_bins: list[int] = field(default_factory=list)
    error: str | None = None


async def call_chat(
    client: Any,
    model: str,
    system_text: str,
    user_payload: dict[str, Any],
    max_tokens: int,
) -> CallResult:
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        dt = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
        return CallResult(text=text, usage=usage, latency_s=dt)
    except Exception as e:
        return CallResult(
            text="", usage={"input_tokens": 0, "output_tokens": 0},
            latency_s=time.time() - t0, error=str(e),
        )


async def process_one_feature(
    client: Any,
    endpoint_name: str,
    model: str,
    feature_id: int,
    rows: pd.DataFrame,
    semaphore: asyncio.Semaphore,
) -> FeatureResult:
    async with semaphore:
        # Description call
        desc_payload = build_description_payload(rows)
        desc_call = await call_chat(
            client, model, SYSTEM_DESCRIPTION, desc_payload, max_tokens=1500
        )
        if desc_call.error:
            return FeatureResult(
                endpoint_name=endpoint_name,
                feature_id=feature_id,
                description="",
                summary="",
                description_usage=desc_call.usage,
                description_latency=desc_call.latency_s,
                prediction_text="",
                prediction_usage={"input_tokens": 0, "output_tokens": 0},
                prediction_latency=0.0,
                pearson_r=float("nan"),
                n_eval_with_prediction=0,
                error=f"description: {desc_call.error}",
            )
        try:
            desc_obj = parse_json_response(desc_call.text)
            description = str(desc_obj.get("description", ""))
            summary = str(desc_obj.get("summary", ""))
        except Exception as e:
            return FeatureResult(
                endpoint_name=endpoint_name,
                feature_id=feature_id,
                description=desc_call.text[:500],
                summary="",
                description_usage=desc_call.usage,
                description_latency=desc_call.latency_s,
                prediction_text="",
                prediction_usage={"input_tokens": 0, "output_tokens": 0},
                prediction_latency=0.0,
                pearson_r=float("nan"),
                n_eval_with_prediction=0,
                error=f"description JSON parse: {e}",
            )

        # Prediction call
        pred_payload = build_prediction_payload(rows, description, summary)
        pred_call = await call_chat(
            client, model, SYSTEM_PREDICTION, pred_payload, max_tokens=1500
        )
        if pred_call.error:
            return FeatureResult(
                endpoint_name=endpoint_name,
                feature_id=feature_id,
                description=description,
                summary=summary,
                description_usage=desc_call.usage,
                description_latency=desc_call.latency_s,
                prediction_text="",
                prediction_usage=pred_call.usage,
                prediction_latency=pred_call.latency_s,
                pearson_r=float("nan"),
                n_eval_with_prediction=0,
                error=f"prediction: {pred_call.error}",
            )
        try:
            preds = parse_json_response(pred_call.text)
            if not isinstance(preds, list):
                raise ValueError("prediction response not a list")
        except Exception as e:
            return FeatureResult(
                endpoint_name=endpoint_name,
                feature_id=feature_id,
                description=description,
                summary=summary,
                description_usage=desc_call.usage,
                description_latency=desc_call.latency_s,
                prediction_text=pred_call.text[:500],
                prediction_usage=pred_call.usage,
                prediction_latency=pred_call.latency_s,
                pearson_r=float("nan"),
                n_eval_with_prediction=0,
                error=f"prediction JSON parse: {e}",
            )

        # Compute Pearson r
        eval_rows = rows[rows["split"] == "eval"]
        measured_by_entry = {
            r["protein_entry"]: int(r["bin"]) for _, r in eval_rows.iterrows()
        }
        predicted_pairs: list[tuple[int, int]] = []
        for p in preds:
            entry = p.get("entry")
            pred = p.get("predicted_bin")
            if entry in measured_by_entry and pred is not None:
                predicted_pairs.append((int(pred), measured_by_entry[entry]))
        if len(predicted_pairs) < 2:
            r = float("nan")
        else:
            xs = np.array([p[0] for p in predicted_pairs], dtype=float)
            ys = np.array([p[1] for p in predicted_pairs], dtype=float)
            r = (
                float(np.corrcoef(xs, ys)[0, 1])
                if xs.std() > 0 and ys.std() > 0
                else float("nan")
            )

        return FeatureResult(
            endpoint_name=endpoint_name,
            feature_id=feature_id,
            description=description,
            summary=summary,
            description_usage=desc_call.usage,
            description_latency=desc_call.latency_s,
            prediction_text=pred_call.text,
            prediction_usage=pred_call.usage,
            prediction_latency=pred_call.latency_s,
            pearson_r=r,
            n_eval_with_prediction=len(predicted_pairs),
            predicted_bins=[p[0] for p in predicted_pairs],
            measured_bins=[p[1] for p in predicted_pairs],
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_endpoint(
    endpoint_name: str,
    base_url: str,
    api_key: str,
    model: str,
    df: pd.DataFrame,
    concurrency: int,
) -> list[FeatureResult]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    feature_groups = list(df.groupby("feature_id"))
    print(
        f"[{endpoint_name}] {len(feature_groups)} features, "
        f"concurrency={concurrency}, model={model}"
    )

    tasks = [
        process_one_feature(client, endpoint_name, model, int(fid), rows, sem)
        for fid, rows in feature_groups
    ]

    results: list[FeatureResult] = []
    t0 = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        res = await coro
        results.append(res)
        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            print(
                f"[{endpoint_name}] {i + 1}/{len(tasks)} done "
                f"({time.time() - t0:.0f}s elapsed)"
            )
    return results


def feature_results_to_df(results: list[FeatureResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "endpoint": r.endpoint_name,
                "feature_id": r.feature_id,
                "description": r.description,
                "summary": r.summary,
                "pearson_r": r.pearson_r,
                "n_eval_with_prediction": r.n_eval_with_prediction,
                "predicted_bins_json": json.dumps(r.predicted_bins),
                "measured_bins_json": json.dumps(r.measured_bins),
                "description_input_tokens": r.description_usage.get("input_tokens", 0),
                "description_output_tokens": r.description_usage.get(
                    "output_tokens", 0
                ),
                "description_latency_s": r.description_latency,
                "prediction_input_tokens": r.prediction_usage.get("input_tokens", 0),
                "prediction_output_tokens": r.prediction_usage.get("output_tokens", 0),
                "prediction_latency_s": r.prediction_latency,
                "prediction_raw_text": r.prediction_text,
                "error": r.error,
            }
            for r in results
        ]
    )


def parse_endpoints(arg: str) -> list[tuple[str, str]]:
    """Parse name1=url1,name2=url2 → [(name1, url1), ...]."""
    out = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"endpoint spec missing name=: {piece}")
        name, url = piece.split("=", 1)
        out.append((name.strip(), url.strip()))
    return out


async def amain(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.input_parquet)
    if args.max_features:
        keep = sorted(df["feature_id"].unique())[: args.max_features]
        df = df[df["feature_id"].isin(keep)]
    if args.feature_ids:
        wanted = {int(x) for x in args.feature_ids.split(",")}
        df = df[df["feature_id"].isin(wanted)]
    print(f"Processing {df['feature_id'].nunique()} features")

    endpoints = parse_endpoints(args.endpoints)
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[FeatureResult] = []
    for name, url in endpoints:
        # Allow per-endpoint model override via env var, else use --model
        model = os.environ.get(f"VLLM_MODEL_{name.upper()}", args.model)
        results = await run_endpoint(
            endpoint_name=name,
            base_url=url,
            api_key=api_key,
            model=model,
            df=df,
            concurrency=args.concurrency,
        )
        endpoint_df = feature_results_to_df(results)
        endpoint_df.to_parquet(args.output_dir / f"results_{name}.parquet", index=False)
        all_results.extend(results)

        valid = endpoint_df.dropna(subset=["pearson_r"])
        n_err = endpoint_df["error"].notna().sum()
        print(
            f"\n[{name}] summary:\n"
            f"  features processed: {len(endpoint_df)}\n"
            f"  errors: {n_err}\n"
            f"  median Pearson r (over {len(valid)} valid): "
            f"{valid['pearson_r'].median() if len(valid) else 'NA'}\n"
            f"  mean description latency: "
            f"{endpoint_df['description_latency_s'].mean():.1f}s\n"
            f"  mean prediction latency: "
            f"{endpoint_df['prediction_latency_s'].mean():.1f}s"
        )

    combined = feature_results_to_df(all_results)
    combined.to_parquet(args.output_dir / "results_combined.parquet", index=False)
    print(f"\nWrote combined results to {args.output_dir / 'results_combined.parquet'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to Per_feature_llm_input.parquet from Step 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write results_{endpoint}.parquet files.",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        required=True,
        help='Comma-separated "name=url" pairs. Example: '
        '"llama=http://localhost:8000/v1,deepseek=http://localhost:8001/v1"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name to send to /v1/chat/completions. Override per-endpoint "
        "with env vars VLLM_MODEL_<NAME>.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max in-flight requests per endpoint.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Take only the first N features from the parquet (0 = all).",
    )
    parser.add_argument(
        "--feature-ids",
        type=str,
        default="",
        help="Comma-separated feature ids to restrict to. Overrides --max-features.",
    )
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
