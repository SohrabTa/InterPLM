"""Diagnostic: characterize DeepSeek V4-Pro failure modes on the autointerp
description + prediction calls, WITHOUT loading ProtT5 (API-only, safe to run
alongside the pilot).

It rebuilds a realistic description prompt from real UniProt metadata for known
feature-1322 (kinase) activators — the exact feature whose DeepSeek description
call failed in the pilot — then calls deepseek-v4-pro under several configs,
repeated R times each, and reports for every call:
  finish_reason | content length | reasoning_content length |
  completion_tokens | reasoning_tokens | JSON-parse OK?

This tells us definitively whether the failures are: empty content, truncation
(finish_reason=length), reasoning-only output, and whether json_mode / bigger
budget / temperature>0 fixes it (critical: temperature 0 would make a
retry-on-empty repeat the same failure deterministically).

Usage:
  uv run --with openai python interplm/llm/diag_deepseek.py --repeats 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from interplm.llm import dryrun as dr
from interplm.llm.prompts import SYSTEM_DESCRIPTION, SYSTEM_PREDICTION, parse_json_response

REPO_ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
ENV_PATH = REPO_ROOT / "repos" / "sparse-crosscoders-prott5" / ".env"
MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"


def load_key() -> str:
    load_dotenv(ENV_PATH)
    import os
    if not os.environ.get("DEEPSEEK_API_KEY") and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    return key


def build_real_payloads(n_high: int = 8, n_zero: int = 6):
    """Reconstruct a description payload (and a prediction payload) for feature
    1322's known kinase activators + some zero proteins, with plausible faked
    activation fields (we only need realistic *metadata* bulk to reproduce the
    prompt size/shape that triggered the failure)."""
    df = pd.read_csv(dr.PROTEINS_TSV, sep="\t")
    highs = dr.KNOWN_TOP_ACTIVATORS[1322][:n_high]
    high_rows = df[df["Entry"].isin(highs)]
    zero_rows = df[~df["Entry"].isin(highs)].sample(n=n_zero, random_state=1)

    def make(entry_rows, is_high):
        recs = []
        for _, row in entry_rows.iterrows():
            rec = dr.ProteinRecord(
                entry=str(row["Entry"]), sequence=str(row["Sequence"]),
                length=int(row["Length"]), local_meta=row.to_dict(),
            )
            import numpy as np
            acts = np.zeros(rec.length)
            if is_high:
                # a handful of "activated residues" near the middle
                mid = rec.length // 2
                for i in range(max(0, mid - 3), min(rec.length, mid + 3)):
                    acts[i] = 0.9
                rec.max_act = 0.95
                rec.bin = (0.9, 1.0)
            else:
                rec.max_act = 0.0
                rec.bin = (0.0, 0.0)
            rec.per_residue_acts = acts
            recs.append(rec)
        return recs

    train = make(high_rows, True) + make(zero_rows.iloc[: n_zero // 2], False)
    eval_ = make(zero_rows.iloc[n_zero // 2:], False) + make(high_rows.iloc[: n_high // 2], True)
    print(f"Fetching UniProt metadata for {len(train)+len(eval_)} proteins...")
    dr.fetch_uniprot_metadata(train + eval_)

    desc_payload = {
        "feature_id": 1322,
        "n_training_proteins": len(train),
        "proteins": [dr.build_protein_example(r, include_activation=True) for r in train],
    }
    # a stand-in description for the prediction-path test
    pred_payload = {
        "feature_description": "Fires on the catalytic/activation-loop region of "
                               "serine/threonine and tyrosine protein kinases.",
        "feature_summary": "Protein kinase catalytic domain detector.",
        "n_proteins": len(eval_),
        "proteins": [dr.build_protein_example(r, include_activation=False) for r in eval_],
    }
    return desc_payload, pred_payload


def call(client, system_text, user_payload, max_tokens, json_mode, temperature):
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    dt = time.time() - t0
    ch = resp.choices[0]
    content = ch.message.content or ""
    reasoning = getattr(ch.message, "reasoning_content", None) or ""
    u = resp.usage
    rtoks = None
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        rtoks = getattr(details, "reasoning_tokens", None)
    ok = True
    try:
        parse_json_response(content)
    except Exception:
        ok = False
    return {
        "finish": ch.finish_reason, "content_len": len(content.strip()),
        "reasoning_len": len(reasoning), "compl_toks": u.completion_tokens,
        "reasoning_toks": rtoks, "parse_ok": ok, "dt": round(dt, 1),
        "content_head": content.strip()[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    load_key()
    from openai import OpenAI
    client = OpenAI(api_key=__import__("os").environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)

    desc_payload, pred_payload = build_real_payloads()

    # (system, payload, max_tokens, json_mode, temperature, label)
    configs = [
        (SYSTEM_DESCRIPTION, desc_payload, 4000, False, 0.0, "DESC no-json mt4000 t0  (repro old)"),
        (SYSTEM_DESCRIPTION, desc_payload, 8000, False, 0.0, "DESC no-json mt8000 t0  (budget only)"),
        (SYSTEM_DESCRIPTION, desc_payload, 8000, True,  0.0, "DESC json    mt8000 t0  (proposed fix)"),
        (SYSTEM_DESCRIPTION, desc_payload, 8000, False, 1.0, "DESC no-json mt8000 t1  (retry w/ temp)"),
        (SYSTEM_PREDICTION,  pred_payload, 8000, False, 0.0, "PRED no-json mt8000 t0  (array path)"),
    ]

    print(f"\nModel={MODEL}  repeats={args.repeats}\n")
    for system_text, payload, mt, jm, temp, label in configs:
        fails = 0
        recs = []
        for i in range(args.repeats):
            try:
                r = call(client, system_text, payload, mt, jm, temp)
            except Exception as e:
                r = {"finish": "EXC", "content_len": 0, "reasoning_len": 0,
                     "compl_toks": 0, "reasoning_toks": None, "parse_ok": False,
                     "dt": 0.0, "content_head": f"EXC: {str(e)[:90]}"}
            recs.append(r)
            if not r["parse_ok"]:
                fails += 1
            print(f"  [{label}] #{i}: finish={r['finish']:<8} content={r['content_len']:<5} "
                  f"reason_len={r['reasoning_len']:<6} compl_tok={r['compl_toks']:<5} "
                  f"reason_tok={r['reasoning_toks']}  parse_ok={r['parse_ok']}  ({r['dt']}s)")
            if not r["parse_ok"]:
                print(f"        head={r['content_head']!r}")
        print(f"  => {label}: {fails}/{args.repeats} FAILED\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
