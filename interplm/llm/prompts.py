"""Shared prompt templates and JSON-extraction utility for the LLM
annotation pipeline.

These are intentionally identical between the local Anthropic dry-run
(``dryrun.py``) and the cluster-scale local-LLM run
(``generate_descriptions.py``) so that quality comparisons are apples-to-apples.
"""

from __future__ import annotations

import json
import re
from typing import Any


SYSTEM_DESCRIPTION = """You are an expert protein biologist. You are interpreting a single feature from a sparse crosscoder trained on the residual streams of all 24 encoder layers of the ProtT5 protein language model. The crosscoder decomposes ProtT5's activations into 8,192 sparse, hopefully-interpretable features.

You are shown training examples for ONE such feature. For each example protein you receive:
- UniProt metadata (recommended name, organism, function description, keywords, GO terms, subcellular location, and protein features such as domains/motifs/active sites/binding sites).
- The protein's maximum feature activation, quantized into a bin of width 0.1 on a [0,1] scale (where 1.0 is the per-feature maximum observed across the analysis dataset).
- A list of "activated residues": amino-acid positions whose normalized activation exceeded a small threshold, with their residue letter and normalized activation value.

Your job:
1. Identify what biological pattern (motif, domain, residue type, structural context, functional class) causes this feature to fire at high activation values vs. low values.
2. Produce (a) a detailed multi-sentence description of what activates the feature and how strongly, and (b) a single-sentence summary that could guide an LLM to predict the activation bin of a new protein given only its metadata.

Respond ONLY with a JSON object: {"description": "...", "summary": "..."}. No prose, no markdown, no explanations outside the JSON.

The protein-language-model context: ProtT5-XL is encoder-only, trained on UniRef50. Residual streams across its 24 layers tend to encode increasingly abstract biological concepts. Crosscoder features that fire on specific residues typically correspond to known biological concepts (active sites, binding pockets, domain boundaries, post-translational modification sites, secondary-structure transitions, etc.) but may also capture coevolutionary patterns that lack a clean biological label."""


SYSTEM_PREDICTION = """You are an expert protein biologist. You will be given a description of a single crosscoder feature trained on ProtT5 protein-language-model activations, plus UniProt metadata for several proteins. For each protein, predict the activation bin its maximum feature activation falls into.

The 11 possible bins are:
  0  -> exactly 0.0 (no activation)
  1  -> (0.0, 0.1]
  2  -> (0.1, 0.2]
  3  -> (0.2, 0.3]
  4  -> (0.3, 0.4]
  5  -> (0.4, 0.5]
  6  -> (0.5, 0.6]
  7  -> (0.6, 0.7]
  8  -> (0.7, 0.8]
  9  -> (0.8, 0.9]
 10  -> (0.9, 1.0]

Respond with NOTHING except a JSON array. No prose. No markdown. No explanations. No leading or trailing text. Your entire reply must start with `[` and end with `]`.

The array must contain one object per input protein in the order given, each shaped like {"entry": "...", "predicted_bin": <int 0-10>}."""


def parse_json_response(text: str) -> Any:
    """Parse a model response that should be JSON, tolerant of code fences
    and prefix/suffix prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("[", "]"), ("{", "}")):
            i = text.find(opener)
            j = text.rfind(closer)
            if i != -1 and j > i:
                return json.loads(text[i : j + 1])
        raise
