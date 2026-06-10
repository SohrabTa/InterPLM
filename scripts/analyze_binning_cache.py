"""Analyze a collect_step1.py Phase-A binning cache (bin_assignments.yaml).

Reports the sanity/quality metrics for the per-protein activation binning that
feeds the LLM-autointerp pipeline: feature count, bin structure, cap compliance,
the zero-activation (close-negative) pool sizes, and how many features would
survive Phase B's selection filter (>= MIN_PROTEINS_IN_TOP_THREE in the top-3
bins) i.e. are describable by the LLM.

Inputs : a bin_assignments.yaml written by collect_step1.py Phase A.
Outputs: prints metrics to stdout (no files written).
Repro  : pure analysis, deterministic. Backs the numbers reported in
         documentation/experiments/06-llm-autointerp.md (AuxK-fix binning run).

Example:
    python scripts/analyze_binning_cache.py \
        data/llm_autointerp/full_auxfix/cache/bin_assignments.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

# Phase B's filter (mirror of collect_step1.MIN_PROTEINS_IN_TOP_THREE).
MIN_PROTEINS_IN_TOP_THREE = 20
TOP3 = ["0.7,0.8", "0.8,0.9", "0.9,1.0"]
BINS = ["0.0,0.0"] + [f"{round(i*0.1,1)},{round((i+1)*0.1,1)}" for i in range(10)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path, help="path to bin_assignments.yaml")
    args = ap.parse_args()

    t0 = time.time()
    d = yaml.unsafe_load(args.cache.open())
    print(f"loaded {args.cache} in {time.time()-t0:.0f}s")

    nf = len(d)
    print(f"features: {nf}")
    print(
        "all features have 11 bins:",
        all(len(v) == 11 for v in d.values()),
        "| bin keys correct:",
        all(set(v.keys()) == set(BINS) for v in d.values()),
    )

    act_bins = [b for b in BINS if b != "0.0,0.0"]
    max_act = max(len(v[b]) for v in d.values() for b in act_bins)
    max_zero = max(len(v["0.0,0.0"]) for v in d.values())
    print(f"max activation-bin size: {max_act} | max zero-bin size: {max_zero}")

    zsz = np.array([len(v["0.0,0.0"]) for v in d.values()])
    print(
        f"zero-bin sizes: mean {zsz.mean():.0f}, median {int(np.median(zsz))}, "
        f"at cap(500): {(zsz>=500).sum()} feats, <100: {(zsz<100).sum()} feats"
    )

    t3 = np.array([sum(len(v[b]) for b in TOP3) for v in d.values()])
    surv = int((t3 >= MIN_PROTEINS_IN_TOP_THREE).sum())
    print(
        f"Phase B survivors (>={MIN_PROTEINS_IN_TOP_THREE} in top-3 bins): "
        f"{surv} / {nf} ({100*surv/nf:.0f}%) | top-3 count mean {t3.mean():.1f}, "
        f"median {int(np.median(t3))}, ==0: {int((t3==0).sum())}"
    )

    total_cells = sum(1 for v in d.values() for b in v.values() if len(b) > 0)
    print(f"non-empty (feature, bin) cells: {total_cells}")
    print("aggregate proteins per bin:")
    for b in BINS:
        print(f"  {b:>10}: {sum(len(v[b]) for v in d.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
