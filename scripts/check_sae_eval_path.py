#!/usr/bin/env python
"""Check the PP-06a eval changes: the multi-dictionary encode and the converter --layer option.

Checks (CPU, small data, the real ProtT5):
  1. Encode regression. The new encode_activations.py with only the crosscoder writes the same
     store as the version on `main` (acts arrays equal, meta.json equal).
  2. Converter regression. Without --layer, the new converter writes the same sidecar and the
     same converted checkpoint as the version on `main`.
  3. Converter --layer. A random one-layer SAE at layer 16 converts, the sidecar and the config
     record layer 16, and a --layer that disagrees with the path stops the run.
  4. One-layer encode. One ProtT5 pass encodes the crosscoder and the SAE. The crosscoder store
     equals check 1, and the SAE store equals a direct encode of the layer-16 slice.

Inputs:
  --crosscoder_dir  JumpReLU crosscoder for the encode (ae.pt + config.yaml)
  --batchtopk_dir   BatchTopK crosscoder for the converter regression (model.pt + config.yaml)
  --metadata_dir    processed_annotations of the eval set (shard_<i>/protein_data.tsv)
  --calib_fasta     one sequence per line, the calibration input of the converter
  --old_encode, --old_convert   the `main` versions of the two scripts
                    (git show main:<path> > <file>)
Output: work_dir/ (stores, converted checkpoints) and a PASS/FAIL line for each check.
Seed: torch.manual_seed(0) for the random SAE.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
NEW_ENCODE = HERE / "encode_activations.py"
NEW_CONVERT = HERE.parents[1] / "sparse-crosscoders-prott5" / "convert_batchtopk_to_jumprelu.py"
LAYER = 16
SAE_NAME = f"sae0_encoder.blocks.{LAYER}.hook_resid_post"
FOLDED_FACTOR = 0.025328  # crosscoder norm factor of layer 16 (exp 12, section 5)


def run(*args) -> None:
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def report(name: str, ok: bool) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {name}", flush=True)
    return ok


def same_store(a: Path, b: Path, shard: int = 0, ignore=("sae_dir",)) -> bool:
    from interplm.analysis.activation_store import load_shard_activations

    acts_a, meta_a = load_shard_activations(a, shard)
    acts_b, meta_b = load_shard_activations(b, shard)
    arrays = (
        all(
            (getattr(acts_a, f) == getattr(acts_b, f)).all() and getattr(acts_a, f).dtype == getattr(acts_b, f).dtype
            for f in ("data", "indices", "indptr")
        )
        and acts_a.shape == acts_b.shape
    )
    meta = {k: v for k, v in meta_a.items() if k not in ignore} == {k: v for k, v in meta_b.items() if k not in ignore}
    return bool(arrays and meta)


def make_random_sae(out: Path) -> Path:
    from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder
    from crosscode.models.activations.topk import BatchTopkActivation

    torch.manual_seed(0)
    sae = ModelHookpointAcausalCrosscoder(
        n_models=1,
        n_hookpoints=1,
        d_model=1024,
        n_latents=256,
        activation_fn=BatchTopkActivation(k_per_example=8),
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.1),
    )
    folded = sae.with_folded_scaling_factors(torch.tensor([[FOLDED_FACTOR]]))
    folded.save(out)
    (out / "config.yaml").write_text((out / "model_cfg.yaml").read_text())
    return out


def assemble(conv_out: Path) -> Path:
    """The layout of submit_convert_jumprelu.sh: ae.pt, model.pt, config.yaml in one dir."""
    conv = conv_out / "converted"
    for name in ("ae.pt", "model.pt"):
        (conv_out / name).write_bytes((conv / "model.pt").read_bytes())
    (conv_out / "config.yaml").write_text((conv / "config.yaml").read_text())
    return conv_out


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in (
        "crosscoder_dir",
        "batchtopk_dir",
        "metadata_dir",
        "calib_fasta",
        "old_encode",
        "old_convert",
        "work_dir",
    ):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--n_proteins", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=8)
    args = ap.parse_args()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    results = []

    meta_dir = work / "metadata"
    (meta_dir / "shard_0").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metadata_dir / "shard_0" / "protein_data.tsv", sep="\t")
    df.head(args.n_proteins).to_csv(meta_dir / "shard_0" / "protein_data.tsv", sep="\t", index=False)
    encode_args = ["--metadata_dir", meta_dir, "--shards", "0", "--batch_size", "8"]

    # 1. Encode regression, crosscoder only.
    run(args.old_encode, "--sae_dir", args.crosscoder_dir, "--output_dir", work / "store_old", *encode_args)
    run(NEW_ENCODE, "--sae_dir", args.crosscoder_dir, "--output_dir", work / "store_new", *encode_args)
    results.append(
        report("encode, crosscoder only: new == main", same_store(work / "store_old", work / "store_new", ignore=()))
    )

    # 3. Converter --layer on a random one-layer SAE.
    sae_dir = make_random_sae(work / SAE_NAME / "final")
    conv_args = [
        "--checkpoint",
        "model.pt",
        "--mode",
        "global",
        "--batch_tokens",
        "512",
        "--max_batches",
        str(args.max_batches),
        "--device",
        "cpu",
        "--write_converted",
    ]
    sae_conv = work / f"{SAE_NAME}_jumprelu"
    run(
        NEW_CONVERT,
        "--crosscoder_dir",
        sae_dir,
        "--fasta",
        args.calib_fasta,
        "--k",
        "8",
        "--layer",
        str(LAYER),
        "--out_dir",
        sae_conv,
        *conv_args,
    )
    assemble(sae_conv)
    sidecar = torch.load(sae_conv / "jumprelu_threshold.pt", weights_only=False)
    cfg = yaml.safe_load((sae_conv / "config.yaml").read_text())
    print(
        f"SAE theta {sidecar['theta']:.6g}, mean_L0_check {sidecar['mean_L0_check']:.2f}, layer {sidecar.get('layer')}"
    )
    results.append(
        report(
            "convert --layer: theta finite, layer 16 in sidecar and config",
            bool(torch.isfinite(torch.tensor(sidecar["theta"])))
            and sidecar.get("layer") == LAYER
            and cfg.get("layer") == LAYER,
        )
    )
    wrong = subprocess.run(
        [
            sys.executable,
            str(NEW_CONVERT),
            "--crosscoder_dir",
            str(sae_dir),
            "--activations",
            str(sae_conv / "_calib_acts"),
            "--layer",
            str(LAYER - 1),
            "--out_dir",
            str(work / "wrong"),
        ],
        capture_output=True,
        text=True,
    )
    results.append(
        report(
            "convert --layer 15 on a layer-16 path stops", wrong.returncode != 0 and "names layer 16" in wrong.stderr
        )
    )

    # 2. Converter regression without --layer, on the same calibration tensors.
    for tag, script in (("old", args.old_convert), ("new", NEW_CONVERT)):
        run(
            script,
            "--crosscoder_dir",
            args.batchtopk_dir,
            "--activations",
            sae_conv / "_calib_acts",
            "--k",
            "32",
            "--out_dir",
            work / f"cc_conv_{tag}",
            *conv_args,
        )
    old_s = torch.load(work / "cc_conv_old" / "jumprelu_threshold.pt", weights_only=False)
    new_s = torch.load(work / "cc_conv_new" / "jumprelu_threshold.pt", weights_only=False)
    same_sidecar = old_s.keys() == new_s.keys() and all(
        torch.equal(old_s[k], new_s[k]) if torch.is_tensor(old_s[k]) else old_s[k] == new_s[k] for k in old_s
    )
    old_sd = torch.load(work / "cc_conv_old" / "converted" / "model.pt", weights_only=True)
    new_sd = torch.load(work / "cc_conv_new" / "converted" / "model.pt", weights_only=True)
    same_model = old_sd.keys() == new_sd.keys() and all(torch.equal(old_sd[k], new_sd[k]) for k in old_sd)
    same_cfg = (work / "cc_conv_old/converted/config.yaml").read_bytes() == (
        work / "cc_conv_new/converted/config.yaml"
    ).read_bytes()
    print(f"crosscoder theta old {old_s['theta']:.6g} new {new_s['theta']:.6g}")
    results.append(
        report(
            "convert without --layer: sidecar, config.yaml and model.pt equal to main",
            same_sidecar and same_model and same_cfg,
        )
    )

    # 4. One pass for the crosscoder and the SAE.
    run(
        NEW_ENCODE,
        "--sae_dir",
        args.crosscoder_dir,
        "--output_dir",
        work / "store_multi_cc",
        "--extra_sae_dirs",
        sae_conv,
        "--extra_output_dirs",
        work / "store_multi_sae",
        *encode_args,
    )
    results.append(
        report(
            "encode, two dictionaries: crosscoder store == crosscoder-only store",
            same_store(work / "store_new", work / "store_multi_cc"),
        )
    )

    from interplm.analysis.activation_store import load_shard_activations
    from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
    from interplm.sae.inference import load_sae

    sae = load_sae(sae_conv, model_name="ae.pt", device="cpu")
    embedder = ProtT5CrosscoderEmbedder(device="cpu")
    seqs = pd.read_csv(meta_dir / "shard_0" / "protein_data.tsv", sep="\t")["Sequence"].tolist()
    direct = []
    for i in range(0, len(seqs), 8):
        embds = embedder.extract_embeddings(seqs[i : i + 8], batch_size=8)
        direct.append(sae.encode(embds[:, :, LAYER - 1 : LAYER, :], normalize_features=False))
    direct = torch.cat(direct).numpy()
    stored, meta = load_shard_activations(work / "store_multi_sae", 0)
    print(f"SAE store: {meta['n_residues']} residues, mean L0 {meta['mean_l0']:.2f}, layer {meta.get('layer')}")
    results.append(
        report(
            "encode, SAE store == direct encode of the layer-16 slice",
            stored.shape == direct.shape and bool((stored.toarray() == direct).all()) and meta.get("layer") == LAYER,
        )
    )

    print(json.dumps({"passed": sum(results), "checks": len(results)}))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
