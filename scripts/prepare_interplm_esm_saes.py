#!/usr/bin/env python
"""Put InterPLM's released ESM-2-650M SAEs into the directory layout our pipeline reads.

Roadmap PP-10a. The normalize stage loads a dictionary with load_sae(sae_dir), which reads
<sae_dir>/config.yaml and <sae_dir>/ae.pt (interplm/sae/inference.py). The HF release has neither
name, so this script makes one directory per layer:

    <out_root>/layer_<L>/ae.pt         a byte copy of layer_<L>/ae_unnormalized.pt
    <out_root>/layer_<L>/config.yaml   interplm/sae/migration/dummy_config_esm2-650m.yaml, which
                                       selects ReLUSAE (10,240 x 1,280), as load_sae_from_hf does
    <out_root>/layer_<L>/SOURCE.json   repo, revision, file name and md5

scripts/encode_activations_esm.py --sae_root <out_root> then encodes with the same files that the
normalize stage loads. The revision is fixed, so a later push to the HF repo changes nothing here.

Example:
    python scripts/prepare_interplm_esm_saes.py --out_root .../model_checkpoints/interplm_esm2_650m
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download

SAE_REPO = "Elana/InterPLM-esm2-650m"
SAE_REVISION = "5121c4c7f3ad0b5fbe0f3b9a457969192bb9912f"
SAE_LAYERS = [1, 9, 18, 24, 30, 33]
DUMMY_CONFIG = Path(__file__).resolve().parents[1] / "interplm/sae/migration/dummy_config_esm2-650m.yaml"


def prepare(out_root: Path, layers: List[int] = SAE_LAYERS):
    for layer in layers:
        filename = f"layer_{layer}/ae_unnormalized.pt"
        src = Path(hf_hub_download(repo_id=SAE_REPO, filename=filename, revision=SAE_REVISION))
        dst = Path(out_root) / f"layer_{layer}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst / "ae.pt")
        shutil.copyfile(DUMMY_CONFIG, dst / "config.yaml")
        md5 = hashlib.md5((dst / "ae.pt").read_bytes()).hexdigest()
        if md5 != hashlib.md5(src.read_bytes()).hexdigest():
            raise RuntimeError(f"layer {layer}: copy differs from the download")
        source = {"repo": SAE_REPO, "revision": SAE_REVISION, "file": filename, "md5": md5}
        (dst / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
        print(f"layer {layer:2d}: {dst} md5 {md5}")


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare)
