from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
import shutil
import numpy as np
from tqdm import tqdm

from interplm.sae.dictionary import Dictionary
from interplm.train.configs import TrainingRunConfig
from interplm.utils import get_device

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# InterPLM paper models available on HuggingFace (Elana/InterPLM-{model})
INTERPLM_HF_MODELS = ["esm2-8m", "esm2-650m"]
INTERPLM_HF_LAYERS = {
    "esm2-8m": [1, 2, 3, 4, 5, 6],
    "esm2-650m": [1, 9, 18, 24, 30, 33],
}


def load_sae(
    model_dir: Union[str, Path], model_name: str = "ae.pt", device: Optional[str] = None
) -> Dictionary:
    """
    Load a pretrained SAE model in inference mode.

    :param model_path: Path to the saved model state dictionary
    :param device: Target device for model computation ('cpu', 'cuda', etc.).
                  If None, automatically determines the best available device
    :return: Loaded SAE model in eval mode with gradients disabled

    This function:
    1. Loads the model state from disk
    2. Reconstructs the SAE architecture based on the saved config
    3. Moves the model to the specified device
    4. Sets the model to evaluation mode
    5. Disables gradient computation for all parameters
    """
    if device is None:
        device = get_device()
    model_dir = Path(model_dir)

    config_path = model_dir / "config.yaml"
    import yaml

    with open(config_path, "r") as f:
        config_data = yaml.unsafe_load(f)

    # Check for Crosscoder (flat config with n_hookpoints)
    if "n_hookpoints" in config_data:
        import sys

        # Assuming crosscode is a sibling of InterPLM
        interplm_root = Path(__file__).parents[2]
        crosscode_path = interplm_root.parent / "crosscode"
        if str(crosscode_path.absolute()) not in sys.path:
            sys.path.append(str(crosscode_path.absolute()))

        # Directly scaffold the crosscoder to handle InterPLM-style filenames (ae.pt, config.yaml)
        # as crosscode's internal load() is rigid about model.pt/model_cfg.yaml
        try:
            from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
            from crosscode.interplm_adapter.crosscoder_dictionary import CrosscoderDictionaryWrapper
            
            # Scaffolding using the config data we already loaded
            autoencoder = ModelHookpointAcausalCrosscoder._scaffold_from_cfg(config_data)
            wrapper = CrosscoderDictionaryWrapper(autoencoder)
            
            # Load weights from ae.pt or ae_normalized.pt
            state_dict = torch.load(model_dir / model_name, map_location=device, weights_only=True)
            
            if any(k.startswith('crosscoder.') for k in state_dict.keys()):
                wrapper.load_state_dict(state_dict, strict=False)
            else:
                autoencoder.load_state_dict(state_dict, strict=False)
                
            wrapper.to(device)
            wrapper.eval()
            
            # Return wrapped for InterPLM
            return wrapper
            
        except Exception as e:
            raise RuntimeError(f"Failed to scaffold Crosscoder from {model_dir}: {e}")

    config = TrainingRunConfig.from_yaml(config_path)
    ae_cls = config.trainer_cfg.trainer_cls().dictionary_cls()

    # Initialize and configure the model
    autoencoder = ae_cls.from_pretrained(model_dir / model_name, device=device)

    return autoencoder


def load_sae_from_hf(
    plm_model: str, plm_layer: str, unnormalized: bool = False
) -> Dictionary:
    """
    Load a pretrained PLM SAE from Hugging Face.

    :param plm_model: ESM2 model name [options = "esm2-8m" or "esm2-650m"]
    :param plm_layer: Layer to use for encoding [options = 1-6 if esm_model = "esm2-8m", or 1,9,18,24,30,33 if esm_model = "esm2-650m"]
    :return: Loaded SAE model
    """

    # Convert to lowercase and replace underscores with hyphens
    plm_model = plm_model.lower().replace("_", "-")
    plm_layer = int(plm_layer)

    if plm_model not in INTERPLM_HF_MODELS:
        raise ValueError(
            f"Invalid ESM model: {plm_model}, options: {INTERPLM_HF_MODELS}"
        )
    if plm_layer not in INTERPLM_HF_LAYERS[plm_model]:
        raise ValueError(
            f"Invalid ESM layer for {plm_model}: {plm_layer}, options: {INTERPLM_HF_LAYERS[plm_model]}"
        )

    # Download the SAE weights from HuggingFace
    weights_path = hf_hub_download(
        repo_id=f"Elana/InterPLM-{plm_model}",
        filename=f"layer_{plm_layer}/ae_{'unnormalized' if unnormalized else 'normalized'}.pt",
    )
    weights_path = Path(weights_path)

    # Also create a dummy config file
    # if it has no config file, create a dummy one (copy from interplm.sae.migration.dummy_config.yaml)
    if not (weights_path.parent / "config.yaml").exists():
        shutil.copy(
            Path(__file__).parent / "migration" / f"dummy_config_{plm_model}.yaml",
            weights_path.parent / "config.yaml",
        )

    # Load the SAE model
    sae = load_sae(
        weights_path.parent,
        model_name=f"ae_{'unnormalized' if unnormalized else 'normalized'}.pt",
    )

    return sae


def load_legacy_sae(
    sae_dir: Union[str, Path],
    plm_model: str,
    model_name: str = "ae.pt",
    device: Optional[str] = None,
) -> Dictionary:
    """
    Load a legacy pretrained SAE model (pre-v1.0.0 format) in inference mode.

    :param sae_dir: Path to the saved model state dictionary
    :param model_name: Name of the model file
    :param device: Device to perform computations on ('cpu', 'cuda', etc.)
    :return: Loaded SAE model in eval mode with gradients disabled
    """
    if device is None:
        device = get_device()
    sae_dir = Path(sae_dir)

    # confirm plm_model is esm2-8m or esm2-650m
    if plm_model not in INTERPLM_HF_MODELS:
        raise ValueError(
            f"Invalid ESM model: {plm_model}, options: {INTERPLM_HF_MODELS}"
        )

    if not (sae_dir / "config.yaml").exists():
        shutil.copy(
            Path(__file__).parent / "migration" / f"dummy_config_{plm_model}.yaml",
            sae_dir / "config.yaml",
        )

    # Load the SAE model
    sae = load_sae(sae_dir, model_name=model_name, device=device)

    return sae


def get_sae_feats_in_batches(
    sae: Dictionary,
    device: str,
    aa_embds: np.ndarray | Tensor,
    chunk_size: int,
    feat_list: Optional[List[int]] = None,
    scale_to_100_in_int8: bool = False,
    normalize_features: bool = False,
) -> Tensor:
    """
    Process large embedding arrays in chunks to generate SAE features.

    :param sae: Trained SAE model
    :param device: Device to perform computations on ('cpu', 'cuda', etc.)
    :param aa_embds: NumPy array of amino acid embeddings to process
    :param chunk_size: Number of embeddings to process in each batch
    :param feat_list: List of feature indices to encode. If None, uses all features
    :return: Tensor containing encoded features for all input embeddings

    This function:
    1. Defaults to using all features if feat_list is None
    2. Processes embeddings in batches to manage memory usage
    3. Shows progress using tqdm
    4. Concatenates all processed batches into a single tensor
    """
    # Use all features if none specified
    if feat_list is None:
        feat_list = list(range(sae.dict_size))

    # Convert input to tensor on specified device
    try:
        if torch.is_tensor(aa_embds):
            aa_embds = aa_embds.to(device)
        elif isinstance(aa_embds, list) and all(torch.is_tensor(x) for x in aa_embds):
            aa_embds = torch.stack(aa_embds, dim=0).to(device)
        else:
            aa_embds = torch.tensor(aa_embds, device=device)
    except Exception as e:
        print(f"Error converting input to tensor: {e}")
        raise e

    all_features = []

    # Process in chunks with progress bar
    for i in tqdm(range(0, len(aa_embds), chunk_size), desc="Encoding features"):
        chunk = aa_embds[i : i + chunk_size]
        features = sae.encode_feat_subset(
            chunk, feat_list, normalize_features=normalize_features
        )
        if scale_to_100_in_int8:
            features = (features * 100).to(torch.int8)
        all_features.append(features)

    # Combine all processed chunks
    all_features = torch.vstack(all_features)
    return all_features


def split_up_feature_list(total_features, max_feature_chunk_size: int = 2560):
    feature_chunk_size = min(max_feature_chunk_size, total_features)
    num_chunks = int(np.ceil(total_features / feature_chunk_size))

    return np.array_split(range(total_features), num_chunks)
