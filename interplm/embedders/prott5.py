"""ProtT5 Crosscoder embedder for InterPLM."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from interplm.embedders.base import BaseEmbedder
from interplm.utils import get_device


class ProtT5CrosscoderEmbedder(BaseEmbedder):
    """ProtT5 embedder that statically extracts all 24 residual streams for Crosscoder evaluation.

    This embedder ignores the `layer` argument. When extract_embeddings is called, it always
    forwards through all 24 blocks of ProtT5, registers hooks, extracts the activations,
    and stacks them into the [Batch, M, P, D] format that the CrosscoderDictionaryWrapper expects.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        device: Optional[str] = None,
        max_length: int = 2000,
    ):
        if device is None:
            device = get_device()

        super().__init__(model_name, device)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        # Crosscoder expectation: P = 24
        self.num_hookpoints = 24
        self.load_model()

    def load_model(self) -> None:
        """Load ProtT5 model and tokenizer from HuggingFace."""
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )

        # Load in mixed precision if optimal
        self.model = T5EncoderModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: int = -1,  # Ignored for Crosscoders
        batch_size: int = 4,
        return_contacts: bool = False,
    ) -> torch.Tensor:
        """Extract multi-layer embeddings from ProtT5 for Crosscoder usage.

        Args:
            sequences: List of protein sequences
            layer: Ignored.
            batch_size: Batch size for processing
            return_contacts: Not used.

        Returns:
            Tensor of shape (total_tokens, M=1, P=24, D=1024)
        """
        all_embeddings = []

        num_batches = (len(sequences) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(sequences), batch_size)
        if num_batches > 1:
            batch_iterator = tqdm(
                batch_iterator, desc="Processing ProtT5 batches", total=num_batches
            )

        for i in batch_iterator:
            batch_sequences = sequences[i : i + batch_size]

            # Pre-processing (Regex replace UZOB -> X, add spaces)
            processed_seqs = [
                " ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_sequences
            ]

            # Tokenize sequence (ProtT5 adds </s> token at the end)
            inputs = self.tokenizer(
                processed_seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Set up Hooks
            cache: Dict[int, torch.Tensor] = {}
            hooks = []

            def get_hook(layer_idx: int):
                def hook(module: torch.nn.Module, input: Any, output: Any):
                    # T5 blocks return a tuple (hidden_states, [attention_outputs])
                    if isinstance(output, tuple):
                        cache[layer_idx] = output[0].detach()
                    else:
                        cache[layer_idx] = output.detach()

                return hook

            # Extract representations for all 24 blocks
            for layer_idx in range(self.num_hookpoints):
                target_module = self.model.encoder.block[layer_idx]
                hooks.append(target_module.register_forward_hook(get_hook(layer_idx)))

            # Forward pass
            try:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            finally:
                for h in hooks:
                    h.remove()

            # Now `cache` contains the (batch_size, seq_len+padding, 1024) outputs for each of the 24 layers.
            # We must strip the padding AND the EOS `</s>` token for each individual sequence.

            for seq_idx, raw_seq in enumerate(batch_sequences):
                # The amino acid sequence length (strips spaces and padding EOS)
                actual_len = len(raw_seq)

                # Gather the 24 layer vectors for this specific sequence
                seq_layers = []
                for layer_idx in range(self.num_hookpoints):
                    layer_tensor = cache[layer_idx]
                    # ProtT5 output corresponds to tokens. Slice up to actual_len to exclude </s>
                    valid_aa_tensor = (
                        layer_tensor[seq_idx, :actual_len, :].detach().cpu()
                    )
                    seq_layers.append(valid_aa_tensor)

                # Stack layers into (actual_len, P=24, D=1024)
                stacked_seq_layers = torch.stack(seq_layers, dim=1)

                # Reshape to (actual_len, M=1, P=24, D=1024) to emulate `[Batch, M, P, D]` Crosscoder standard
                final_shape = stacked_seq_layers.unsqueeze(1)
                all_embeddings.append(final_shape)

        # Concatenate across the amino-acid token dimension
        if len(all_embeddings) == 0:
            return torch.empty(0, 1, 24, self.model.config.d_model)

        full_tensor = torch.cat(all_embeddings, dim=0)
        return full_tensor

    def extract_embeddings_with_boundaries(
        self,
        sequences: List[str],
        layer: int = -1,
        batch_size: int = 8,
    ) -> Dict[str, Union[torch.Tensor, List[Tuple[int, int]]]]:
        """Overrides default to prevent passing list over all tokens manually."""
        embeddings = self.extract_embeddings(sequences, layer, batch_size)

        boundaries = []
        current_pos = 0
        for sequence in sequences:
            seq_len = len(sequence)
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len

        return {"embeddings": embeddings, "boundaries": boundaries}

    def extract_embeddings_multiple_layers(
        self, sequences, layers, batch_size=8, shuffle=False
    ):
        """Crosscoder automatically handles all layers; just alias to single run."""
        res = self.extract_embeddings(sequences, -1, batch_size)
        if shuffle:
            perm = torch.randperm(res.size(0))
            res = res[perm]
        return {-1: res}

    def embed_single_sequence(self, sequence: str, layer: int = -1) -> torch.Tensor:
        embeddings = self.extract_embeddings([sequence], layer, batch_size=1)
        return embeddings

    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int = -1,
        output_path: Optional[Path] = None,
        batch_size: int = 8,
    ):
        # Read FASTA file
        sequences = []
        with open(fasta_path, "r") as f:
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))

        # ProtT5 needs lots of VRAM
        effective_batch_size = 4 if self.device != "cpu" else batch_size
        embeddings = self.extract_embeddings(sequences, layer, effective_batch_size)

        if output_path:
            output_path = Path(output_path)
            if not str(output_path).endswith(".pt"):
                output_path = output_path.with_suffix(".pt")
            torch.save(embeddings, output_path)
            return None
        return embeddings

    def embed_fasta_file_multiple_layers(
        self, fasta_path, layers, output_dir=None, batch_size=8, shuffle=False
    ):
        """Legacy pass-through. Emulates standard embedders but saves crosscoder struct."""
        result = self.embed_fasta_file(fasta_path, -1, None, batch_size)
        if shuffle:
            perm = torch.randperm(result.size(0))
            result = result[perm]

        if output_dir:
            output_dir = Path(output_dir)
            import yaml

            # Just pretend it's layer 'crosscoder'
            layer_dir = output_dir / "layer_crosscoder"
            layer_dir.mkdir(parents=True, exist_ok=True)
            shard_dir = layer_dir / fasta_path.stem
            shard_dir.mkdir(parents=True, exist_ok=True)

            output_path = shard_dir / "activations.pt"
            torch.save(result, output_path)

            metadata = {
                "model": "ProtT5 Crosscoder",
                "layer": "all",
                "d_model": int(result.shape[3]),
                "total_tokens": int(result.shape[0]),
                "dtype": "float32",
            }
            with open(shard_dir / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)
            return None
        else:
            return {-1: result}

    def get_embedding_dim(self, layer: int) -> int:
        return self.model.config.d_model if self.model else 1024

    @property
    def available_layers(self) -> List[int]:
        return [-1]

    @property
    def max_sequence_length(self) -> int:
        return self.max_length

    def tokenize(self, sequences: List[str]) -> Dict:
        pass
