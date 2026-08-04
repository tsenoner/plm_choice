#!/usr/bin/env python3
"""
Unified script for generating protein embeddings using various PLMs.
Combines functionalities for model loading/downloading, FASTA processing,
embedding generation, and HDF5 saving.

--- Ivan infrastructure (2026-03-19) ---
Added --random_init flag to generate embeddings from randomly initialized
(non-pretrained) models. This captures the architectural prior (attention
patterns, layer norms, positional encoding) WITHOUT learned biology.
Conceptually different from random_embeddings.py which generates i.i.d.
noise vectors — a randomly initialized model produces *contextual*
embeddings where nearby residues still influence each other through the
architecture, just without any training signal.

Use case: null hypothesis baseline for "does pretraining help?" in the
pLM Choice paper revision.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Optional

import h5py
import torch
import numpy as np
from tqdm import tqdm
from pyfaidx import Fasta

# --- Hugging Face Transformers Imports ---
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EsmModel,
    T5EncoderModel,
    T5Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from huggingface_hub import login as hf_login

# --- Native ESM (EvolutionaryScale) Imports ---
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig


# --------------------------------------------------------------------------- #
#                            MODEL CONFIGURATION
# --------------------------------------------------------------------------- #
# Define a type for more readable model configurations
ModelConfig = Dict[str, Any]

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # --- ESM models via HuggingFace Transformers ---
    "esm2_8m": {
        "hf_id": "facebook/esm2_t6_8M_UR50D",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "esm_transformer",
    },
    "esm2_35m": {
        "hf_id": "facebook/esm2_t12_35M_UR50D",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "esm_transformer",
    },
    "esm2_150m": {
        "hf_id": "facebook/esm2_t30_150M_UR50D",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "esm_transformer",
    },
    "esm2_650m": {
        "hf_id": "facebook/esm2_t33_650M_UR50D",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "esm_transformer",
    },
    "esm2_3b": {
        "hf_id": "facebook/esm2_t36_3B_UR50D",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "esm_transformer",
    },
    # --- Native ESM3/ESMC models (requires 'esm' package) ---
    "esm3_open": {
        "hf_id": "esm3-open",
        "loader": "native_esm",
        "model_class": ESM3,
        "tokenizer_class": None,
        "family_key": "native_esm3",
        "notes": "Uses EvolutionaryScale/esm3-sm-open-v1 if hf_id is 'esm3-open', or specify full ID.",
        "requires_explicit_login": True,
    },
    "esmc_300m": {
        "hf_id": "esmc_300m",
        "loader": "native_esm",
        "model_class": ESMC,
        "tokenizer_class": None,
        "family_key": "native_esmc",
        "requires_explicit_login": True,
    },
    "esmc_600m": {
        "hf_id": "esmc_600m",
        "loader": "native_esm",
        "model_class": ESMC,
        "tokenizer_class": None,
        "family_key": "native_esmc",
        "requires_explicit_login": True,
    },
    # --- Ankh models via HuggingFace Transformers ---
    "ankh_base": {
        "hf_id": "ElnaggarLab/ankh-base",
        "loader": "transformers",
        "model_class": T5EncoderModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "ankh",
    },
    "ankh_large": {
        "hf_id": "ElnaggarLab/ankh-large",
        "loader": "transformers",
        "model_class": T5EncoderModel,
        "tokenizer_class": AutoTokenizer,
        "family_key": "ankh",
    },
    # --- ProtT5 models via HuggingFace Transformers ---
    "prot_t5": {
        "hf_id": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "loader": "transformers",
        "model_class": T5EncoderModel,
        "tokenizer_class": T5Tokenizer,
        "family_key": "prot_t5",
        "load_kwargs": {"torch_dtype": torch.float16},
        "post_load_hook": lambda m: m.half() if hasattr(m, "half") else m,
    },
    "prost_t5": {
        "hf_id": "Rostlab/ProstT5_fp16",
        "loader": "transformers",
        "model_class": T5EncoderModel,
        "tokenizer_class": T5Tokenizer,
        "tokenizer_load_kwargs": {"do_lower_case": False},
        "family_key": "prost_t5",
        "load_kwargs": {"torch_dtype": torch.float16},
        "post_load_hook": lambda m: m.half(),
    },
}

# --------------------------------------------------------------------------- #
#                            UTILITY FUNCTIONS
# --------------------------------------------------------------------------- #


def get_device() -> torch.device:
    """Determines the most appropriate device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for MPS (Apple Silicon GPU)
    # Add this check if you intend to support MPS.
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return torch.device("mps")
    return torch.device("cpu")


def login_to_huggingface(token_path_str: Optional[str] = None):
    """Attempts to log in to Hugging Face Hub, optionally using a token from a specified path."""
    token = None
    token_file = None
    if token_path_str:
        token_file = Path(token_path_str)
    else:
        # Default token locations
        potential_paths = [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ]
        for p_path in potential_paths:
            if p_path.is_file():
                token_file = p_path
                break

    if token_file and token_file.is_file():
        try:
            token = token_file.read_text().strip()
            print(f"ℹ️ Found Hugging Face token file at: {token_file}")
        except Exception as e:
            print(f"⚠️ Could not read token from {token_file}: {e}")
            token = None

    try:
        if token:
            hf_login(token=token)
            print("✓ Hugging Face login successful (used token from file).")
        else:
            print(
                "ℹ️ No token file specified or found in default locations. Attempting default login (e.g., cached session, env var)."
            )
            hf_login()  # Attempts login using env variables or cached token
            print(
                "✓ Hugging Face login successful or already authenticated (default method)."
            )
    except Exception as login_exc:
        print(f"⚠️ Hugging Face login attempt failed: {login_exc}")
        print(
            "   Proceeding. This may fail if the model requires authentication for download."
        )


def read_fasta_sequences(fasta_path: Path) -> List[Tuple[str, str]]:
    """Reads sequences from a FASTA file."""
    sequences = []
    with Fasta(str(fasta_path)) as fasta_data:
        for record in fasta_data:
            sequences.append((record.name, str(record)))
    if not sequences:
        sys.stderr.write(f"WARNING: No sequences found in FASTA file: {fasta_path}\n")
    return sequences


def preprocess_sequence(sequence: str, model_family_key: str) -> str:
    """Applies model-specific preprocessing to a sequence string."""
    # Replace U, Z, O, B with X for most models.
    # Specific models might have other requirements.
    processed_seq = (
        sequence.upper()
        .replace("U", "X")
        .replace("Z", "X")
        .replace("O", "X")
        .replace("B", "X")
    )

    if model_family_key == "prost_t5":
        # Add spaces, then prefix for ProstT5
        spaced_seq = " ".join(list(processed_seq))
        return "<AA2fold> " + spaced_seq
    elif model_family_key == "prot_t5":
        # For regular ProtT5, only add spaces
        return " ".join(list(processed_seq))
    return processed_seq


# --------------------------------------------------------------------------- #
#                        MODEL LOADING & EMBEDDING
# --------------------------------------------------------------------------- #


def _reinit_weights(module: torch.nn.Module, seed: int = 42) -> None:
    """Re-initialise a model in place, per module type.

    Only needed for the native-ESM loader, which has no ``from_config`` path —
    for HuggingFace models, constructing from an ``AutoConfig`` already performs
    the architecture's own initialisation and this must NOT be applied on top.

    Crucially this is type-aware. Overwriting *every* parameter with
    ``N(0, 0.02)`` — which is what a naive loop does — sets LayerNorm gains to
    ~0 and biases to ~0.02 instead of 1.0 and 0. Measured on a 4-layer ESM
    config that collapses the hidden-state magnitude 34x and pushes the mean
    residue-residue correlation from +0.686 to +0.960, i.e. every residue ends
    up with nearly the same vector and the mean-pooled embedding degenerates to
    a constant. The point of this baseline is an untrained *architecture*, not a
    broken one, so the scheme below mirrors ``EsmPreTrainedModel._init_weights``:

        Linear / Embedding weight ~ N(0, 0.02);  bias = 0
        LayerNorm weight = 1.0;                  bias = 0
        padding_idx embedding row = 0

    Being type-aware means being *incomplete*: any parameter owned by a module
    that is none of those three keeps the value it was loaded with. This path
    starts from ``from_pretrained``, so an unhandled module leaves PRETRAINED
    weights inside a baseline whose entire premise is that nothing was learned —
    and ESM3/ESMC are exactly the custom-block architectures where that is
    plausible. Any such parameter is therefore reported loudly rather than left
    to pass for an untrained model.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    handled: set[int] = set()

    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02, generator=generator)
            handled.add(id(submodule.weight))
            if submodule.bias is not None:
                torch.nn.init.zeros_(submodule.bias)
                handled.add(id(submodule.bias))
        elif isinstance(submodule, torch.nn.Embedding):
            torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02, generator=generator)
            handled.add(id(submodule.weight))
            if submodule.padding_idx is not None:
                with torch.no_grad():
                    submodule.weight[submodule.padding_idx].fill_(0.0)
        elif isinstance(submodule, torch.nn.LayerNorm):
            # Neither tensor is guaranteed to exist: ESM-3 and ESM-C build their
            # LayerNorms with bias=False, and elementwise_affine=False drops the
            # weight as well. Unguarded, zeros_(None) raises and takes out three
            # of the top four arms on fident and hfsp.
            if submodule.weight is not None:
                torch.nn.init.ones_(submodule.weight)
                handled.add(id(submodule.weight))
            if submodule.bias is not None:
                torch.nn.init.zeros_(submodule.bias)
                handled.add(id(submodule.bias))

    untouched = [name for name, p in module.named_parameters() if id(p) not in handled]
    if untouched:
        preview = ", ".join(untouched[:10])
        suffix = f", ... (+{len(untouched) - 10} more)" if len(untouched) > 10 else ""
        print(
            f"⚠️ --random_init: {len(untouched)} parameter tensor(s) are not covered by "
            f"the Linear/Embedding/LayerNorm re-init and therefore KEEP THEIR "
            f"PRETRAINED VALUES: {preview}{suffix}\n"
            f"   This baseline is only an untrained architecture if that list is empty. "
            f"Extend _reinit_weights before reporting numbers from this model."
        )


def default_output_path(
    fasta_file: Path,
    model_key: str,
    random_init: bool,
    random_seed: int,
) -> Path:
    """Where embeddings go when ``--output_hdf5_file`` is not given.

    The seed is part of the random-init filename because D-6 reports the
    untrained arm as mean±sd over seeds 0/1/2. Without it all three seeds resolve
    to one path, the append-mode writer skips every already-present protein, and
    the run exits 0 having produced a single seed's vectors — published as
    ``sd = 0.000``.
    """
    sanitized = model_key.replace("/", "_")
    if random_init:
        return fasta_file.with_name(f"random_init_{sanitized}_seed{random_seed}.h5")
    return fasta_file.with_name(f"{fasta_file.stem}_{sanitized}.h5")


def masked_mean_pool(
    hidden_states: "torch.Tensor", attention_mask: "torch.Tensor"
) -> "torch.Tensor":
    """Mean-pool ``(B, L, D)`` hidden states over real tokens only.

    Batching is the 25-75x lever on this workload, but a plain ``.mean(dim=1)``
    over a padded batch averages the padding in too, and the error grows the
    shorter a protein is relative to the longest member of its batch. That is a
    silent corruption: the vectors still look plausible.
    """
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    token_counts = mask.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise ValueError(
            "at least one sequence has no unmasked tokens; pooling it would "
            "divide by zero and write NaNs into the HDF5"
        )
    return (hidden_states * mask).sum(dim=1) / token_counts


def content_mask(
    attention_mask: "torch.Tensor", n_lead: int, n_trail: int
) -> "torch.Tensor":
    """Mask selecting real residues only: no padding, no special tokens.

    The unbatched path strips special tokens by slicing ``[1:-1]`` or ``[:-1]``,
    which is correct only because the batch holds one sequence. Under
    right-padding a short sequence's trailing ``</s>`` sits in the middle of its
    row, so ``[:-1]`` would strip a pad token and keep the special token —
    quietly averaging a learned end-of-sequence vector into the protein.

    Both counts are per-family: ESM and ProstT5 drop one leading and one
    trailing token, ProtT5 and Ankh drop only the trailing one.
    """
    if n_lead < 0 or n_trail < 0:
        raise ValueError(f"n_lead/n_trail must be >= 0, got {n_lead}/{n_trail}")

    lengths = attention_mask.sum(dim=1)
    if bool((lengths <= n_lead + n_trail).any()):
        raise ValueError(
            f"at least one sequence has {int(lengths.min())} token(s), which is not "
            f"more than the {n_lead + n_trail} special token(s) being stripped; "
            "pooling it would average nothing"
        )

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0)  # (1, L)
    keep = (positions >= n_lead) & (positions < (lengths.unsqueeze(1) - n_trail))
    return (keep & attention_mask.bool()).to(attention_mask.dtype)


def length_buckets(
    sequences: List[str],
    token_budget: int,
    min_efficiency: float = 0.6,
    slack: int = 16,
) -> List[List[int]]:
    """Group sequence indices into batches of similar length.

    Two separate constraints, because the token budget alone does not express
    what we actually want:

    * ``token_budget`` bounds ``len(batch) * longest_in_batch`` so a batch fits
      in GPU memory;
    * ``min_efficiency`` bounds the share of that rectangle which is real
      residues. A generous budget will otherwise happily put a 10-residue
      protein in the same batch as a 500-residue one — legal, and 73% of the
      compute spent on padding.

    ``slack`` exempts batches whose length spread is only a few tokens, so short
    proteins do not fragment into batches of one over a couple of residues.

    Returns indices into ``sequences``, every index exactly once, so the caller
    can restore the original order.
    """
    if token_budget < 1:
        raise ValueError(f"token_budget must be >= 1, got {token_budget}")
    if not 0.0 < min_efficiency <= 1.0:
        raise ValueError(f"min_efficiency must be in (0, 1], got {min_efficiency}")

    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    batches: List[List[int]] = []
    current: List[int] = []
    current_real = 0

    for idx in order:
        n = len(sequences[idx])
        if not current:
            current, current_real = [idx], n
            continue

        longest = max(len(sequences[i]) for i in current + [idx])
        slots = (len(current) + 1) * longest
        efficiency = (current_real + n) / slots
        spread = longest - min(len(sequences[i]) for i in current + [idx])

        over_budget = slots > token_budget
        too_ragged = efficiency < min_efficiency and spread > slack
        if over_budget or too_ragged:
            batches.append(current)
            current, current_real = [idx], n
        else:
            current.append(idx)
            current_real += n

    if current:
        batches.append(current)
    return batches


def should_apply_post_load_hook(
    post_load_hook: Optional[Callable[[Any], None]], random_init: bool
) -> bool:
    """Whether to run a model's post-load hook (for ProtT5/ProstT5, ``.half()``).

    This deliberately reverses an earlier choice to skip the hook under
    ``--random_init`` "to avoid masking initialization effects with quantization
    noise". That reasoning is defensible in isolation, but the baseline's whole
    claim is ``pretrained - random_init``, and that subtraction is only a
    statement about *weights* if every other axis is pinned. Leaving the hook off
    makes the untrained ProtT5/ProstT5 compute in fp32 while their pretrained
    twins computed in fp16, and that gap lands in the headline number looking
    exactly like a pretraining effect.

    If the precision question is worth asking, it deserves its own arm rather
    than a confound inside this one.
    """
    return post_load_hook is not None


def load_model_and_tokenizer(
    model_key: str,
    config: ModelConfig,
    weights_dir: Optional[Path],
    device: torch.device,
    random_init: bool = False,
    random_seed: int = 42,
) -> Tuple[
    Any, Optional[PreTrainedTokenizer | Any], str
]:  # Model, Tokenizer (or None), FamilyKey
    """
    Loads the specified model and tokenizer.

    If random_init=True, the model architecture is loaded but all weights
    are reset to random values (sigma=0.02). The tokenizer is always loaded
    pretrained — tokenization must be identical to produce comparable
    sequence representations.
    """

    hf_id = config["hf_id"]
    loader = config["loader"]
    model_class: Type[PreTrainedModel | Any] = config["model_class"]
    tokenizer_class: Optional[Type[PreTrainedTokenizer | Any]] = config.get(
        "tokenizer_class"
    )
    tokenizer_load_kwargs = config.get("tokenizer_load_kwargs", {})
    family_key = config["family_key"]
    load_kwargs = config.get("load_kwargs", {})
    post_load_hook: Optional[Callable[[Any], None]] = config.get("post_load_hook")

    actual_cache_dir = None
    if weights_dir:
        actual_cache_dir = weights_dir.expanduser().resolve()
        actual_cache_dir.mkdir(parents=True, exist_ok=True)
        # For transformers, pass cache_dir to from_pretrained.
        # For native ESM, also pass it.
        # We can also set HF_HUB_CACHE, which `run_esm.py` does.
        os.environ["HF_HUB_CACHE"] = str(actual_cache_dir)
        print(f"ℹ️ Using Hugging Face cache directory: {actual_cache_dir}")

    mode_str = "RANDOM INIT" if random_init else "pretrained"
    print(f"→ Loading model '{model_key}' ({hf_id}) [{mode_str}] using {loader} loader...")
    model = None
    tokenizer = None

    try:
        if loader == "transformers":
            if random_init:
                # --- Ivan infrastructure (2026-03-19): random init baseline ---
                # Load architecture config from HuggingFace, then instantiate
                # with random weights. This preserves the model's structure
                # (attention heads, layer norms, positional encoding) but
                # removes all learned representations.
                model_config = AutoConfig.from_pretrained(
                    hf_id, cache_dir=actual_cache_dir
                )
                # Constructing the model from a config already performs the
                # architecture's OWN random initialisation — HuggingFace's
                # PreTrainedModel.post_init() runs _init_weights per module, so
                # Linear/Embedding weights get N(0, initializer_range), biases
                # get 0 and LayerNorm gains get 1.0. Seeding immediately before
                # construction is all that is needed to make it reproducible.
                #
                # Do NOT blanket-overwrite every parameter with N(0, 0.02)
                # afterwards. Measured on a 4-layer ESM config, doing so drives
                # LayerNorm gains from 1.0 to ~0.0004 and biases from 0 to
                # ~0.016, collapsing the hidden-state magnitude 34x (8.0e-01 ->
                # 2.4e-02) and raising the mean residue-residue correlation from
                # +0.686 to +0.960. At that point every residue vector is nearly
                # identical, so the mean-pooled protein embedding is a constant
                # plus noise — the opposite of the contextual
                # architecture-only prior this baseline is meant to provide.
                torch.manual_seed(random_seed)
                model = model_class(model_config)
                print(
                    f"  Initialized {model_key} with random weights "
                    f"(architecture default init, seed={random_seed})"
                )
            else:
                model = model_class.from_pretrained(
                    hf_id, cache_dir=actual_cache_dir, **load_kwargs
                )

            # Tokenizer is ALWAYS loaded pretrained — tokenization must match
            if tokenizer_class:
                tokenizer = tokenizer_class.from_pretrained(
                    hf_id, cache_dir=actual_cache_dir, **tokenizer_load_kwargs
                )

        elif loader == "native_esm":
            # NATIVE_ESM_AVAILABLE check is removed as we assume it's true
            if model_class is None:  # Should not happen if MODEL_CONFIGS is correct
                raise ValueError(
                    f"Model class not defined for native ESM model {model_key}"
                )
            # Native ESM models are loaded via their class directly.
            # The hf_id might be an alias like "esm3-open" or a full one.
            model = model_class.from_pretrained(
                hf_id
            )  # Native ESM handles its own caching via HF_HUB_CACHE
            # Native ESM models typically have built-in .encode() methods, so tokenizer_class is None.

            if random_init:
                # Native ESM has no from_config(), so load pretrained and then
                # re-initialise per module type (see _reinit_weights).
                _reinit_weights(model, seed=random_seed)
                print(f"  Re-initialized {model_key} with random weights (seed={random_seed})")

        else:
            raise ValueError(f"Unknown loader type: {loader}")

        if model is None:
            raise RuntimeError(f"Failed to load model for {model_key}")

        # Applied for random init too, so the untrained arm computes at the same
        # precision as its pretrained twin — see should_apply_post_load_hook.
        if should_apply_post_load_hook(post_load_hook, random_init):
            post_load_hook(model)

        model.to(device).eval()
        print(f"✓ Model '{model_key}' ready on {device.type.upper()} [{mode_str}].")
        return model, tokenizer, family_key

    except Exception as e:
        print(
            f"ERROR: Failed to load model '{model_key}' ({hf_id}). Details:",
            file=sys.stderr,
        )
        print(
            "  Ensure model name is correct, you have internet access, and accepted any licenses on Hugging Face.",
            file=sys.stderr,
        )
        # Removed check for NATIVE_ESM_AVAILABLE as it's assumed true
        print(f"  Original error: {e}", file=sys.stderr)
        sys.exit(1)


def generate_single_embedding(
    model: Any,
    tokenizer: Optional[PreTrainedTokenizer | Any],
    sequence: str,
    family_key: str,
    embedding_type: str,  # "per_protein" or "per_residue"
    device: torch.device,
) -> np.ndarray:
    """Generates embedding for a single preprocessed sequence."""

    if family_key.startswith("native_esm"):
        # NATIVE_ESM_AVAILABLE check removed
        protein = ESMProtein(sequence=sequence)

        if family_key == "native_esm3":
            # Native ESM3 embedding extraction
            if not hasattr(model, "encode") or not hasattr(model, "forward_and_sample"):
                raise AttributeError(
                    "Loaded native ESM3 model does not have expected methods."
                )
            tok = model.encode(protein)  # Tokenizes

            out = model.forward_and_sample(
                tok.to(device), SamplingConfig(return_per_residue_embeddings=True)
            )
            per_res_emb_raw = out.per_residue_embedding.squeeze(0).cpu()  # (L, D)

            # Directly slice to remove BOS/EOS tokens
            cleaned_per_res_emb = per_res_emb_raw[1:-1, :]

        elif family_key == "native_esmc":
            # Native ESMC embedding extraction
            if not hasattr(model, "encode") or not hasattr(model, "logits"):
                raise AttributeError(
                    "Loaded native ESMC model does not have expected methods."
                )
            tok = model.encode(protein)  # Tokenizes

            out = model.logits(
                tok.to(device), LogitsConfig(sequence=True, return_embeddings=True)
            )
            per_res_emb_raw = out.embeddings.squeeze(0).cpu()  # (L, D)

            # Directly slice to remove BOS/EOS tokens
            cleaned_per_res_emb = per_res_emb_raw[1:-1, :]
        else:
            raise ValueError(f"Unknown native ESM family key: {family_key}")

        if embedding_type == "per_protein":
            return cleaned_per_res_emb.mean(dim=0).numpy()
        elif embedding_type == "per_residue":
            return cleaned_per_res_emb.numpy()
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}")

    elif family_key in ["esm_transformer", "ankh", "prot_t5", "prost_t5"]:
        if tokenizer is None:
            raise ValueError(
                f"Tokenizer is required for transformer model family: {family_key}"
            )

        # Common Transformers-based embedding generation
        tokenization_input = sequence
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "add_special_tokens": True,
        }

        if family_key == "ankh":
            tokenization_input = list(sequence)  # e.g. ['M', 'L', 'K']
            tokenizer_kwargs["is_split_into_words"] = True

        inputs = tokenizer(tokenization_input, **tokenizer_kwargs).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        # Correct slicing based on family_key
        token_embeddings_for_avg = None
        per_residue_slice = None

        if family_key == "esm_transformer":
            # Skip <cls> and <eos>
            token_embeddings_for_avg = embeddings[1:-1, :]
            per_residue_slice = embeddings[1:-1, :]
        elif family_key == "prost_t5":
            # Skip <AA2fold> prefix (and potential BOS) and <eos>
            token_embeddings_for_avg = embeddings[1:-1, :]
            per_residue_slice = embeddings[1:-1, :]
        elif family_key == "ankh":
            token_embeddings_for_avg = embeddings[:-1, :]
            per_residue_slice = embeddings[:-1, :]
        elif family_key == "prot_t5":
            token_embeddings_for_avg = embeddings[:-1, :]
            per_residue_slice = embeddings[:-1, :]
        else:
            token_embeddings_for_avg = embeddings
            per_residue_slice = embeddings

        if embedding_type == "per_protein":
            if token_embeddings_for_avg.shape[0] == 0:
                print(
                    f"WARNING: No token embeddings to average for sequence after slicing. Family: {family_key}, Original shape: {embeddings.shape}",
                    file=sys.stderr,
                )
                return np.array([])
            return token_embeddings_for_avg.mean(axis=0)
        elif embedding_type == "per_residue":
            return per_residue_slice
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}")
    else:
        raise ValueError(
            f"Unsupported model family key for embedding generation: {family_key}"
        )


# --------------------------------------------------------------------------- #
#                        MAIN PROCESSING FUNCTION
# --------------------------------------------------------------------------- #


def process_sequences_and_save(
    sequences_to_process: List[Tuple[str, str]],
    model: Any,
    tokenizer: Optional[PreTrainedTokenizer | Any],
    family_key: str,
    embedding_type: str,
    device: torch.device,
    h5_output_path: Path,
    max_seq_len: Optional[int],
    model_key_for_filename: str,  # Used for logging and progress bar description
    random_init: bool = False,
):
    """
    Processes sequences, generates embeddings, and saves them to an HDF5 file.
    Embeddings are saved as top-level datasets in the HDF5 file.

    Append mode lets a long pretrained run resume, which is worth keeping. But
    the "already exists" branch below counts a skip as a success, so under
    ``--random_init`` appending is how a second seed silently produces a file
    holding only the first seed's vectors and still exits 0. Random-init runs
    therefore open exclusively and fail loudly on collision.
    """
    num_successfully_embedded = 0

    # "w-" raises FileExistsError rather than resuming into another seed's file.
    open_mode = "w-" if random_init else "a"
    with h5py.File(h5_output_path, open_mode) as h5_file:
        progress_bar = tqdm(
            sequences_to_process,
            unit="seq",
            desc=f"Embedding ({model_key_for_filename})",
        )

        for header, original_sequence in progress_bar:
            base_header = header.split()[0]

            if base_header in h5_file:
                tqdm.write(
                    f"ℹ️ Embedding for '{base_header}' already exists in {h5_output_path}. Skipping."
                )
                num_successfully_embedded += 1
                continue

            if not original_sequence:
                tqdm.write(f"⚠️ Sequence '{base_header}' is empty. Skipping.")
                continue

            # Determine sequence length for filtering (before ProstT5 prefixing/spacing)
            seq_len_for_check = len(original_sequence)
            if max_seq_len is not None and seq_len_for_check > max_seq_len:
                tqdm.write(
                    f"⚠️ Sequence '{base_header}' (length {seq_len_for_check}) "
                    f"exceeds max length {max_seq_len}. Skipping."
                )
                continue

            progress_bar.set_postfix_str(
                f"Processing: {base_header[:25]}... (len: {seq_len_for_check})"
            )

            try:
                processed_sequence_for_embedding = preprocess_sequence(
                    original_sequence, family_key
                )
                embedding = generate_single_embedding(
                    model,
                    tokenizer,
                    processed_sequence_for_embedding,
                    family_key,
                    embedding_type,
                    device,
                )

                if embedding.size == 0:
                    tqdm.write(
                        f"ERROR: Embedding for '{base_header}' resulted in an empty array. Skipping."
                    )
                    continue

                h5_file.create_dataset(
                    name=base_header, data=embedding.astype(np.float32)
                )
                h5_file.flush()
                num_successfully_embedded += 1

            except Exception as e:
                error_msg = (
                    f"\nERROR processing sequence '{base_header}' (Model: {model_key_for_filename}): "
                    f"{type(e).__name__}: {e}\n"
                )
                sys.stderr.write(error_msg)
                tqdm.write(error_msg.strip())

    return num_successfully_embedded


# --------------------------------------------------------------------------- #
#                                MAIN FUNCTION
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using various PLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta_file", type=Path, help="Path to the input FASTA file.")
    parser.add_argument(
        "model_key",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Key of the model to use (see MODEL_CONFIGS in script).",
    )
    parser.add_argument(
        "--output_hdf5_file",
        type=Path,
        default=None,
        help="Optional path to the HDF5 file where embeddings will be saved. "
        "If not provided, it defaults to '[fasta_filename_stem]_[model_key].h5' "
        "in the same directory as the FASTA file.",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=None,
        help="Optional directory for Hugging Face model cache/weights. "
        "If set, HF_HUB_CACHE will be pointed here.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["per_protein", "per_residue"],
        default="per_protein",
        help="Type of embedding to generate.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2000,
        help="Maximum sequence length. Longer sequences will be skipped. (default: 2000)",
    )
    parser.add_argument(
        "--token_path",
        type=str,
        default=None,
        help="Optional path to Hugging Face token file for login (primarily for models like native ESM).",
    )
    # --- Ivan infrastructure (2026-03-19): random init baseline ---
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Use randomly initialized (non-pretrained) model weights. "
        "Produces contextual embeddings shaped by architecture alone, "
        "without any learned biology. Output files are named "
        "random_init_<model>_seed<N>.h5 — the seed is part of the name because D-6 "
        "reports this arm as mean±sd over seeds 0/1/2, and a shared filename would "
        "make runs 2 and 3 skip every protein and exit 0. "
        "See also random_embeddings.py for simpler i.i.d. random vectors.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for weight initialization when --random_init is used.",
    )

    args = parser.parse_args()

    if not args.fasta_file.is_file():
        print(f"ERROR: FASTA file not found: {args.fasta_file}", file=sys.stderr)
        sys.exit(1)

    model_config = MODEL_CONFIGS.get(args.model_key)
    if not model_config:
        print(
            f"ERROR: Model key '{args.model_key}' not found in MODEL_CONFIGS.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Ivan infrastructure (2026-03-19): random init output naming ---
    # random_init embeddings go to random_init_<model>_seed<N>.h5 to distinguish
    # them from pretrained <model>.h5, from i.i.d. random_<dim>.h5, and — the
    # part that matters for D-6's mean±sd — from each other.
    if args.output_hdf5_file is None:
        output_h5_path = default_output_path(
            args.fasta_file,
            args.model_key,
            random_init=args.random_init,
            random_seed=args.random_seed,
        )
    else:
        output_h5_path = args.output_hdf5_file

    # Ensure the output directory exists
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"ℹ️ Embeddings will be saved to: {output_h5_path}")

    # Removed NATIVE_ESM_AVAILABLE check for model config here

    max_len_to_use = args.max_seq_len
    if max_len_to_use is not None:
        print(f"ℹ️ Using max sequence length: {max_len_to_use}")
    else:
        print("ℹ️ No max sequence length limit applied (beyond inherent model limits).")

    # Conditional login based on model config
    if model_config.get("requires_explicit_login", False):
        print(
            f"ℹ️ Model '{args.model_key}' suggests explicit login. Attempting Hugging Face login..."
        )
        login_to_huggingface(args.token_path)  # Pass token_path here
    else:
        print(
            f"ℹ️ Model '{args.model_key}' does not require explicit script-driven login. Relying on transformers library or cached credentials if needed."
        )

    device = get_device()
    print(f"ℹ️ Selected device: {device.type}")

    model, tokenizer, family_key = load_model_and_tokenizer(
        args.model_key,
        model_config,
        args.weights_dir,
        device,
        random_init=args.random_init,
        random_seed=args.random_seed,
    )

    print(f"Reading sequences from: {args.fasta_file}")
    all_sequences = read_fasta_sequences(args.fasta_file)
    if not all_sequences:
        print("No sequences to process. Exiting.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(all_sequences)} sequences in FASTA file.")

    num_embedded = process_sequences_and_save(
        sequences_to_process=all_sequences,
        model=model,
        tokenizer=tokenizer,
        family_key=family_key,
        embedding_type=args.embedding_type,
        device=device,
        h5_output_path=output_h5_path,
        max_seq_len=max_len_to_use,
        model_key_for_filename=args.model_key,
        random_init=args.random_init,
    )

    print("\n--- Embedding Generation Complete ---")
    print(f"Model: {args.model_key}")
    print(f"Total sequences processed/found in HDF5: {num_embedded}")
    print(f"Embeddings saved as datasets in HDF5 file: {output_h5_path}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean up .fai file
    fai_file_path = args.fasta_file.with_suffix(args.fasta_file.suffix + ".fai")
    if fai_file_path.is_file():
        try:
            fai_file_path.unlink()
            print(f"✓ Cleaned up index file: {fai_file_path}")
        except OSError as e:
            print(
                f"⚠️ Could not delete index file {fai_file_path}: {e}", file=sys.stderr
            )

    print("✓ Done.")


if __name__ == "__main__":
    main()
