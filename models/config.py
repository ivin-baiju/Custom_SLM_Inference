"""
Model configuration parsing for HuggingFace-format ``config.json`` files.

Supports auto-detection of architecture type from the ``architectures`` field
and handles config variations across Llama, Qwen2, Phi-3, and Gemma2 families.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical architecture name mapping
# ---------------------------------------------------------------------------

_ARCHITECTURE_ALIASES: dict[str, str] = {
    # Llama family
    "LlamaForCausalLM": "LlamaForCausalLM",
    "MistralForCausalLM": "LlamaForCausalLM",
    "CodeLlamaForCausalLM": "LlamaForCausalLM",
    # Qwen family
    "Qwen2ForCausalLM": "Qwen2ForCausalLM",
    "QWenLMHeadModel": "Qwen2ForCausalLM",
    # Phi family
    "Phi3ForCausalLM": "Phi3ForCausalLM",
    "PhiForCausalLM": "Phi3ForCausalLM",
    "Phi3SmallForCausalLM": "Phi3ForCausalLM",
    # Gemma family
    "Gemma2ForCausalLM": "Gemma2ForCausalLM",
    "GemmaForCausalLM": "Gemma2ForCausalLM",
}


def _resolve_architecture(architectures: list[str]) -> str:
    """Map the first recognised architecture string to a canonical name.

    Parameters
    ----------
    architectures:
        The ``architectures`` list from a HuggingFace ``config.json``.

    Returns
    -------
    str
        Canonical architecture identifier understood by the registry.

    Raises
    ------
    ValueError
        If none of the listed architectures are recognised.
    """
    for arch in architectures:
        if arch in _ARCHITECTURE_ALIASES:
            resolved = _ARCHITECTURE_ALIASES[arch]
            logger.info("Resolved architecture %s → %s", arch, resolved)
            return resolved
    raise ValueError(
        f"Unrecognised architecture(s): {architectures}. "
        f"Supported: {sorted(_ARCHITECTURE_ALIASES.keys())}"
    )


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Complete transformer model configuration.

    Fields mirror the HuggingFace ``config.json`` schema with sane defaults
    so that architecture-specific omissions are handled gracefully.
    """

    # --- core dimensions ---------------------------------------------------
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # == num_attention_heads → MHA; < → GQA
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096

    # --- normalization & activation ----------------------------------------
    rms_norm_eps: float = 1e-6

    # --- RoPE --------------------------------------------------------------
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Any]] = None
    head_dim: Optional[int] = None  # derived if not set

    # --- special token ids -------------------------------------------------
    bos_token_id: int = 1
    eos_token_id: Union[int, list[int]] = 2
    pad_token_id: Optional[int] = None

    # --- weight tying & arch -----------------------------------------------
    tie_word_embeddings: bool = False
    architecture_type: str = "LlamaForCausalLM"

    # --- misc / per-arch extras -------------------------------------------
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # --- Qwen-specific ----------------------------------------------------
    qkv_bias: bool = False

    # --- Gemma-specific ---------------------------------------------------
    embedding_multiplier: Optional[float] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    # -- convenience helpers ------------------------------------------------

    @property
    def kv_heads_per_group(self) -> int:
        """Number of query heads that share each KV head (GQA ratio)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads < self.num_attention_heads

    def get(self, key: str, default: Any = None) -> Any:
        """Lookup a raw config value that may not be a first-class field."""
        return self._raw.get(key, default)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

# Map from HF config key → ModelConfig field name where they differ.
_KEY_MAP: dict[str, str] = {
    "num_key_value_heads": "num_key_value_heads",
    "num_kv_heads": "num_key_value_heads",
    "rms_norm_eps": "rms_norm_eps",
    "layer_norm_eps": "rms_norm_eps",
    "norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta",
    "partial_rotary_factor": "partial_rotary_factor",
}


def _extract_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull known fields out of a raw config dict, applying renames."""
    fields: dict[str, Any] = {}
    known_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}

    for key, value in raw.items():
        target = _KEY_MAP.get(key, key)
        if target in known_fields:
            fields[target] = value

    # --- Qwen-specific: attention_bias → qkv_bias -------------------------
    if "attention_bias" in raw:
        fields["qkv_bias"] = raw["attention_bias"]

    # --- Gemma-specific: hidden_size-based embedding scale -----------------
    if raw.get("model_type") in ("gemma", "gemma2"):
        fields["embedding_multiplier"] = raw.get(
            "embedding_multiplier", None
        )

    return fields


def load_config(path: Union[str, Path]) -> ModelConfig:
    """Load and parse a HuggingFace ``config.json`` into a :class:`ModelConfig`.

    Parameters
    ----------
    path:
        Path to either a ``config.json`` file directly **or** a directory
        containing one.

    Returns
    -------
    ModelConfig
        Fully-populated configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be located.
    ValueError
        If the architecture is not supported.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    logger.debug("Raw config keys: %s", sorted(raw.keys()))

    # Determine architecture
    architectures = raw.get("architectures", [])
    if not architectures:
        model_type = raw.get("model_type", "")
        arch_guess = {
            "llama": "LlamaForCausalLM",
            "mistral": "LlamaForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen": "Qwen2ForCausalLM",
            "phi3": "Phi3ForCausalLM",
            "phi": "Phi3ForCausalLM",
            "gemma": "Gemma2ForCausalLM",
            "gemma2": "Gemma2ForCausalLM",
        }.get(model_type, "")
        if arch_guess:
            architectures = [arch_guess]
            logger.warning(
                "No 'architectures' key; inferred %s from model_type=%s",
                arch_guess,
                model_type,
            )
        else:
            raise ValueError(
                f"Cannot determine architecture: no 'architectures' key and "
                f"unknown model_type '{model_type}'"
            )

    architecture_type = _resolve_architecture(architectures)

    extracted = _extract_fields(raw)
    extracted["architecture_type"] = architecture_type
    extracted["_raw"] = raw

    config = ModelConfig(**extracted)
    logger.info(
        "Loaded config: arch=%s, layers=%d, hidden=%d, heads=%d, kv_heads=%d",
        config.architecture_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    )
    return config
