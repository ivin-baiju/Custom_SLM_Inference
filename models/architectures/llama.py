"""
Llama architecture implementation for Llama 2 / Llama 3 / Mistral family models.
"""

from __future__ import annotations

import logging
from typing import Dict, Any
import torch

from models.config import ModelConfig
from models.registry import register_architecture
from models.architectures.transformer import BaseTransformerModel

logger = logging.getLogger(__name__)


@register_architecture("LlamaForCausalLM")
class LlamaModel(BaseTransformerModel):
    """Llama / Mistral causal language model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    @classmethod
    def remap_weight_key(cls, key: str) -> str:
        """Map HuggingFace state dict key to internal model key.
        
        Examples:
            model.embed_tokens.weight -> embed_tokens.weight
            model.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
            model.norm.weight -> norm.weight
            lm_head.weight -> lm_head.weight
        """
        if key.startswith("model."):
            return key[len("model.") :]
        return key
