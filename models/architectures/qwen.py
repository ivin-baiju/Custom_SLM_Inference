"""
Qwen2 architecture implementation.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from models.config import ModelConfig
from models.registry import register_architecture
from models.architectures.transformer import (
    BaseTransformerModel,
    TransformerBlock,
    Attention,
)

logger = logging.getLogger(__name__)


class Qwen2Attention(Attention):
    """Qwen2 attention with optional QKV bias."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Qwen2 supports QKV bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


class Qwen2Block(TransformerBlock):
    """Qwen2 transformer block using Qwen2Attention."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2Attention(config, layer_idx)


@register_architecture("Qwen2ForCausalLM")
class Qwen2Model(BaseTransformerModel):
    """Qwen2 causal language model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            Qwen2Block(config, i) for i in range(config.num_hidden_layers)
        ])

    @classmethod
    def remap_weight_key(cls, key: str) -> str:
        """Map HuggingFace state dict key to internal model key."""
        if key.startswith("model."):
            return key[len("model.") :]
        return key
