"""
Gemma2 architecture implementation.
"""

from __future__ import annotations

import math
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.registry import register_architecture
from models.architectures.transformer import (
    BaseTransformerModel,
    TransformerBlock,
    Attention,
    RMSNorm,
)
from cache.kv_cache import KVCache

logger = logging.getLogger(__name__)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm (weights added to 1.0)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        # Gemma adds 1.0 to weight
        return (normed * (1.0 + self.weight.to(torch.float32))).to(input_dtype)


class GemmaMLP(nn.Module):
    """Gemma GeGLU feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma2Block(nn.Module):
    """Gemma 2 block with pre- and post-attention/feed-forward normalization."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            start_pos=start_pos,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = hidden_states + attn_out

        ffn_in = self.pre_feedforward_layernorm(hidden_states)
        ffn_out = self.mlp(ffn_in)
        ffn_out = self.post_feedforward_layernorm(ffn_out)
        hidden_states = hidden_states + ffn_out
        return hidden_states


@register_architecture("Gemma2ForCausalLM")
class Gemma2Model(BaseTransformerModel):
    """Gemma2 causal language model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.normalizer = math.sqrt(config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma2Block(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Gemma scales the token embeddings
        hidden_states = self.embed_tokens(input_ids) * self.normalizer

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                start_pos=start_pos,
                kv_cache=kv_cache,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def remap_weight_key(cls, key: str) -> str:
        if key.startswith("model."):
            return key[len("model.") :]
        return key
