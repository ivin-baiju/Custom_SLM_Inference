"""
Phi-3 architecture implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.registry import register_architecture
from models.architectures.transformer import (
    BaseTransformerModel,
    TransformerBlock,
    Attention,
    FeedForward,
    RMSNorm,
)
from cache.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Phi3Attention(Attention):
    """Phi-3 attention module handling fused or unfused QKV projections."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        total_qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, total_qkv_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        query_states = qkv[:, :, :q_dim]
        key_states = qkv[:, :, q_dim : q_dim + kv_dim]
        value_states = qkv[:, :, q_dim + kv_dim :]

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, seq_len, offset=start_pos)
        from models.architectures.transformer import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, key_states, value_states, position=start_pos)
            key_states, value_states = kv_cache.get(self.layer_idx)

        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        total_kv_len = key_states.shape[2]
        if seq_len > 1:
            causal_mask = torch.full(
                (seq_len, total_kv_len),
                fill_value=float("-inf"),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            for i in range(seq_len):
                causal_mask[i, : start_pos + i + 1] = 0.0
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(1)
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Phi3Block(TransformerBlock):
    """Phi-3 transformer block."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Phi3Attention(config, layer_idx)


@register_architecture("Phi3ForCausalLM")
class Phi3Model(BaseTransformerModel):
    """Phi-3 causal language model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            Phi3Block(config, i) for i in range(config.num_hidden_layers)
        ])

    @classmethod
    def remap_weight_key(cls, key: str) -> str:
        if key.startswith("model."):
            return key[len("model.") :]
        return key
