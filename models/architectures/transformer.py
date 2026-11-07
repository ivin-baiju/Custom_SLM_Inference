"""
Core transformer building blocks for custom SLM inference engine.

Implements RMSNorm, RotaryEmbedding (RoPE), Multi-Head / Grouped-Query Attention (GQA),
FeedForward (SwiGLU/GeGLU), TransformerBlock, and BaseTransformerModel.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from cache.kv_cache import KVCache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.to(torch.float32)).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=torch.float32)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if total_len > self.cos_cached.shape[0]:
            self._build_cache(max(total_len, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[offset:total_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[offset:total_len].to(dtype=x.dtype, device=x.device)
        # Reshape to (1, 1, seq_len, dim) for broadcasting with (batch, num_heads, seq_len, head_dim)
        return cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-Head / Grouped-Query Attention (GQA) with KV Cache support."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)
        self.num_kv_heads = config.num_key_value_heads or self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        rope_scaling_val = 1.0
        if config.rope_scaling and "factor" in config.rope_scaling:
            rope_scaling_val = float(config.rope_scaling["factor"])

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling_val,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, seq_len, offset=start_pos)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV Cache
        if kv_cache is not None:
            # Update cache with new key and value states
            kv_cache.update(self.layer_idx, key_states, value_states, position=start_pos)
            # Retrieve full cached sequence
            key_states, value_states = kv_cache.get(self.layer_idx)

        # Key-Value heads repeating for GQA (zero-copy view expansion)
        total_kv_len = key_states.shape[2]
        if self.num_kv_groups > 1:
            key_states = (
                key_states.unsqueeze(2)
                .expand(batch_size, self.num_kv_heads, self.num_kv_groups, total_kv_len, self.head_dim)
                .reshape(batch_size, self.num_heads, total_kv_len, self.head_dim)
            )
            value_states = (
                value_states.unsqueeze(2)
                .expand(batch_size, self.num_kv_heads, self.num_kv_groups, total_kv_len, self.head_dim)
                .reshape(batch_size, self.num_heads, total_kv_len, self.head_dim)
            )

        # Scaled dot-product attention
        # query_states: (B, H, T_q, D)
        # key_states:   (B, H, T_k, D)
        # attn_weights: (B, H, T_q, T_k)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        if seq_len > 1:
            # Vectorized upper-triangular causal mask during prefill (O(1) kernel dispatch)
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, total_kv_len),
                    fill_value=float("-inf"),
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=start_pos + 1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(1)
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard Transformer Layer combining Attention, Norms, and Feed-Forward."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm Self Attention + Residual
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            start_pos=start_pos,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_out

        # Pre-norm MLP + Residual
        normed_post = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_post)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class BaseTransformerModel(nn.Module, ABC):
    """Abstract Base Transformer Model for Causal Language Modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through token embedding, transformer layers, norm, and lm_head.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            start_pos: Starting sequence position for RoPE and KV Cache
            kv_cache: Optional KVCache instance
            attention_mask: Optional attention mask
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embed_tokens(input_ids)

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
