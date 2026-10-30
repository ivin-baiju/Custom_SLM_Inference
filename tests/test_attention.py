"""
Unit tests for transformer attention, RoPE, RMSNorm, and BaseTransformerModel.
"""

import pytest
import torch
from models.config import ModelConfig
from models.architectures.transformer import (
    RMSNorm,
    RotaryEmbedding,
    Attention,
    TransformerBlock,
    BaseTransformerModel,
)
from cache.kv_cache import DynamicKVCache


def test_rmsnorm():
    hidden_size = 64
    norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)
    x = torch.randn(2, 8, hidden_size)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance is approximately 1.0
    var = out.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_rotary_embedding():
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_position_embeddings=128)
    dummy = torch.randn(1, 1, 10, dim)
    cos, sin = rope(dummy, seq_len=10, offset=0)
    assert cos.shape == (1, 1, 10, dim)
    assert sin.shape == (1, 1, 10, dim)


def test_attention_mha_and_gqa():
    # Test GQA with 4 query heads and 2 kv heads
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    attn = Attention(config, layer_idx=0)
    x = torch.randn(2, 6, 64)
    out = attn(x)
    assert out.shape == (2, 6, 64)


def test_attention_with_kv_cache():
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    attn = Attention(config, layer_idx=0)
    cache = DynamicKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=128,
        dtype=torch.float32,
    )

    # 1. Prefill step with 4 tokens
    x_prefill = torch.randn(1, 4, 64)
    out_prefill = attn(x_prefill, start_pos=0, kv_cache=cache)
    assert out_prefill.shape == (1, 4, 64)
    assert cache.get_seq_length() == 4

    # 2. Decode step with 1 token at position 4
    x_decode = torch.randn(1, 1, 64)
    out_decode = attn(x_decode, start_pos=4, kv_cache=cache)
    assert out_decode.shape == (1, 1, 64)
    assert cache.get_seq_length() == 5
