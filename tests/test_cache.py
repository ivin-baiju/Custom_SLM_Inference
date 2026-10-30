"""
Unit tests for KV cache strategies (Dynamic, Static, Paged, Prefix, MemoryManager).
"""

import pytest
import torch
from cache.kv_cache import DynamicKVCache, StaticKVCache
from cache.block_manager import PagedKVCache
from cache.prefix_cache import PrefixCacheManager
from cache.memory_manager import MemoryManager
from models.config import ModelConfig


def test_dynamic_kv_cache():
    cache = DynamicKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=64,
        dtype=torch.float32,
    )
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    cache.update(layer_idx=0, key=k, value=v, position=0)
    assert cache.get_seq_length() == 4

    cached_k, cached_v = cache.get(layer_idx=0)
    assert cached_k.shape == (1, 2, 4, 16)
    assert torch.equal(cached_k, k)


def test_static_kv_cache():
    cache = StaticKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=64,
        dtype=torch.float32,
    )
    k = torch.randn(1, 2, 3, 16)
    v = torch.randn(1, 2, 3, 16)
    cache.update(layer_idx=1, key=k, value=v, position=0)
    assert cache.get_seq_length() == 3


def test_paged_kv_cache():
    paged = PagedKVCache(
        num_blocks=10,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
    )
    seq_id = "test_seq_1"
    # Allocate 2 blocks (capacity = 32 tokens)
    blocks = paged.allocate_blocks(seq_id, num_blocks=2)
    assert len(blocks) == 2
    assert paged.num_free_blocks == 8

    # Free blocks
    paged.free_blocks(seq_id)
    assert paged.num_free_blocks == 10


def test_memory_manager():
    mgr = MemoryManager()
    config = {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    }
    kv_mem = mgr.estimate_kv_memory(config, seq_len=1024, batch_size=1, dtype=torch.float16)
    assert kv_mem > 0
