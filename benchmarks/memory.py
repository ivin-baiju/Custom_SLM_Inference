"""
Memory benchmarking module for measuring weight and KV Cache allocations.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Any, Union
import torch
from models.config import ModelConfig
from cache.memory_manager import MemoryManager


def benchmark_memory(
    config: Union[ModelConfig, Dict[str, Any]],
    context_lengths: list[int] = [512, 1024, 2048, 4096],
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Calculate theoretical and measured memory footprint across sequence lengths."""
    mem_mgr = MemoryManager()
    cfg_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config

    model_mem = mem_mgr.estimate_model_memory(cfg_dict, dtype=dtype)

    kv_scaling = {}
    for ctx in context_lengths:
        kv_mem = mem_mgr.estimate_kv_memory(
            cfg_dict,
            seq_len=ctx,
            batch_size=batch_size,
            dtype=dtype,
        )
        kv_scaling[f"ctx_{ctx}"] = {
            "bytes": kv_mem,
            "mib": round(kv_mem / (1024 * 1024), 2),
        }

    return {
        "model_weights_mib": round(model_mem / (1024 * 1024), 2),
        "kv_cache_scaling": kv_scaling,
    }
