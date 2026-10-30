"""
NVIDIA CUDA execution backend.
"""

from __future__ import annotations

from typing import Dict, Any
import torch

from backends.base import ExecutionBackend


class CUDABackend(ExecutionBackend):
    """NVIDIA CUDA GPU execution backend."""

    def __init__(self, device_index: int = 0):
        super().__init__(torch.device(f"cuda:{device_index}"))
        self.device_index = device_index

    @property
    def name(self) -> str:
        return f"cuda:{self.device_index}"

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def memory_info(self) -> Dict[str, Any]:
        """CUDA memory statistics."""
        if not torch.cuda.is_available():
            return {"device": "cuda", "available": False}

        idx = self.device_index
        total = torch.cuda.get_device_properties(idx).total_memory
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        return {
            "device": f"cuda:{idx}",
            "device_name": torch.cuda.get_device_name(idx),
            "total_bytes": total,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "free_bytes": total - reserved,
        }
