"""
Apple Silicon MPS (Metal Performance Shaders) execution backend.
"""

from __future__ import annotations

import logging
from typing import Dict, Any
import psutil
import torch

from backends.base import ExecutionBackend

logger = logging.getLogger(__name__)


class MPSBackend(ExecutionBackend):
    """Apple Silicon GPU execution backend via PyTorch MPS."""

    def __init__(self):
        super().__init__(torch.device("mps"))

    @property
    def name(self) -> str:
        return "mps"

    @classmethod
    def is_available(cls) -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def synchronize(self) -> None:
        """Synchronize MPS stream."""
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def memory_info(self) -> Dict[str, Any]:
        """MPS and Unified Memory statistics."""
        vm = psutil.virtual_memory()
        info: Dict[str, Any] = {
            "device": "mps",
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "unified_memory": True,
        }
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                info["allocated_bytes"] = torch.mps.current_allocated_memory()
            except Exception:
                info["allocated_bytes"] = None
        return info
