"""
CPU Execution Backend using PyTorch CPU operations.
"""

from __future__ import annotations

import psutil
from typing import Dict, Any
import torch

from backends.base import ExecutionBackend


class CPUBackend(ExecutionBackend):
    """CPU execution backend."""

    def __init__(self):
        super().__init__(torch.device("cpu"))

    @property
    def name(self) -> str:
        return "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def memory_info(self) -> Dict[str, Any]:
        """System RAM usage via psutil."""
        vm = psutil.virtual_memory()
        return {
            "device": "cpu",
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_bytes": vm.used,
            "free_bytes": vm.free,
            "percent": vm.percent,
        }
