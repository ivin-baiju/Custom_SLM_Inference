"""
Hardware backends module.
"""

from backends.base import ExecutionBackend, detect_best_backend, get_backend
from backends.cpu import CPUBackend
from backends.mps import MPSBackend
from backends.cuda import CUDABackend

__all__ = [
    "ExecutionBackend",
    "CPUBackend",
    "MPSBackend",
    "CUDABackend",
    "detect_best_backend",
    "get_backend",
]
