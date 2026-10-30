"""
Hardware execution backends for the custom SLM inference engine.

Provides uniform interface across CPU, Apple Silicon (MPS), and NVIDIA CUDA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional, Tuple, Dict, Any
import torch

logger = logging.getLogger(__name__)


class ExecutionBackend(ABC):
    """Abstract execution backend for tensor and neural network operations."""

    def __init__(self, device: torch.device):
        self.device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend (cpu, mps, cuda)."""
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend hardware is available on the host."""
        ...

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to backend device."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Allocate an uninitialized or empty tensor on device."""
        return torch.empty(shape, dtype=dtype, device=self.device)

    def zeros(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Allocate a zeros tensor on device."""
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication."""
        return torch.matmul(a, b)

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Linear projection: x @ weight.T + bias."""
        return torch.nn.functional.linear(x, weight, bias)

    def rms_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Root Mean Square layer normalization."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    def layer_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Standard LayerNorm."""
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), weight=weight, bias=bias, eps=eps
        )

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU (Swish) activation function."""
        return torch.nn.functional.silu(x)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        """GELU activation function."""
        return torch.nn.functional.gelu(x)

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax operation."""
        return torch.nn.functional.softmax(x, dim=dim)

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embedding to query or key tensor.
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim)
            cos: Tensor of shape (1, 1, seq_len, head_dim)
            sin: Tensor of shape (1, 1, seq_len, head_dim)
        """
        # Split head into two halves
        head_dim = x.shape[-1]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rotate_half = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotate_half * sin)

    def synchronize(self) -> None:
        """Wait for outstanding asynchronous operations on this device."""
        pass

    @abstractmethod
    def memory_info(self) -> Dict[str, Any]:
        """Return memory statistics dictionary."""
        ...


def detect_best_backend() -> str:
    """Auto-detect the most performant available hardware backend."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_backend(name: Optional[str] = None, device_index: int = 0) -> ExecutionBackend:
    """Factory to instantiate the appropriate execution backend."""
    from backends.cpu import CPUBackend
    from backends.mps import MPSBackend
    from backends.cuda import CUDABackend

    if not name or name == "auto":
        name = detect_best_backend()

    name = name.lower()
    if name == "cpu":
        return CPUBackend()
    elif name == "mps":
        if not MPSBackend.is_available():
            logger.warning("MPS requested but not available; falling back to CPU")
            return CPUBackend()
        return MPSBackend()
    elif name == "cuda":
        if not CUDABackend.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            return CPUBackend()
        return CUDABackend(device_index=device_index)
    else:
        raise ValueError(f"Unknown backend '{name}'. Supported: cpu, mps, cuda, auto")
