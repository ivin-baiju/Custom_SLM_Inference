"""Memory management utilities for the SLM inference engine.

Provides device memory introspection, model/KV-cache memory estimation,
and OOM prevention with configurable reserve margins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Byte conversion helpers.
_KB = 1024
_MB = _KB * 1024
_GB = _MB * 1024

# Element sizes by dtype (bytes).
_DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int32: 4,
    torch.int64: 8,
}


@dataclass
class DeviceMemoryInfo:
    """Snapshot of a device's memory."""

    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.allocated_bytes

    @property
    def utilization(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes

    def __repr__(self) -> str:
        return (
            f"DeviceMemory(total={self.total_bytes / _GB:.2f}GB, "
            f"alloc={self.allocated_bytes / _GB:.2f}GB, "
            f"free={self.free_bytes / _GB:.2f}GB, "
            f"util={self.utilization:.1%})"
        )


class MemoryManager:
    """Centralized memory manager with OOM prevention.

    Parameters
    ----------
    reserve_margin : float
        Fraction of device memory to keep free (0.0–1.0).  Defaults to
        0.10 (10 %).
    """

    def __init__(self, reserve_margin: float = 0.10) -> None:
        if not 0.0 <= reserve_margin < 1.0:
            raise ValueError("reserve_margin must be in [0.0, 1.0)")
        self.reserve_margin = reserve_margin

    # ------------------------------------------------------------------
    # Device introspection
    # ------------------------------------------------------------------

    def get_device_memory(self, device: str | torch.device) -> DeviceMemoryInfo:
        """Return total and available memory for *device*.

        Supports ``"cpu"``, ``"mps"``, and ``"cuda"`` / ``"cuda:N"``.
        """
        device = torch.device(device)

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            idx = device.index or 0
            total = torch.cuda.get_device_properties(idx).total_mem
            alloc = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            return DeviceMemoryInfo(total, alloc, reserved)

        if device.type == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS is not available")
            alloc = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
            # MPS doesn't expose total device memory directly; use
            # driver_allocated as a proxy for reserved.
            reserved = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") else 0
            # Fall back to a conservative estimate if total is unavailable.
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                total = int(result.stdout.strip())
            except Exception:
                total = 16 * _GB  # conservative default
            return DeviceMemoryInfo(total, alloc, reserved)

        # CPU – report system RAM.
        try:
            import psutil

            vm = psutil.virtual_memory()
            return DeviceMemoryInfo(vm.total, vm.total - vm.available, 0)
        except ImportError:
            pass
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total = int(result.stdout.strip())
            return DeviceMemoryInfo(total, 0, 0)
        except Exception:
            return DeviceMemoryInfo(16 * _GB, 0, 0)

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_model_memory(config: Dict[str, Any], dtype: torch.dtype) -> int:
        """Predict the memory footprint of a model given its *config*.

        The *config* dict should contain at least:

        - ``hidden_size`` (int)
        - ``num_hidden_layers`` (int)
        - ``intermediate_size`` (int)
        - ``vocab_size`` (int)

        Returns
        -------
        int
            Estimated bytes.
        """
        elem = _DTYPE_BYTES.get(dtype, 2)
        hidden = config.get("hidden_size", 4096)
        layers = config.get("num_hidden_layers", 32)
        intermediate = config.get("intermediate_size", hidden * 4)
        vocab = config.get("vocab_size", 32000)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = hidden // num_heads

        # Embedding
        embedding_params = vocab * hidden

        # Per-layer: Q, K, V, O projections + gate/up/down MLP + norms
        qkv_params = hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
        o_params = num_heads * head_dim * hidden
        mlp_params = hidden * intermediate * 3  # gate + up + down
        norm_params = hidden * 2  # attention_norm + ffn_norm
        per_layer = qkv_params + o_params + mlp_params + norm_params

        # Final norm + lm_head
        final_params = hidden + vocab * hidden

        total_params = embedding_params + layers * per_layer + final_params
        return int(total_params * elem)

    @staticmethod
    def estimate_kv_memory(
        config: Dict[str, Any],
        seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Predict KV cache memory for a given configuration.

        Parameters
        ----------
        config : dict
            Model config with ``num_hidden_layers``, ``num_key_value_heads``,
            ``hidden_size``, ``num_attention_heads``.
        seq_len : int
            Sequence length (number of cached tokens).
        batch_size : int
            Batch size.
        dtype : torch.dtype
            Data type of cached tensors.

        Returns
        -------
        int
            Estimated bytes.
        """
        elem = _DTYPE_BYTES.get(dtype, 2)
        layers = config.get("num_hidden_layers", 32)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        hidden = config.get("hidden_size", 4096)
        head_dim = hidden // num_heads

        # 2 = key + value
        return int(2 * batch_size * layers * num_kv_heads * seq_len * head_dim * elem)

    # ------------------------------------------------------------------
    # Allocation guard
    # ------------------------------------------------------------------

    def can_allocate(
        self,
        requested_bytes: int,
        device: str | torch.device,
    ) -> bool:
        """Check whether *requested_bytes* can be safely allocated.

        Takes into account the configured ``reserve_margin``.
        """
        info = self.get_device_memory(device)
        usable = info.total_bytes * (1.0 - self.reserve_margin)
        return (info.allocated_bytes + requested_bytes) <= usable

    def assert_can_allocate(
        self,
        requested_bytes: int,
        device: str | torch.device,
        label: str = "allocation",
    ) -> None:
        """Raise ``RuntimeError`` if allocation would exceed safe limits."""
        if not self.can_allocate(requested_bytes, device):
            info = self.get_device_memory(device)
            raise RuntimeError(
                f"OOM prevention: cannot allocate "
                f"{requested_bytes / _MB:.1f} MB for '{label}' on {device}. "
                f"Device has {info.free_bytes / _MB:.1f} MB free, "
                f"reserve margin = {self.reserve_margin:.0%}."
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_memory_stats(
        self, device: Optional[str | torch.device] = None
    ) -> Dict[str, Any]:
        """Return a memory usage breakdown.

        If *device* is ``None``, reports stats for all available devices.
        """
        devices_to_check: list[str] = []
        if device is not None:
            devices_to_check.append(str(device))
        else:
            devices_to_check.append("cpu")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devices_to_check.append("mps")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices_to_check.append(f"cuda:{i}")

        stats: Dict[str, Any] = {}
        for d in devices_to_check:
            try:
                info = self.get_device_memory(d)
                stats[d] = {
                    "total_mb": info.total_bytes / _MB,
                    "allocated_mb": info.allocated_bytes / _MB,
                    "free_mb": info.free_bytes / _MB,
                    "utilization": info.utilization,
                    "reserve_margin": self.reserve_margin,
                    "usable_mb": info.total_bytes * (1 - self.reserve_margin) / _MB,
                }
            except RuntimeError as exc:
                stats[d] = {"error": str(exc)}
        return stats

    def __repr__(self) -> str:
        return f"MemoryManager(reserve_margin={self.reserve_margin:.0%})"
