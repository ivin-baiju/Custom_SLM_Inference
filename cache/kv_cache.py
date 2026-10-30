"""Core KV cache implementations.

Provides ``KVCache`` (base), ``DynamicKVCache`` (grows on demand),
and ``StaticKVCache`` (pre-allocated) for transformer inference.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class KVCache:
    """Base key-value cache for transformer attention layers.

    Stores key/value tensors per layer and tracks the current sequence
    length so that ``get`` returns only the valid portion of the cache.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers.
    num_kv_heads : int
        Number of key/value attention heads (may differ from query heads
        when using Grouped-Query Attention).
    head_dim : int
        Dimensionality of each attention head.
    max_seq_len : int
        Maximum sequence length the cache can hold.
    dtype : torch.dtype
        Data type for cached tensors.
    device : str | torch.device
        Device to allocate cache on.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)

        # Initialise empty – subclasses decide allocation strategy.
        self._keys: list[Optional[torch.Tensor]] = [None] * num_layers
        self._values: list[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_len: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> None:
        """Store new key/value entries at *position* for *layer_idx*.

        Parameters
        ----------
        layer_idx : int
            Transformer layer index.
        key : torch.Tensor
            Key tensor of shape ``(batch, num_kv_heads, seq, head_dim)``.
        value : torch.Tensor
            Value tensor, same shape as *key*.
        position : int
            Starting position in the sequence to write at.
        """
        self._validate_layer(layer_idx)
        seq_len = key.shape[2]
        end_pos = position + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Position {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        self._ensure_allocated(layer_idx, key.shape[0])
        assert self._keys[layer_idx] is not None
        assert self._values[layer_idx] is not None

        self._keys[layer_idx][:, :, position:end_pos, :] = key
        self._values[layer_idx][:, :, position:end_pos, :] = value
        self._seq_len = max(self._seq_len, end_pos)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached K, V for *layer_idx* up to current sequence length.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(key, value)`` each of shape
            ``(batch, num_kv_heads, seq_len, head_dim)``.

        Raises
        ------
        RuntimeError
            If the cache for this layer has not been populated yet.
        """
        self._validate_layer(layer_idx)
        k = self._keys[layer_idx]
        v = self._values[layer_idx]
        if k is None or v is None:
            raise RuntimeError(
                f"KV cache for layer {layer_idx} has not been populated"
            )
        return k[:, :, : self._seq_len, :], v[:, :, : self._seq_len, :]

    def get_seq_length(self) -> int:
        """Return the current cached sequence length."""
        return self._seq_len

    def reset(self) -> None:
        """Clear all cached data and reset sequence length."""
        for i in range(self.num_layers):
            if self._keys[i] is not None:
                self._keys[i].zero_()
            if self._values[i] is not None:
                self._values[i].zero_()
        self._seq_len = 0

    def memory_usage(self) -> int:
        """Return total bytes consumed by the KV cache."""
        total = 0
        for i in range(self.num_layers):
            for t in (self._keys[i], self._values[i]):
                if t is not None:
                    total += t.nelement() * t.element_size()
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

    def _ensure_allocated(self, layer_idx: int, batch_size: int) -> None:
        """Allocate cache tensors if not already present (lazy init)."""
        if self._keys[layer_idx] is None:
            shape = (batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
            self._keys[layer_idx] = torch.zeros(
                shape, dtype=self.dtype, device=self.device
            )
            self._values[layer_idx] = torch.zeros(
                shape, dtype=self.dtype, device=self.device
            )

    def __repr__(self) -> str:
        mb = self.memory_usage() / (1024 * 1024)
        return (
            f"{self.__class__.__name__}("
            f"layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"seq_len={self._seq_len}, mem={mb:.2f}MB)"
        )


class DynamicKVCache(KVCache):
    """KV cache that grows dynamically as tokens are generated.

    Starts with a small allocation and doubles capacity when the cache
    is full.  This avoids large upfront allocations for short sequences.

    Parameters
    ----------
    initial_size : int
        Initial capacity (number of tokens).  Defaults to 256.
    growth_factor : float
        Multiplicative factor when growing.  Defaults to 2.0.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cpu",
        initial_size: int = 256,
        growth_factor: float = 2.0,
    ) -> None:
        # We still respect max_seq_len as the absolute upper bound, but
        # internally start smaller.
        super().__init__(
            num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device
        )
        self._current_capacity = min(initial_size, max_seq_len)
        self._growth_factor = growth_factor

    def _ensure_allocated(self, layer_idx: int, batch_size: int) -> None:
        """Allocate at *_current_capacity* instead of *max_seq_len*."""
        if self._keys[layer_idx] is None:
            shape = (
                batch_size,
                self.num_kv_heads,
                self._current_capacity,
                self.head_dim,
            )
            self._keys[layer_idx] = torch.zeros(
                shape, dtype=self.dtype, device=self.device
            )
            self._values[layer_idx] = torch.zeros(
                shape, dtype=self.dtype, device=self.device
            )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> None:
        self._validate_layer(layer_idx)
        seq_len = key.shape[2]
        end_pos = position + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Position {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        self._ensure_allocated(layer_idx, key.shape[0])

        # Grow if necessary.
        if end_pos > self._current_capacity:
            self._grow(end_pos, key.shape[0])

        assert self._keys[layer_idx] is not None
        assert self._values[layer_idx] is not None
        self._keys[layer_idx][:, :, position:end_pos, :] = key
        self._values[layer_idx][:, :, position:end_pos, :] = value
        self._seq_len = max(self._seq_len, end_pos)

    def _grow(self, min_capacity: int, batch_size: int) -> None:
        """Grow internal buffers to hold at least *min_capacity* tokens."""
        new_cap = self._current_capacity
        while new_cap < min_capacity:
            new_cap = min(int(new_cap * self._growth_factor), self.max_seq_len)
        if new_cap == self._current_capacity:
            return

        logger.debug(
            "DynamicKVCache: growing %d → %d", self._current_capacity, new_cap
        )
        new_shape = (batch_size, self.num_kv_heads, new_cap, self.head_dim)
        for i in range(self.num_layers):
            if self._keys[i] is not None:
                new_k = torch.zeros(new_shape, dtype=self.dtype, device=self.device)
                new_v = torch.zeros(new_shape, dtype=self.dtype, device=self.device)
                old_len = self._keys[i].shape[2]
                new_k[:, :, :old_len, :] = self._keys[i]
                new_v[:, :, :old_len, :] = self._values[i]  # type: ignore[index]
                self._keys[i] = new_k
                self._values[i] = new_v
        self._current_capacity = new_cap

    def reset(self) -> None:
        """Reset cache and release grown buffers."""
        for i in range(self.num_layers):
            self._keys[i] = None
            self._values[i] = None
        self._seq_len = 0
        self._current_capacity = min(256, self.max_seq_len)


class StaticKVCache(KVCache):
    """Pre-allocated KV cache with a fixed size.

    All memory is allocated upfront in ``__init__``, making runtime
    allocation-free.  Best when the maximum sequence length is known
    ahead of time.

    Parameters
    ----------
    batch_size : int
        Batch size to pre-allocate for.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device
        )
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        for i in range(num_layers):
            self._keys[i] = torch.zeros(shape, dtype=dtype, device=device)
            self._values[i] = torch.zeros(shape, dtype=dtype, device=device)
        logger.info(
            "StaticKVCache: pre-allocated %.2f MB",
            self.memory_usage() / (1024 * 1024),
        )
