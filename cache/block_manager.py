"""Paged KV cache inspired by vLLM.

Implements a block-based memory manager that maps logical sequence
positions to fixed-size physical blocks, enabling efficient memory
sharing and fine-grained allocation / deallocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class KVBlock:
    """A fixed-size block storing K/V data for ``block_size`` tokens.

    Attributes
    ----------
    block_id : int
        Unique physical block identifier.
    block_size : int
        Number of tokens this block can hold.
    num_layers : int
        Number of transformer layers.
    num_kv_heads : int
        Number of KV attention heads.
    head_dim : int
        Dimensionality per head.
    ref_count : int
        Number of sequences sharing this block.
    num_filled : int
        Number of token slots currently occupied.
    """

    block_id: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    ref_count: int = 0
    num_filled: int = 0

    # Allocated lazily on first write.
    _keys: Optional[torch.Tensor] = field(default=None, repr=False)
    _values: Optional[torch.Tensor] = field(default=None, repr=False)

    def _ensure_allocated(self) -> None:
        if self._keys is None:
            shape = (
                self.num_layers,
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
            )
            self._keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._values = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def write(
        self,
        layer_idx: int,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write a single token's K/V into *slot* for *layer_idx*.

        Parameters
        ----------
        layer_idx : int
            Transformer layer.
        slot : int
            Slot index inside this block (0-based).
        key : torch.Tensor
            Shape ``(num_kv_heads, head_dim)``.
        value : torch.Tensor
            Shape ``(num_kv_heads, head_dim)``.
        """
        if slot >= self.block_size:
            raise IndexError(f"slot {slot} ≥ block_size {self.block_size}")
        self._ensure_allocated()
        assert self._keys is not None and self._values is not None
        self._keys[layer_idx, :, slot, :] = key
        self._values[layer_idx, :, slot, :] = value

    def read(
        self, layer_idx: int, num_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K/V for *layer_idx*.

        Parameters
        ----------
        layer_idx : int
            Transformer layer.
        num_tokens : int, optional
            Read only the first *num_tokens* slots.  Defaults to all
            filled slots.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(key, value)`` each ``(num_kv_heads, num_tokens, head_dim)``.
        """
        self._ensure_allocated()
        assert self._keys is not None and self._values is not None
        n = num_tokens if num_tokens is not None else self.num_filled
        return self._keys[layer_idx, :, :n, :], self._values[layer_idx, :, :n, :]

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.block_size

    def memory_bytes(self) -> int:
        """Return bytes consumed by this block (0 if not yet allocated)."""
        if self._keys is None:
            return 0
        return (self._keys.nelement() + self._values.nelement()) * self._keys.element_size()  # type: ignore[union-attr]

    def reset(self) -> None:
        """Reset the block so it can be reused."""
        self.num_filled = 0
        self.ref_count = 0
        if self._keys is not None:
            self._keys.zero_()
            self._values.zero_()  # type: ignore[union-attr]


class BlockTable:
    """Maps logical sequence positions to physical :class:`KVBlock` instances.

    A sequence's token at logical position *p* lives in block
    ``p // block_size`` at slot ``p % block_size``.
    """

    def __init__(self) -> None:
        self._table: List[KVBlock] = []

    def append(self, block: KVBlock) -> None:
        block.ref_count += 1
        self._table.append(block)

    def get_block(self, logical_block_idx: int) -> KVBlock:
        return self._table[logical_block_idx]

    @property
    def num_blocks(self) -> int:
        return len(self._table)

    def clear(self) -> None:
        for blk in self._table:
            blk.ref_count -= 1
        self._table.clear()

    def __iter__(self):
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


class PagedKVCache:
    """Paged KV cache with block-level memory management.

    Parameters
    ----------
    num_blocks : int
        Total number of physical blocks in the pool.
    block_size : int
        Tokens per block.
    num_layers : int
        Number of transformer layers.
    num_kv_heads : int
        Number of KV attention heads.
    head_dim : int
        Dimensionality per head.
    dtype : torch.dtype
        Data type for KV tensors.
    device : str | torch.device
        Device for tensors.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cpu",
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # Physical block pool.
        self._free_blocks: List[KVBlock] = [
            KVBlock(
                block_id=i,
                block_size=block_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=self.device,
            )
            for i in range(num_blocks)
        ]
        self._total_blocks = num_blocks

        # Per-sequence state.
        self._seq_tables: Dict[int, BlockTable] = {}
        self._seq_lengths: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Block allocation / release
    # ------------------------------------------------------------------

    def allocate_blocks(self, seq_id: int, num_blocks: int) -> List[KVBlock]:
        """Allocate *num_blocks* physical blocks for *seq_id*.

        Raises
        ------
        RuntimeError
            If not enough free blocks.
        """
        if num_blocks > len(self._free_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks; "
                f"only {len(self._free_blocks)} free"
            )
        if seq_id not in self._seq_tables:
            self._seq_tables[seq_id] = BlockTable()
            self._seq_lengths[seq_id] = 0

        allocated: List[KVBlock] = []
        for _ in range(num_blocks):
            block = self._free_blocks.pop()
            block.reset()
            self._seq_tables[seq_id].append(block)
            allocated.append(block)
        logger.debug(
            "PagedKVCache: allocated %d blocks for seq %d (free: %d)",
            num_blocks,
            seq_id,
            len(self._free_blocks),
        )
        return allocated

    def free_blocks(self, seq_id: int) -> int:
        """Release all blocks belonging to *seq_id*.

        Returns
        -------
        int
            Number of blocks freed.
        """
        table = self._seq_tables.pop(seq_id, None)
        if table is None:
            return 0
        freed = 0
        for block in table:
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.reset()
                self._free_blocks.append(block)
                freed += 1
        self._seq_lengths.pop(seq_id, None)
        logger.debug("PagedKVCache: freed %d blocks for seq %d", freed, seq_id)
        return freed

    # ------------------------------------------------------------------
    # Token-level operations
    # ------------------------------------------------------------------

    def append_token(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append a single token's K/V for *layer_idx* into *seq_id*'s cache.

        Automatically allocates a new block if the current one is full.

        Parameters
        ----------
        seq_id : int
            Sequence identifier.
        layer_idx : int
            Transformer layer index.
        key : torch.Tensor
            Shape ``(num_kv_heads, head_dim)``.
        value : torch.Tensor
            Shape ``(num_kv_heads, head_dim)``.
        """
        if seq_id not in self._seq_tables:
            self.allocate_blocks(seq_id, 1)

        table = self._seq_tables[seq_id]
        seq_len = self._seq_lengths[seq_id]
        block_idx = seq_len // self.block_size
        slot = seq_len % self.block_size

        # Need a new block?
        if block_idx >= table.num_blocks:
            self.allocate_blocks(seq_id, 1)

        block = table.get_block(block_idx)
        block.write(layer_idx, slot, key, value)

        # Track filled slots (only update on the first layer to avoid
        # double-counting across layers).
        if layer_idx == 0:
            block.num_filled = max(block.num_filled, slot + 1)
            self._seq_lengths[seq_id] = seq_len + 1

    def get_kv(
        self, seq_id: int, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the full K/V tensors for *seq_id* at *layer_idx*.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(key, value)`` each ``(num_kv_heads, total_tokens, head_dim)``.

        Raises
        ------
        KeyError
            If *seq_id* is unknown.
        """
        if seq_id not in self._seq_tables:
            raise KeyError(f"Unknown sequence {seq_id}")
        table = self._seq_tables[seq_id]
        seq_len = self._seq_lengths[seq_id]

        keys: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        remaining = seq_len
        for block in table:
            n = min(remaining, self.block_size)
            k, v = block.read(layer_idx, num_tokens=n)
            keys.append(k)
            values.append(v)
            remaining -= n
            if remaining <= 0:
                break

        return (
            torch.cat(keys, dim=1),   # (num_kv_heads, total, head_dim)
            torch.cat(values, dim=1),
        )

    # ------------------------------------------------------------------
    # Properties / stats
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Number of currently unallocated blocks."""
        return len(self._free_blocks)

    def memory_usage(self) -> Dict[str, int]:
        """Return memory usage breakdown.

        Returns
        -------
        dict[str, int]
            ``{"total": …, "per_sequence": {seq_id: bytes, …}}``.
        """
        per_seq: Dict[int, int] = {}
        for seq_id, table in self._seq_tables.items():
            per_seq[seq_id] = sum(blk.memory_bytes() for blk in table)
        return {
            "total": sum(per_seq.values()),
            "per_sequence": per_seq,  # type: ignore[dict-item]
        }

    def __repr__(self) -> str:
        used = self._total_blocks - len(self._free_blocks)
        return (
            f"PagedKVCache(total_blocks={self._total_blocks}, "
            f"used={used}, free={len(self._free_blocks)}, "
            f"sequences={len(self._seq_tables)})"
        )
