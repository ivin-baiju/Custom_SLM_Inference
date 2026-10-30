"""Prefix caching for shared prompts.

Implements a trie-based prefix tree that maps token-ID prefixes to
cached KV data, enabling multiple requests that share a common
system/prompt prefix to reuse the KV computations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class _TrieNode:
    """Internal trie node."""

    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    kv_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ref_count: int = 0
    last_access: float = 0.0
    depth: int = 0  # distance from root (== number of tokens along path)

    @property
    def has_kv(self) -> bool:
        return self.kv_data is not None

    def touch(self) -> None:
        self.last_access = time.monotonic()


class PrefixTree:
    """Trie-based prefix matcher on token-ID sequences.

    Each path from root to a marked node represents a cached prefix.
    KV data is stored at the *deepest* node of the prefix so that
    overlapping prefixes share structure.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size: int = 0  # number of stored prefixes

    def insert(
        self,
        token_ids: List[int],
        kv_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Insert *token_ids* → *kv_data* mapping.

        Parameters
        ----------
        token_ids : list[int]
            The token prefix to cache.
        kv_data : list[tuple[Tensor, Tensor]]
            Per-layer ``(key, value)`` cache for this prefix.
        """
        node = self._root
        for depth, tid in enumerate(token_ids, start=1):
            if tid not in node.children:
                node.children[tid] = _TrieNode(depth=depth)
            node = node.children[tid]
        node.kv_data = kv_data
        node.ref_count += 1
        node.touch()
        self._size += 1

    def find_longest_prefix(
        self, token_ids: List[int]
    ) -> Tuple[int, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Find the longest cached prefix of *token_ids*.

        Returns
        -------
        tuple[int, list | None]
            ``(prefix_length, kv_data)`` where *prefix_length* is the
            number of matched tokens and *kv_data* is the cached data
            (or ``None`` if no prefix was found).
        """
        node = self._root
        best_len = 0
        best_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        for depth, tid in enumerate(token_ids, start=1):
            if tid not in node.children:
                break
            node = node.children[tid]
            if node.has_kv:
                best_len = depth
                best_kv = node.kv_data
                node.touch()
        return best_len, best_kv

    def remove(self, token_ids: List[int]) -> bool:
        """Remove the prefix *token_ids* from the trie.

        Returns ``True`` if a prefix was actually removed.
        """
        # Walk down to find the node, keeping the path for cleanup.
        path: List[Tuple[int, _TrieNode]] = []
        node = self._root
        for tid in token_ids:
            if tid not in node.children:
                return False
            path.append((tid, node))
            node = node.children[tid]
        if not node.has_kv:
            return False

        node.ref_count -= 1
        if node.ref_count <= 0:
            node.kv_data = None
            node.ref_count = 0
            # Prune empty leaf nodes upward.
            for tid, parent in reversed(path):
                child = parent.children[tid]
                if not child.children and not child.has_kv:
                    del parent.children[tid]
                else:
                    break
        self._size -= 1
        return True

    @property
    def size(self) -> int:
        return self._size

    def all_prefixes(self) -> List[Tuple[List[int], _TrieNode]]:
        """Return all stored prefixes as ``(token_ids, node)`` pairs."""
        results: List[Tuple[List[int], _TrieNode]] = []
        self._walk(self._root, [], results)
        return results

    def _walk(
        self,
        node: _TrieNode,
        path: List[int],
        acc: List[Tuple[List[int], _TrieNode]],
    ) -> None:
        if node.has_kv:
            acc.append((list(path), node))
        for tid, child in node.children.items():
            path.append(tid)
            self._walk(child, path, acc)
            path.pop()


class PrefixCacheManager:
    """High-level prefix cache with LRU eviction and reference counting.

    Parameters
    ----------
    max_prefixes : int
        Maximum number of cached prefixes before eviction triggers.
    """

    def __init__(self, max_prefixes: int = 256) -> None:
        self._tree = PrefixTree()
        self._max_prefixes = max_prefixes

    def find_prefix(
        self, token_ids: List[int]
    ) -> Tuple[int, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Find the longest cached prefix matching *token_ids*.

        Returns
        -------
        tuple[int, list | None]
            ``(prefix_length, kv_data)``.
        """
        return self._tree.find_longest_prefix(token_ids)

    def store_prefix(
        self,
        token_ids: List[int],
        kv_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Cache a prefix's KV data, evicting LRU entries if over capacity.

        Parameters
        ----------
        token_ids : list[int]
            Token prefix.
        kv_data : list[tuple[Tensor, Tensor]]
            Per-layer ``(key, value)`` tensors.
        """
        while self._tree.size >= self._max_prefixes:
            self.evict_lru()
        self._tree.insert(token_ids, kv_data)
        logger.debug(
            "PrefixCacheManager: stored prefix of length %d "
            "(total cached: %d)",
            len(token_ids),
            self._tree.size,
        )

    def evict_lru(self) -> bool:
        """Evict the least-recently-used prefix.

        Returns ``True`` if a prefix was evicted.
        """
        prefixes = self._tree.all_prefixes()
        if not prefixes:
            return False
        # Find the prefix with the oldest last_access.
        lru_ids, lru_node = min(prefixes, key=lambda p: p[1].last_access)
        removed = self._tree.remove(lru_ids)
        if removed:
            logger.debug(
                "PrefixCacheManager: evicted prefix of length %d",
                len(lru_ids),
            )
        return removed

    def add_ref(self, token_ids: List[int]) -> None:
        """Increment reference count for *token_ids* prefix."""
        node = self._tree._root
        for tid in token_ids:
            if tid not in node.children:
                return
            node = node.children[tid]
        if node.has_kv:
            node.ref_count += 1

    def release_ref(self, token_ids: List[int]) -> None:
        """Decrement reference count for *token_ids* prefix."""
        node = self._tree._root
        for tid in token_ids:
            if tid not in node.children:
                return
            node = node.children[tid]
        if node.has_kv:
            node.ref_count = max(0, node.ref_count - 1)

    @property
    def num_cached(self) -> int:
        return self._tree.size

    def memory_usage(self) -> int:
        """Total bytes consumed by all cached KV data."""
        total = 0
        for _, node in self._tree.all_prefixes():
            if node.kv_data is not None:
                for k, v in node.kv_data:
                    total += k.nelement() * k.element_size()
                    total += v.nelement() * v.element_size()
        return total

    def __repr__(self) -> str:
        mb = self.memory_usage() / (1024 * 1024)
        return (
            f"PrefixCacheManager(cached={self.num_cached}, "
            f"max={self._max_prefixes}, mem={mb:.2f}MB)"
        )
