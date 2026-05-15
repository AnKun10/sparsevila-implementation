"""SparseCache: DynamicCache + snapshot/reset (multi-turn ready) + select_visual."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    from ..retrieval.packed_kv import PackedKV


@dataclass
class CacheAnchor:
    per_layer_len: list[int] = field(default_factory=list)


class SparseCache(DynamicCache):
    """Adds:
    - `snapshot()` -> CacheAnchor of current per-layer seq lengths.
    - `reset_to(anchor)` truncates each layer back to the anchor length.
    - `select_visual(...)` (added in Task 9) returns a PackedKV view.
    """

    def snapshot(self) -> CacheAnchor:
        lengths = [k.shape[2] for k in self.key_cache]
        return CacheAnchor(per_layer_len=lengths)

    def reset_to(self, anchor: CacheAnchor) -> None:
        # When anchor was taken with N layers and we currently have M >= N,
        # truncate the first N layers and drop any layers beyond.
        n = len(anchor.per_layer_len)
        for l in range(n):
            target = anchor.per_layer_len[l]
            self.key_cache[l] = self.key_cache[l][:, :, :target, :].contiguous()
            self.value_cache[l] = self.value_cache[l][:, :, :target, :].contiguous()
        # Truncate the underlying lists if more layers exist
        del self.key_cache[n:]
        del self.value_cache[n:]
        # Update HF internal counter if present
        if hasattr(self, "_seen_tokens"):
            self._seen_tokens = anchor.per_layer_len[0] if anchor.per_layer_len else 0
