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

    def select_visual(
        self,
        layer_idx: int,
        visual_range: tuple[int, int],
        keep_idx: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> "PackedKV":
        from ..retrieval.packed_kv import PackedKV   # local to break cycle

        v_start, v_end = visual_range
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        seq_len = k.shape[2]

        if (keep_idx.max().item() >= (v_end - v_start)) or (keep_idx.min().item() < 0):
            raise IndexError(
                f"keep_idx out of visual range [0, {v_end - v_start}): "
                f"min={keep_idx.min().item()}, max={keep_idx.max().item()}"
            )

        absolute_keep = (keep_idx + v_start).to(k.device)
        pre_k  = k[:, :, :v_start, :]
        mid_k  = k.index_select(2, absolute_keep)
        post_k = k[:, :, v_end:, :]
        pre_v  = v[:, :, :v_start, :]
        mid_v  = v.index_select(2, absolute_keep)
        post_v = v[:, :, v_end:, :]

        packed_k = torch.cat([pre_k, mid_k, post_k], dim=2)
        packed_v = torch.cat([pre_v, mid_v, post_v], dim=2)

        # Build position_ids of the packed sequence: keep sys + select-visual + post
        # Note: incoming position_ids must match the FULL cache seq_len.
        if position_ids.shape[-1] != seq_len:
            raise ValueError(
                f"position_ids length {position_ids.shape[-1]} != cache seq_len {seq_len}"
            )
        pos_pre  = position_ids[:, :v_start]
        pos_mid  = position_ids[:, v_start + keep_idx.cpu()]
        pos_post = position_ids[:, v_end:]
        packed_pos = torch.cat([pos_pre, pos_mid, pos_post], dim=1)

        return PackedKV(
            k=packed_k, v=packed_v, layer=layer_idx,
            visual_keep=keep_idx, visual_range=visual_range,
            position_ids=packed_pos,
        )
