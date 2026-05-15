"""Per-layer proxy cache used by the cache-packing decode path.

When the SparseVILA attention patch substitutes ``past_key_value`` on a
single :class:`~transformers.models.llama.modeling_llama.LlamaAttention`
call, we need a cache object that:

* exposes the *packed* (visual-subset) K/V for that layer as ``past``,
* accepts the layer's ``update(k_new, v_new, layer_idx=N)`` call regardless
  of ``N`` (the model layer index does not match the proxy's single
  internal slot),
* returns the concatenated K/V so attention computes against the packed
  past plus the new token.

After the layer's forward call, the proxy's K/V are read back and stored
on the PackedKV so the next decode step sees the appended token.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


class PerLayerProxyCache(DynamicCache):
    """A Cache that pretends to be the cache for a *single* model layer.

    All ``update`` / ``get_seq_length`` / ``get_usable_length`` calls,
    regardless of the requested ``layer_idx``, are routed to this proxy's
    one internal slot. This lets us swap in a per-layer packed view of a
    multi-layer cache so the standard LlamaAttention.forward — which uses
    its own ``self.layer_idx`` to index — works unchanged.
    """

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        super().__init__()
        self.key_cache = [k]
        self.value_cache = [v]
        self._seen_tokens = k.shape[-2]

    # --- DynamicCache overrides ---

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[0] = torch.cat([self.key_cache[0], key_states], dim=-2)
        self.value_cache[0] = torch.cat([self.value_cache[0], value_states], dim=-2)
        self._seen_tokens = self.key_cache[0].shape[-2]
        return self.key_cache[0], self.value_cache[0]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.key_cache[0].shape[-2]

    def get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:
        return self.key_cache[0].shape[-2]

    def get_max_length(self):
        return None

    # --- Convenience accessors ---

    @property
    def k(self) -> torch.Tensor:
        return self.key_cache[0]

    @property
    def v(self) -> torch.Tensor:
        return self.value_cache[0]
