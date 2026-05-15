"""Algorithm 3: query-aware decode-time KV retrieval, per layer."""
from __future__ import annotations
import torch
from ..cache.sparse_cache import SparseCache
from ..kernels.salience import column_salience
from .packed_kv import PackedKV


def select_packed_kv_for_layer(
    layer_idx: int,
    q_proj: torch.Tensor,                  # (B, H, |Q|, D)
    cache: SparseCache,
    visual_range: tuple[int, int],
    position_ids: torch.Tensor,            # (B, full_seq_len)
    decode_ratio: float,
    head_aggregation: str = "mean",
    use_flash: bool = True,
) -> PackedKV:
    v_start, v_end = visual_range
    V = v_end - v_start

    if decode_ratio == 0.0 or V == 0:
        keep_idx = torch.arange(V, device=q_proj.device)
        return cache.select_visual(
            layer_idx=layer_idx, visual_range=visual_range,
            keep_idx=keep_idx, position_ids=position_ids,
        )

    k_visual = cache.key_cache[layer_idx][:, :, v_start:v_end, :]   # (B, H, V, D)

    salience = column_salience(
        q_proj, k_visual, reduction="sum", is_causal=False, use_flash=use_flash,
    )  # (B, H, V)

    if head_aggregation == "mean":
        aggregated = salience.mean(dim=1)        # (B, V)
    elif head_aggregation == "max":
        aggregated = salience.max(dim=1).values  # (B, V)
    else:
        raise ValueError(f"head_aggregation must be mean|max, got {head_aggregation}")

    # Use the first batch row (paper assumes B=1)
    sal_b = aggregated[0]                        # (V,)
    threshold = sal_b.quantile(decode_ratio)
    keep_mask = sal_b > threshold
    if keep_mask.sum().item() == 0:
        k = max(1, round((1.0 - decode_ratio) * V))
        topk = sal_b.topk(k).indices
        keep_mask = torch.zeros_like(sal_b, dtype=torch.bool)
        keep_mask[topk] = True

    keep_idx = keep_mask.nonzero(as_tuple=True)[0].sort().values

    return cache.select_visual(
        layer_idx=layer_idx, visual_range=visual_range,
        keep_idx=keep_idx, position_ids=position_ids,
    )
