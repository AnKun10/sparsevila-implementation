import torch
from sparsevila.cache.sparse_cache import SparseCache
from sparsevila.retrieval.decode_retriever import select_packed_kv_for_layer


def _make_cache_with_one_layer(seq_len=10, b=1, h=2, d=4):
    cache = SparseCache()
    cache.update(torch.randn(b, h, seq_len, d), torch.randn(b, h, seq_len, d), 0)
    return cache


def test_returns_packed_kv_with_expected_count():
    cache = _make_cache_with_one_layer(seq_len=12)
    q_proj = torch.randn(1, 2, 3, 4)  # |Q|=3
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0,
        q_proj=q_proj,
        cache=cache,
        visual_range=(2, 10),         # 8 visual tokens
        position_ids=pos_ids,
        decode_ratio=0.5,             # keep ~50% = 4
        head_aggregation="mean",
        use_flash=False,
    )
    # sys(2) + kept_visual(~4) + post(2) = ~8
    assert packed.k.shape[2] == 2 + 4 + 2
    assert packed.layer == 0


def test_ratio_zero_keeps_all_visual():
    cache = _make_cache_with_one_layer(seq_len=12)
    q_proj = torch.randn(1, 2, 2, 4)
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0, q_proj=q_proj, cache=cache,
        visual_range=(2, 10), position_ids=pos_ids,
        decode_ratio=0.0, head_aggregation="mean", use_flash=False,
    )
    assert packed.k.shape[2] == 12     # nothing pruned


def test_ties_fallback_to_topk():
    cache = _make_cache_with_one_layer(seq_len=12)
    # If we set q_proj to zero, all scores become uniform -> ties.
    q_proj = torch.zeros(1, 2, 2, 4)
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0, q_proj=q_proj, cache=cache,
        visual_range=(2, 10), position_ids=pos_ids,
        decode_ratio=0.5, head_aggregation="mean", use_flash=False,
    )
    # Fallback path: keep round((1-0.5)*8)=4 tokens
    assert packed.k.shape[2] == 2 + 4 + 2
