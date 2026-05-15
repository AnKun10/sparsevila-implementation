# tests/test_llava15_llm_decode_hook.py
import torch
from sparsevila.config import SparseVILAConfig
from sparsevila.cache.sparse_cache import SparseCache
from sparsevila.models.llava_15 import LlavaFifteenAdapter, SparseLlavaContext


def test_select_packed_kv_for_all_layers():
    """Smoke: build a fake 2-layer cache and verify the adapter produces
    one PackedKV per layer with the configured retrieval ratio."""
    cfg = SparseVILAConfig(decode_retrieval_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    cache = SparseCache()
    seq_len = 12
    for l in range(2):
        cache.update(torch.randn(1, 2, seq_len, 4), torch.randn(1, 2, seq_len, 4), l)

    # Build a SparseLlavaContext as the adapter expects.
    ctx = SparseLlavaContext(
        visual_range=(2, 10),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        cache=cache,
        q_proj_per_layer=[torch.randn(1, 2, 3, 4) for _ in range(2)],
        config=cfg,
    )

    packed_kvs = adapter.build_packed_kvs(ctx)
    assert len(packed_kvs) == 2
    for pk in packed_kvs:
        # sys(2) + keep_visual(~4) + post(2) = 8
        assert pk.k.shape[2] in (7, 8, 9)


def test_build_packed_kvs_ratio_zero_bypasses():
    cfg = SparseVILAConfig(decode_retrieval_ratio=0.0)
    adapter = LlavaFifteenAdapter()
    cache = SparseCache()
    seq_len = 8
    cache.update(torch.randn(1, 2, seq_len, 4), torch.randn(1, 2, seq_len, 4), 0)
    ctx = SparseLlavaContext(
        visual_range=(2, 6),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        cache=cache,
        q_proj_per_layer=[torch.randn(1, 2, 1, 4)],
        config=cfg,
    )
    packed_kvs = adapter.build_packed_kvs(ctx)
    assert packed_kvs[0].k.shape[2] == seq_len    # all kept
