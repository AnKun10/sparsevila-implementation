"""Tests for SparseCache snapshot/reset functionality."""
import torch
from sparsevila.cache.sparse_cache import SparseCache, CacheAnchor


def _populate(cache, n_layers=2, b=1, h=2, s=5, d=4):
    for l in range(n_layers):
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        cache.update(k, v, l)
    return cache


def test_snapshot_records_lengths():
    cache = SparseCache()
    _populate(cache, n_layers=3, s=7)
    anchor = cache.snapshot()
    assert isinstance(anchor, CacheAnchor)
    assert anchor.per_layer_len == [7, 7, 7]


def test_reset_to_truncates():
    cache = SparseCache()
    _populate(cache, n_layers=2, s=5)
    anchor = cache.snapshot()
    # Append more
    for l in range(2):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(k, v, l)
    assert cache.key_cache[0].shape[2] == 8
    cache.reset_to(anchor)
    assert cache.key_cache[0].shape[2] == 5
    assert cache.value_cache[0].shape[2] == 5


def test_reset_to_empty_anchor():
    cache = SparseCache()
    empty = cache.snapshot()              # len = []
    _populate(cache, n_layers=1, s=3)
    cache.reset_to(empty)
    # Empty anchor has 0 layers, so reset_to drops all layers
    assert len(cache.key_cache) == 0


import pytest


def test_select_visual_concat_shape():
    cache = SparseCache()
    # Layer 0: total seq=10; visual span is [3, 9), so 6 visual tokens.
    cache.update(torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4), 0)
    keep = torch.tensor([0, 2, 5])     # keep 3 of 6 visual tokens
    pos_ids = torch.arange(10).unsqueeze(0)
    packed = cache.select_visual(
        layer_idx=0, visual_range=(3, 9), keep_idx=keep, position_ids=pos_ids
    )
    # 3 (sys-ish) + 3 (kept visual) + 1 (post-visual) = 7
    assert packed.k.shape == (1, 2, 7, 4)
    assert packed.v.shape == (1, 2, 7, 4)
    assert packed.layer == 0


def test_select_visual_does_not_mutate_cache():
    cache = SparseCache()
    k0 = torch.randn(1, 2, 8, 4)
    cache.update(k0.clone(), torch.randn(1, 2, 8, 4), 0)
    before = cache.key_cache[0].clone()
    cache.select_visual(
        layer_idx=0, visual_range=(2, 6),
        keep_idx=torch.tensor([0, 1]),
        position_ids=torch.arange(8).unsqueeze(0),
    )
    assert torch.equal(cache.key_cache[0], before)


def test_select_visual_storage_independent():
    cache = SparseCache()
    cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
    packed = cache.select_visual(
        layer_idx=0, visual_range=(2, 6),
        keep_idx=torch.tensor([0, 1]),
        position_ids=torch.arange(8).unsqueeze(0),
    )
    # Mutating packed.k must not change the source.
    src_ptr = cache.key_cache[0].data_ptr()
    packed.k.add_(1.0)
    assert cache.key_cache[0].data_ptr() == src_ptr  # not reallocated
    # Storage is shared only if pointers overlap; check no overlap.
    assert packed.k.data_ptr() != cache.key_cache[0].data_ptr()


def test_select_visual_keep_idx_out_of_range_raises():
    cache = SparseCache()
    cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
    with pytest.raises(IndexError):
        cache.select_visual(
            layer_idx=0, visual_range=(2, 6),
            keep_idx=torch.tensor([10]),     # out of range
            position_ids=torch.arange(8).unsqueeze(0),
        )
