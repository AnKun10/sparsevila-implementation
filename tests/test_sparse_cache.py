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
