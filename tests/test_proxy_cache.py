import torch
from sparsevila.cache.proxy_cache import PerLayerProxyCache


def test_initial_state():
    k = torch.randn(1, 32, 100, 128)
    v = torch.randn(1, 32, 100, 128)
    p = PerLayerProxyCache(k, v)
    assert p.get_seq_length() == 100
    assert p.get_seq_length(layer_idx=15) == 100   # ignores layer_idx
    assert p.k is k
    assert p.v is v


def test_update_appends_and_returns_full():
    k0 = torch.randn(1, 32, 100, 128)
    v0 = torch.randn(1, 32, 100, 128)
    p = PerLayerProxyCache(k0, v0)
    k_new = torch.randn(1, 32, 1, 128)
    v_new = torch.randn(1, 32, 1, 128)
    # LlamaAttention passes its model layer_idx (could be anything).
    k_out, v_out = p.update(k_new, v_new, layer_idx=17)
    assert k_out.shape == (1, 32, 101, 128)
    assert v_out.shape == (1, 32, 101, 128)
    assert p.get_seq_length() == 101
    assert torch.equal(k_out[:, :, :100, :], k0)
    assert torch.equal(k_out[:, :, 100:, :], k_new)


def test_update_works_for_any_layer_idx():
    """The proxy's behavior must NOT depend on layer_idx — the same
    proxy will be passed to many different LlamaAttention layers across
    different decode steps."""
    k = torch.randn(1, 1, 5, 8)
    v = torch.randn(1, 1, 5, 8)
    for layer_idx in (0, 1, 15, 31):
        p = PerLayerProxyCache(k, v)
        k_out, _ = p.update(torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8),
                            layer_idx=layer_idx)
        assert k_out.shape == (1, 1, 6, 8)


def test_get_usable_length_returns_seq_len():
    k = torch.randn(1, 1, 10, 4)
    v = torch.randn(1, 1, 10, 4)
    p = PerLayerProxyCache(k, v)
    assert p.get_usable_length(1, layer_idx=5) == 10
