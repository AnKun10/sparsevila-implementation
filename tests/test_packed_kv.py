import torch
from sparsevila.retrieval.packed_kv import PackedKV


def _dummy_packed(b=1, h=2, s=5, d=4):
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    return PackedKV(
        k=k, v=v, layer=0,
        visual_keep=torch.arange(3),
        visual_range=(1, 4),
        position_ids=torch.arange(s).unsqueeze(0),
    )


def test_construct_shape():
    p = _dummy_packed()
    assert p.k.shape == (1, 2, 5, 4)
    assert p.v.shape == (1, 2, 5, 4)


def test_append_new_kv_grows_sequence():
    p = _dummy_packed(s=5)
    k_new = torch.randn(1, 2, 1, 4)
    v_new = torch.randn(1, 2, 1, 4)
    p2 = p.append_new_kv(k_new, v_new)
    assert p2.k.shape == (1, 2, 6, 4)
    assert p2.v.shape == (1, 2, 6, 4)
    assert torch.equal(p2.k[:, :, -1:, :], k_new)


def test_append_new_kv_does_not_mutate_original():
    p = _dummy_packed(s=5)
    k_before = p.k.clone()
    p.append_new_kv(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    assert torch.equal(p.k, k_before)
