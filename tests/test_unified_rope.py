import torch
from sparsevila.rope.unified_rope import build_compressed_position_ids


def test_lengths_sum():
    pos = build_compressed_position_ids(sys_len=10, kept_visual_count=288, text_len=20)
    assert pos.shape == (1, 318)


def test_contiguous_no_gaps():
    pos = build_compressed_position_ids(sys_len=5, kept_visual_count=100, text_len=7)
    diffs = pos[0, 1:] - pos[0, :-1]
    assert torch.all(diffs == 1)


def test_starts_at_zero():
    pos = build_compressed_position_ids(sys_len=3, kept_visual_count=8, text_len=2)
    assert pos[0, 0].item() == 0


def test_dtype_long():
    pos = build_compressed_position_ids(sys_len=1, kept_visual_count=1, text_len=1)
    assert pos.dtype == torch.long


def test_zero_visual():
    pos = build_compressed_position_ids(sys_len=5, kept_visual_count=0, text_len=3)
    assert pos.shape == (1, 8)
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
