import math
import torch

from sparsevila.rope.unified_rope import (
    build_compressed_position_ids,
    rerotate_keys,
)


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


# ---- rerotate_keys ----

class _MiniRotary(torch.nn.Module):
    """Minimal stand-in for transformers.LlamaRotaryEmbedding for testing.

    Implements the same math but without the buffer-registration machinery,
    so the test does not depend on transformers being importable.
    """

    def __init__(self, dim: int = 8, base: int = 10000, max_pos: int = 32):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_pos, device=torch.device("cpu"),
                                dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def _apply_rope_at(k, positions, rotary):
    """Reference: rotate K freshly at given absolute positions."""
    cos = rotary.cos_cached[positions].to(k.dtype)  # (S, D)
    sin = rotary.sin_cached[positions].to(k.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = k.shape[-1] // 2
    rotated_half = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return k * cos + rotated_half * sin


def test_rerotate_keys_matches_fresh_rotation():
    """Re-rotating K from positions p_old to p_new should match applying
    RoPE freshly at p_new."""
    torch.manual_seed(0)
    rotary = _MiniRotary(dim=8, max_pos=32)
    B, H, S, D = 1, 2, 5, 8
    k_unrot = torch.randn(B, H, S, D)
    p_old = torch.tensor([0, 2, 4, 7, 10], dtype=torch.long)
    p_new = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # compressed
    delta = p_new - p_old

    k_at_old = _apply_rope_at(k_unrot, p_old, rotary)
    k_via_rerotate = rerotate_keys(k_at_old, delta, rotary)
    k_at_new_direct = _apply_rope_at(k_unrot, p_new, rotary)

    torch.testing.assert_close(k_via_rerotate, k_at_new_direct, atol=1e-5, rtol=1e-5)


def test_rerotate_keys_zero_delta_is_identity():
    rotary = _MiniRotary(dim=8, max_pos=32)
    k = torch.randn(1, 2, 4, 8)
    delta = torch.zeros(4, dtype=torch.long)
    out = rerotate_keys(k, delta, rotary)
    torch.testing.assert_close(out, k, atol=1e-6, rtol=1e-6)


def test_rerotate_keys_positive_negative_symmetry():
    """Rotating by +d then -d should be the identity."""
    torch.manual_seed(1)
    rotary = _MiniRotary(dim=8, max_pos=32)
    k = torch.randn(1, 2, 4, 8)
    delta_pos = torch.tensor([3, 5, 1, 7], dtype=torch.long)
    delta_neg = -delta_pos
    out = rerotate_keys(rerotate_keys(k, delta_pos, rotary), delta_neg, rotary)
    torch.testing.assert_close(out, k, atol=1e-5, rtol=1e-5)


def test_rerotate_keys_extends_cache():
    rotary = _MiniRotary(dim=8, max_pos=8)
    k = torch.randn(1, 1, 3, 8)
    # Force a delta larger than current cache.
    delta = torch.tensor([15, 16, 20], dtype=torch.long)
    _ = rerotate_keys(k, delta, rotary)
    assert rotary.max_seq_len_cached >= 21
