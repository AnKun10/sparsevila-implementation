"""Equivalence test: flash-kernel salience vs attn-map salience.

Builds a tiny synthetic 'model' with the minimum surface needed by
``per_layer_salience_via_flash_kernel`` and a paired set of attention
tensors that share the same logical math. The two paths should select
the same top-K visual indices on this fixture (per-layer).
"""
import math
import types

import pytest
import torch
from torch import nn

from sparsevila.inference.decode_salience import (
    per_layer_salience_from_attn_maps,
    per_layer_salience_via_flash_kernel,
)


class _MiniRotary(nn.Module):
    """Minimal LlamaRotaryEmbedding-shaped object."""

    def __init__(self, dim, max_pos=64):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0
        self._set_cos_sin_cache(max_pos, torch.device("cpu"), torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)   # (B, 1, S, D)
    sin = sin[position_ids].unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class _MiniAttn(nn.Module):
    """Simulates LlamaAttention's shape + q/k_proj."""

    def __init__(self, hidden_size, num_heads, head_dim, rotary):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.rotary_emb = rotary


class _MiniLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, rotary):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attn = _MiniAttn(hidden_size, num_heads, head_dim, rotary)


class _MiniModel(nn.Module):
    def __init__(self, n_layers, hidden_size, num_heads, head_dim, rotary):
        super().__init__()
        self.layers = nn.ModuleList([
            _MiniLayer(hidden_size, num_heads, head_dim, rotary)
            for _ in range(n_layers)
        ])


def _build_fixture(seed=0):
    """Make a tiny prefill state where we know exactly what Q, K were."""
    torch.manual_seed(seed)
    n_layers, hidden_size, num_heads, head_dim = 2, 16, 2, 8
    full_len = 12
    v_start, v_end = 2, 8   # 6 visual K positions
    q_start, q_end = 8, 12  # 4 text-Q positions

    rotary = _MiniRotary(dim=head_dim, max_pos=full_len)
    model = _MiniModel(n_layers, hidden_size, num_heads, head_dim, rotary)
    # Bind config so per_layer_salience_via_flash_kernel can read it.
    model.config = types.SimpleNamespace(num_hidden_layers=n_layers)
    # Top-level model has model.model.layers; mimic that.
    outer = nn.Module()
    outer.model = model
    outer.config = model.config

    # Random hidden states per layer (entry 0 = input to layer 0, etc.)
    hidden_states_per_layer = [
        torch.randn(1, full_len, hidden_size) for _ in range(n_layers)
    ]

    # Build the *expected* K cache and attn_maps consistently with those
    # hidden states, the LayerNorm, q/k_proj, and the standard Llama attention.
    cache = types.SimpleNamespace(key_cache=[], value_cache=[])
    attn_maps = []
    position_ids_full = torch.arange(full_len).unsqueeze(0)
    cos, sin = rotary(torch.empty(1, 1, 1, head_dim), seq_len=full_len)

    for i, layer in enumerate(model.layers):
        h = hidden_states_per_layer[i]
        h_norm = layer.input_layernorm(h)
        q_full = layer.self_attn.q_proj(h_norm).view(1, full_len, num_heads, head_dim).transpose(1, 2)
        k_full = layer.self_attn.k_proj(h_norm).view(1, full_len, num_heads, head_dim).transpose(1, 2)
        q_full = _apply_rope(q_full, cos, sin, position_ids_full)
        k_full = _apply_rope(k_full, cos, sin, position_ids_full)
        cache.key_cache.append(k_full)
        cache.value_cache.append(torch.zeros_like(k_full))

        # Full softmax attention (no causal — Llama is causal but for this
        # fixture we only look at Q-text rows attending to all K, which is
        # within the causal triangle if Q-text positions are at the end).
        scale = 1.0 / math.sqrt(head_dim)
        scores = q_full @ k_full.transpose(-2, -1) * scale
        # Apply standard causal mask matching Llama.
        causal = torch.triu(torch.ones(full_len, full_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn_maps.append(probs)

    return (
        outer, hidden_states_per_layer, cache, attn_maps,
        q_start, q_end, v_start, v_end,
    )


def test_flash_path_and_attn_map_path_pick_same_top_k():
    """Top-K visual indices should agree between the two paths.

    Caveat: the flash path runs a *non-causal* full softmax across all KV
    positions, while the attention maps come from Llama's causal attention.
    For the text-Q rows (the LAST 4 positions in our fixture) and visual-K
    columns (within the prefix), causality is trivially satisfied (all
    visual K are at earlier positions). So both paths softmax over the same
    set of in-bounds keys and should yield identical normalized
    distributions over visual positions.
    """
    outer, hs, cache, attn_maps, q_start, q_end, v_start, v_end = _build_fixture()

    sal_flash = per_layer_salience_via_flash_kernel(
        outer, hs, cache, q_start, q_end, v_start, v_end, use_flash=False,
    )
    sal_attn = per_layer_salience_from_attn_maps(
        attn_maps, q_start, q_end, v_start, v_end,
    )

    assert len(sal_flash) == len(sal_attn) == len(outer.model.layers)
    for layer_idx, (sf, sa) in enumerate(zip(sal_flash, sal_attn)):
        assert sf.shape == sa.shape == (v_end - v_start,), (
            f"Layer {layer_idx}: shape mismatch"
        )
        # Top-3 indices must match.
        k = 3
        top_f = sf.topk(k).indices.sort().values
        top_a = sa.topk(k).indices.sort().values
        assert torch.equal(top_f, top_a), (
            f"Layer {layer_idx}: top-{k} mismatch.\n"
            f"flash sal={sf.tolist()}\n"
            f"attn  sal={sa.tolist()}\n"
            f"flash top={top_f.tolist()} vs attn top={top_a.tolist()}"
        )


def test_flash_path_numerically_close_to_attn_map_path():
    """The two paths should produce numerically close salience tensors
    (not just same ranking)."""
    outer, hs, cache, attn_maps, q_start, q_end, v_start, v_end = _build_fixture(seed=42)

    sal_flash = per_layer_salience_via_flash_kernel(
        outer, hs, cache, q_start, q_end, v_start, v_end, use_flash=False,
    )
    sal_attn = per_layer_salience_from_attn_maps(
        attn_maps, q_start, q_end, v_start, v_end,
    )

    for layer_idx, (sf, sa) in enumerate(zip(sal_flash, sal_attn)):
        torch.testing.assert_close(sf, sa, atol=1e-5, rtol=1e-4)
