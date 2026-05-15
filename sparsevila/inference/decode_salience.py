"""Decode-side salience computation for SparseVILA Algorithm 3.

Two paths are exposed:

* :func:`per_layer_salience_from_attn_maps` — reads the already-materialized
  per-layer attention tensors returned by
  ``LlamaForCausalLM(... output_attentions=True ...)``. Simple but allocates
  a full ``(B, H, L, L)`` matrix per layer at prefill time (~800MB for
  LLaVA-1.5-7B at 624-token prefill).

* :func:`per_layer_salience_via_flash_kernel` — reads ``output_hidden_states``
  (~165MB) instead, recomputes per-layer post-RoPE ``Q`` for the trailing
  text-question span, then calls
  :func:`sparsevila.kernels.salience.column_salience`
  (which dispatches to the Triton flash-colreduce kernel on CUDA, or a naive
  PyTorch fallback on CPU). Matches paper Algorithm 3:
  softmax over the *full* KV row, reduce-mean over Q rows, slice visual cols.

Both paths return a list of per-layer salience tensors of shape ``(V,)``
where V is the number of visual K positions, **mean over heads & Q tokens**
(paper formulation: ``sum(dim=(1,2)) / (H * Q_len)``).
"""
from __future__ import annotations
import math
from typing import List, Sequence

import torch

from ..kernels.salience import column_salience
from ..rope.unified_rope import _rotate_half


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
    """Apply standard Llama RoPE: ``x * cos[pos] + rotate_half(x) * sin[pos]``.

    Implemented inline (rather than importing from ``transformers``) so the
    code is robust to the apply_rotary_pos_emb signature changes between
    transformers 4.37 (our LLaVA pin) and newer versions.

    Args:
        x:            (B, H, S, D).
        cos, sin:     (max_seq_len, D) rotary tables.
        position_ids: (B, S) long.
    """
    cos_sel = cos[position_ids].unsqueeze(1)   # (B, 1, S, D)
    sin_sel = sin[position_ids].unsqueeze(1)
    return x * cos_sel + _rotate_half(x) * sin_sel


@torch.inference_mode()
def per_layer_salience_from_attn_maps(
    attentions: Sequence[torch.Tensor],
    q_start: int,
    q_end: int,
    v_start: int,
    v_end: int,
) -> List[torch.Tensor]:
    """Reduce per-layer attention maps directly.

    Args:
        attentions: tuple of per-layer attention tensors, each shape
            ``(B, H, L, L)``. Comes from
            ``LlamaForCausalLM(... output_attentions=True ...)``.
        q_start, q_end: text-question token range in the full sequence.
        v_start, v_end: visual K token range in the full sequence.
    """
    out: List[torch.Tensor] = []
    for layer_attn in attentions:
        contrib = layer_attn[:, :, q_start:q_end, v_start:v_end]
        salience = contrib.float().mean(dim=(0, 1, 2))   # mean over batch, head, Q
        out.append(salience)
    return out


@torch.inference_mode()
def per_layer_salience_via_flash_kernel(
    model,
    hidden_states_per_layer: Sequence[torch.Tensor],
    cache,
    q_start: int,
    q_end: int,
    v_start: int,
    v_end: int,
    use_flash: bool = True,
) -> List[torch.Tensor]:
    """Compute decode salience without materializing dense attention maps.

    Per layer:
      1. Re-derive the layer's post-RoPE ``Q`` for the text-Q range from the
         input hidden_states (cheap: one LayerNorm + one Linear over a small
         span, plus a single RoPE rotation).
      2. Pull the layer's full K cache (post-RoPE, kept-and-projected during
         prefill).
      3. Call :func:`column_salience` with ``reduction="mean"`` to softmax
         over **all** KV columns and reduce-mean over Q rows in a streaming
         manner (no ``(M, N)`` intermediate).
      4. Slice the visual-K columns and reduce over heads.

    The softmax denominator covers all KV positions, matching paper
    Algorithm 3 exactly (slicing visual columns from the resulting
    distribution rather than softmaxing over just the visual span).

    Args:
        model: ``LlavaLlamaForCausalLM``-shaped object whose
            ``model.layers[i].self_attn`` exposes ``q_proj``, ``num_heads``,
            ``head_dim``, and ``rotary_emb`` (Llama-style).
        hidden_states_per_layer: per-layer hidden states, length
            ``num_hidden_layers + 1`` (entry ``i`` is the input to layer
            ``i`` after the embedding + previous layers). Returned by
            ``model.forward(..., output_hidden_states=True).hidden_states``.
        cache: populated cache (``key_cache[i]`` is per-layer post-RoPE K).
        q_start, q_end, v_start, v_end: same as the attn-map variant.
        use_flash: forward to ``column_salience``. Set False to force the
            naive PyTorch path (mostly for tests / CPU).
    """
    n_layers = model.config.num_hidden_layers
    if len(hidden_states_per_layer) < n_layers:
        raise ValueError(
            f"hidden_states_per_layer has {len(hidden_states_per_layer)} entries; "
            f"expected at least {n_layers} (one per layer)."
        )

    out: List[torch.Tensor] = []
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # 1. Q for the text-Q span, post-RoPE.
        h = hidden_states_per_layer[layer_idx]            # (1, full_len, D_model)
        h_norm = layer.input_layernorm(h)
        q_full = attn.q_proj(h_norm)                      # (1, full_len, H*D_h)
        B, _, _ = q_full.shape
        H = attn.num_heads
        D = attn.head_dim
        q_full = q_full.view(B, -1, H, D).transpose(1, 2) # (1, H, full_len, D)
        q_text = q_full[:, :, q_start:q_end, :].contiguous()
        # RoPE at original absolute positions [q_start..q_end).
        cos, sin = attn.rotary_emb(q_text, seq_len=q_end)
        position_ids_text = torch.arange(
            q_start, q_end, device=h.device,
        ).unsqueeze(0)
        q_text_rope = _apply_rope(q_text, cos, sin, position_ids_text)

        # 2. Full K from cache (already post-RoPE).
        k_full = cache.key_cache[layer_idx]               # (1, H_kv, full_len, D)
        # Handle GQA by repeating K up to Q's head count if needed.
        num_kv_heads = k_full.shape[1]
        if num_kv_heads != H:
            assert H % num_kv_heads == 0, (
                f"num_heads ({H}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )
            repeat_factor = H // num_kv_heads
            k_full = k_full.repeat_interleave(repeat_factor, dim=1)

        # 3. Column salience over full KV; softmax denominator is full row.
        # is_causal=True matches Llama's prefill causal attention via the
        # right-aligned semantics of flash_colreduce: query row m attends to
        # keys 0..m + (N - M), which is exactly what Llama does when q is the
        # trailing Q-text span (q rows live at absolute positions
        # q_start..q_end-1 in a length-full_len sequence).
        scale = 1.0 / math.sqrt(D)
        sal_per_head = column_salience(
            q_text_rope, k_full,
            reduction="mean", is_causal=True, scale=scale,
            use_flash=use_flash,
        )                                                  # (1, H, full_len)

        # 4. Slice visual cols + mean over heads -> (V,)
        sal_visual = sal_per_head[:, :, v_start:v_end].float().mean(dim=1).squeeze(0)
        out.append(sal_visual)
    return out
