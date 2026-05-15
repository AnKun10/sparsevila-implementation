"""Top-level SparseVILA-aware generation for LLaVA-1.5.

Two variants are exposed:

* :func:`sparse_generate` — attention-mask strategy. Drops nothing from
  the cache; instead softmax-masks dropped visual positions to ~0 via an
  additive bias hooked into each ``LlamaAttention``. Simpler; no RoPE
  recomputation needed; **no real speedup** (full attention still
  computed).

* :func:`sparse_generate_packed` — cache-packing strategy that matches
  the paper's Algorithm 3 + 4.1 (real decode speedup). Slices each
  layer's visual K/V down to the kept subset, re-applies RoPE to bring
  surviving K's onto compressed positions, then runs the decode loop
  against the packed cache. Uses :class:`SparseInferenceSession`'s
  attention-patch + per-layer proxy cache.

Decode retrieval — attention-mask strategy
------------------------------------------
For v1 we use an **attention-mask** strategy rather than physically packing
the KV cache:

1. Run one multimodal prefill (``model.forward(..., images=...,
   output_attentions=True, use_cache=True)``). Populates a SparseCache and
   exposes per-layer attention maps for salience computation.
2. Identify the visual span ``[v_start, v_end)`` in the embed sequence
   from the IMAGE_TOKEN_INDEX position plus the encoder wrapper's recorded
   ``last_kept_idx`` count.
3. For each LLM layer separately, compute query-aware salience from the
   layer's own attention map: how much the trailing text Q rows attend to
   each visual K column. Select the top
   ``(1 - decode_retrieval_ratio) * V`` visual indices to keep (per-layer
   indices, matching paper Algorithm 3).
4. Build one additive bias tensor per layer of shape
   ``(1, 1, 1, full_seq_len + max_new_tokens + 1)`` with ``-inf`` at the
   dropped visual positions and ``0`` elsewhere.
5. Register a forward-pre-hook on each ``LlamaAttention`` that ADDs that
   layer's bias to the layer's ``attention_mask`` kwarg, so the dropped
   visual K positions get softmax-zeroed during every decode step.

Trade-offs
----------
* Correctness ✓ — mathematically equivalent to dropping the K positions
  for that layer's attention, within FP16 fudge.
* RoPE ✓ — surviving K/V keep their original positional rotation; no
  recomputation.
* Speed ✗ — full attention is still computed; only dropped positions get
  softmax-masked. A real KV-packing variant is left as future work.
* Per-layer indices ✓ — each layer picks its own visual subset from its
  own attention map (paper Algorithm 3).
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import List, Optional

import torch

from ..cache.sparse_cache import SparseCache
from ..retrieval.packed_kv import PackedKV
from ..rope.unified_rope import rerotate_keys
from .decode_salience import (
    per_layer_salience_from_attn_maps,
    per_layer_salience_via_flash_kernel,
)


IMAGE_TOKEN_INDEX = -200    # LLaVA constant


def _compute_decode_salience(
    model, prefill_out, full_cache, q_start, q_end, v_start, v_end,
    use_flash_kernel: bool,
):
    """Dispatch decode salience to either the flash path or the attn-map path.

    Returns a list of per-layer salience tensors of shape ``(V,)``.
    """
    if use_flash_kernel:
        return per_layer_salience_via_flash_kernel(
            model=model,
            hidden_states_per_layer=prefill_out.hidden_states,
            cache=full_cache,
            q_start=q_start, q_end=q_end,
            v_start=v_start, v_end=v_end,
            use_flash=True,
        )
    return per_layer_salience_from_attn_maps(
        attentions=prefill_out.attentions,
        q_start=q_start, q_end=q_end,
        v_start=v_start, v_end=v_end,
    )


@contextmanager
def _install_sparse_attention_bias(model, sparse_biases: List[torch.Tensor]):
    """Register a forward-pre-hook on each LlamaAttention that adds the
    layer's ``sparse_biases[layer_idx]`` to the layer's ``attention_mask``
    kwarg.

    Each bias has shape ``(1, 1, 1, max_kv_len)`` with ``-inf`` at dropped
    visual positions and ``0`` everywhere else. It is broadcast over the
    q dimension of the 4D attention mask the layer receives.

    If a layer's ``attention_mask`` arrives as ``None`` (SDPA's fast
    is_causal path), we reconstruct a fresh 4D additive mask containing
    just the bias so the dropped positions still get masked.
    """
    handles = []

    def make_hook(layer_idx: int):
        bias = sparse_biases[layer_idx]

        def hook(module, args, kwargs):
            am = kwargs.get("attention_mask", None)
            if am is None:
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                q_len = hidden_states.shape[1]
                past_kv = kwargs.get("past_key_value", None)
                past_len = 0
                if past_kv is not None and getattr(past_kv, "key_cache", None):
                    li = getattr(module, "layer_idx", 0)
                    if li < len(past_kv.key_cache):
                        past_len = past_kv.key_cache[li].shape[-2]
                kv_len = past_len + q_len
                dtype = hidden_states.dtype
                new_am = bias[..., :kv_len].to(dtype).expand(1, 1, q_len, kv_len).clone()
            else:
                kv_len = am.shape[-1]
                new_am = am + bias[..., :kv_len].to(am.dtype)
            kwargs["attention_mask"] = new_am
            return args, kwargs

        return hook

    try:
        for i, layer in enumerate(model.model.layers):
            h = layer.self_attn.register_forward_pre_hook(
                make_hook(i), with_kwargs=True,
            )
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def sparse_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    decode_retrieval_ratio: float = 0.0,
    max_new_tokens: int = 80,
    eos_token_id: Optional[int] = None,
    use_flash_kernel: bool = True,
) -> torch.Tensor:
    """Greedy SparseVILA generation for LLaVA-1.5 single-image, batch=1.

    Returns only the *generated* tokens (excludes the prompt).

    When ``use_flash_kernel=True`` (default), decode salience is computed via
    :func:`~sparsevila.inference.decode_salience.per_layer_salience_via_flash_kernel`,
    avoiding the dense ``(L, L)`` attention map allocation in prefill. Set
    False to fall back to the simpler ``output_attentions=True`` path for
    debugging or numerical-equivalence comparison.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("v1 supports batch_size=1 only")
    if not 0.0 <= decode_retrieval_ratio < 1.0:
        raise ValueError(
            f"decode_retrieval_ratio must be in [0, 1), got {decode_retrieval_ratio}"
        )

    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    need_retrieval = decode_retrieval_ratio > 0.0
    n_layers = model.config.num_hidden_layers
    model_dtype = next(model.parameters()).dtype

    # ---- 1. PREFILL ----
    cache = SparseCache()
    prefill_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        past_key_values=cache,
        use_cache=True,
        output_attentions=(need_retrieval and not use_flash_kernel),
        output_hidden_states=(need_retrieval and use_flash_kernel),
        return_dict=True,
    )
    cache = prefill_out.past_key_values
    full_len = cache.get_seq_length()
    bias_len = full_len + max_new_tokens + 1

    # ---- 2. Per-layer sparse biases from salience ----
    if need_retrieval:
        image_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if image_pos.numel() != 1:
            raise NotImplementedError(
                "v1 supports exactly one IMAGE_TOKEN_INDEX marker (one image)"
            )
        K_visual = int(model.get_vision_tower().last_kept_idx.numel())
        v_start = int(image_pos.item())
        v_end = v_start + K_visual
        q_start, q_end = v_end, full_len
        keep_count = max(1, int(round((1.0 - decode_retrieval_ratio) * K_visual)))

        sal_per_layer = _compute_decode_salience(
            model, prefill_out, cache, q_start, q_end, v_start, v_end,
            use_flash_kernel=use_flash_kernel,
        )

        sparse_biases: List[torch.Tensor] = []
        for salience in sal_per_layer:
            kept_idx = salience.topk(keep_count).indices
            drop_mask = torch.ones(K_visual, dtype=torch.bool, device=device)
            drop_mask[kept_idx] = False
            absolute_drop = drop_mask.nonzero(as_tuple=True)[0] + v_start
            b = torch.zeros((1, 1, 1, bias_len), device=device, dtype=model_dtype)
            b[..., absolute_drop] = torch.finfo(model_dtype).min
            sparse_biases.append(b)
    else:
        zero_bias = torch.zeros((1, 1, 1, bias_len), device=device, dtype=model_dtype)
        sparse_biases = [zero_bias] * n_layers

    # Save next-token logits before discarding prefill_out (attentions/hiddens are large).
    first_logits = prefill_out.logits[0, -1, :].detach().clone()
    del prefill_out

    # ---- 3. First sampled token (from prefill logits) ----
    generated_ids: List[int] = []
    next_token = first_logits.argmax()
    generated_ids.append(int(next_token.item()))
    if int(next_token.item()) == eos_token_id:
        return torch.tensor([generated_ids], device=device)

    # ---- 4. Decode loop with per-layer sparse-bias hooks ----
    with _install_sparse_attention_bias(model, sparse_biases):
        for _ in range(max_new_tokens - 1):
            kv_len = cache.get_seq_length()
            cur_attn_mask = torch.ones(
                (1, kv_len + 1), device=device, dtype=attention_mask.dtype,
            )
            position_ids = torch.tensor([[kv_len]], device=device, dtype=torch.long)
            out = model(
                input_ids=next_token.view(1, 1),
                attention_mask=cur_attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_token = out.logits[0, -1, :].argmax()
            generated_ids.append(int(next_token.item()))
            if int(next_token.item()) == eos_token_id:
                break

    return torch.tensor([generated_ids], device=device)


# ---------------------------------------------------------------------------
# Cache-packing variant (paper Algorithm 3 + Section 4.1 latency claim)
# ---------------------------------------------------------------------------


@contextmanager
def _activate_packed_kvs(model, packed_kvs):
    """Install ``packed_kvs`` on the model so the per-layer attention patch
    (set up by :meth:`LlavaFifteenAdapter._patch_llama_attention`) routes
    each layer's attention through its packed view via
    :class:`~sparsevila.cache.proxy_cache.PerLayerProxyCache`.

    Sets a non-None ``_sparsevila_ctx`` sentinel so the patch fires. The
    actual ctx value is not read by the patch in this mode.
    """
    model._sparsevila_ctx = "packed"   # any non-None value activates the patch
    model._sparsevila_packed_kvs = packed_kvs
    try:
        yield
    finally:
        model._sparsevila_ctx = None
        model._sparsevila_packed_kvs = None


def _sync_packed_cache(packed_cache: SparseCache, packed_kvs: List[PackedKV]) -> None:
    """Pull the latest K/V from each PackedKV into the outer SparseCache.

    The per-layer attention patch writes the new token's K/V into
    ``packed_kvs[i].k/v`` after each forward, but the outer cache still
    holds the pre-step tensor references. Sync them so the next forward
    sees the correct ``past_key_values_length``.
    """
    for i, pk in enumerate(packed_kvs):
        packed_cache.key_cache[i] = pk.k
        packed_cache.value_cache[i] = pk.v
    packed_cache._seen_tokens = packed_kvs[0].k.shape[-2]


@torch.inference_mode()
def sparse_generate_packed(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    decode_retrieval_ratio: float = 0.0,
    max_new_tokens: int = 80,
    eos_token_id: Optional[int] = None,
    use_flash_kernel: bool = True,
) -> torch.Tensor:
    """Greedy SparseVILA generation with **cache-packing** retrieval.

    Mirrors :func:`sparse_generate` but actually shrinks the KV cache for
    every decode step, rather than softmax-masking dropped positions. The
    surviving visual K tensors are RoPE-recompressed to contiguous
    positions so attention against them stays numerically faithful.

    See module docstring for the trade-off discussion vs the attention-mask
    variant. ``use_flash_kernel`` controls the salience-computation path
    (see :func:`sparse_generate`).
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("v1 supports batch_size=1 only")
    if not 0.0 <= decode_retrieval_ratio < 1.0:
        raise ValueError(
            f"decode_retrieval_ratio must be in [0, 1), got {decode_retrieval_ratio}"
        )

    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    n_layers = model.config.num_hidden_layers
    need_retrieval = decode_retrieval_ratio > 0.0

    # ---- 1. PREFILL ----
    cache = SparseCache()
    prefill_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        past_key_values=cache,
        use_cache=True,
        output_attentions=(need_retrieval and not use_flash_kernel),
        output_hidden_states=(need_retrieval and use_flash_kernel),
        return_dict=True,
    )
    full_cache = prefill_out.past_key_values
    full_len = full_cache.get_seq_length()

    first_logits = prefill_out.logits[0, -1, :].detach().clone()

    # ---- Trivial path: no retrieval => just decode against the full cache ----
    if not need_retrieval:
        del prefill_out
        return _greedy_decode_full(
            model, full_cache, first_logits, eos_token_id,
            max_new_tokens, attention_mask, device,
        )

    # ---- 2. Locate visual span + per-layer kept indices ----
    image_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if image_pos.numel() != 1:
        raise NotImplementedError("v1 supports exactly one image marker")
    K_visual = int(model.get_vision_tower().last_kept_idx.numel())
    v_start = int(image_pos.item())
    v_end = v_start + K_visual
    q_start, q_end = v_end, full_len
    keep_count = max(1, int(round((1.0 - decode_retrieval_ratio) * K_visual)))
    K_drop = K_visual - keep_count
    packed_len = full_len - K_drop

    sal_per_layer = _compute_decode_salience(
        model, prefill_out, full_cache, q_start, q_end, v_start, v_end,
        use_flash_kernel=use_flash_kernel,
    )
    kept_per_layer = [
        sal.topk(keep_count).indices.sort().values for sal in sal_per_layer
    ]

    del prefill_out

    # ---- 3. Build per-layer PackedKV with RoPE re-rotation ----
    packed_kvs: List[PackedKV] = []
    for layer_idx in range(n_layers):
        k = full_cache.key_cache[layer_idx]   # (B, H, full_len, D)
        v = full_cache.value_cache[layer_idx]
        kept = kept_per_layer[layer_idx].to(k.device)

        absolute_keep = kept + v_start              # original positions
        compressed_keep = torch.arange(
            v_start, v_start + keep_count, dtype=torch.long, device=k.device,
        )
        delta_mid = compressed_keep - absolute_keep   # (K_keep,)
        rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb

        pre_k = k[:, :, :v_start, :]
        pre_v = v[:, :, :v_start, :]
        mid_k = k.index_select(2, absolute_keep)
        mid_v = v.index_select(2, absolute_keep)
        post_k = k[:, :, v_end:, :]
        post_v = v[:, :, v_end:, :]

        # Re-rotate K's so they live at compressed positions.
        mid_k = rerotate_keys(mid_k, delta_mid, rotary_emb)
        if post_k.shape[2] > 0 and K_drop != 0:
            delta_post = torch.full(
                (post_k.shape[2],), -K_drop, dtype=torch.long, device=k.device,
            )
            post_k = rerotate_keys(post_k, delta_post, rotary_emb)

        packed_k = torch.cat([pre_k, mid_k, post_k], dim=2)
        packed_v = torch.cat([pre_v, mid_v, post_v], dim=2)
        pos_ids = torch.arange(
            packed_len, dtype=torch.long, device=k.device,
        ).unsqueeze(0)
        packed_kvs.append(PackedKV(
            k=packed_k, v=packed_v, layer=layer_idx,
            visual_keep=kept, visual_range=(v_start, v_end),
            position_ids=pos_ids,
        ))

    # ---- 4. Build the outer "packed cache" the model sees as past_key_values ----
    packed_cache = SparseCache()
    for pk in packed_kvs:
        packed_cache.key_cache.append(pk.k)
        packed_cache.value_cache.append(pk.v)
    packed_cache._seen_tokens = packed_len

    # ---- 5. Decode loop with the per-layer attention patch ACTIVE ----
    next_token = first_logits.argmax()
    generated: List[int] = [int(next_token.item())]
    if int(next_token.item()) == eos_token_id:
        return torch.tensor([generated], device=device)

    with _activate_packed_kvs(model, packed_kvs):
        for _ in range(max_new_tokens - 1):
            cur_packed_len = packed_cache.get_seq_length()
            position_ids = torch.tensor(
                [[cur_packed_len]], device=device, dtype=torch.long,
            )
            cur_attn_mask = torch.ones(
                (1, cur_packed_len + 1), device=device, dtype=attention_mask.dtype,
            )
            out = model(
                input_ids=next_token.view(1, 1),
                attention_mask=cur_attn_mask,
                position_ids=position_ids,
                past_key_values=packed_cache,
                use_cache=True,
                return_dict=True,
            )
            _sync_packed_cache(packed_cache, packed_kvs)
            next_token = out.logits[0, -1, :].argmax()
            generated.append(int(next_token.item()))
            if int(next_token.item()) == eos_token_id:
                break

    return torch.tensor([generated], device=device)


@torch.inference_mode()
def _greedy_decode_full(
    model, cache: SparseCache, first_logits: torch.Tensor,
    eos_token_id: int, max_new_tokens: int,
    attention_mask: torch.Tensor, device,
) -> torch.Tensor:
    """Greedy decode helper used by sparse_generate_packed when retrieval is off."""
    next_token = first_logits.argmax()
    generated: List[int] = [int(next_token.item())]
    if int(next_token.item()) == eos_token_id:
        return torch.tensor([generated], device=device)
    for _ in range(max_new_tokens - 1):
        kv_len = cache.get_seq_length()
        cur_attn_mask = torch.ones(
            (1, kv_len + 1), device=device, dtype=attention_mask.dtype,
        )
        position_ids = torch.tensor([[kv_len]], device=device, dtype=torch.long)
        out = model(
            input_ids=next_token.view(1, 1),
            attention_mask=cur_attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_token = out.logits[0, -1, :].argmax()
        generated.append(int(next_token.item()))
        if int(next_token.item()) == eos_token_id:
            break
    return torch.tensor([generated], device=device)
