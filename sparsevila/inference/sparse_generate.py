"""Top-level SparseVILA-aware generation for LLaVA-1.5.

Wires encoder-side query-agnostic pruning (already plumbed by
``load_sparse_llava``) together with decode-side query-aware retrieval
into a single function.

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
3. Compute salience aggregated across LLM layers: how much the trailing
   text Q rows attend to each visual K column.
4. Select the top ``(1 - decode_retrieval_ratio) * V`` visual indices to
   keep; build an additive bias of length ``full_seq_len + max_new_tokens``
   that is ``-inf`` at the *dropped* visual positions and ``0`` elsewhere.
5. Register a forward-pre-hook on each ``LlamaAttention`` that ADDs this
   bias to the layer's ``attention_mask`` kwarg, so the dropped visual
   K positions get softmax-zeroed during every decode step.

Trade-offs
----------
* Correctness ✓ — mathematically equivalent to dropping the K positions,
  within FP16 fudge.
* RoPE ✓ — surviving K/V keep their original positional rotation; no
  recomputation.
* Speed ✗ — full attention is still computed; only dropped positions get
  softmax-masked. A real KV-packing variant is left as future work.
* Per-layer indices — v1 uses a single set of indices across all layers
  (mean salience over layers). Algorithm 3 uses per-layer indices; also
  future work.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import List, Optional

import torch

from ..cache.sparse_cache import SparseCache


IMAGE_TOKEN_INDEX = -200    # LLaVA constant


@contextmanager
def _install_sparse_attention_bias(model, sparse_bias: torch.Tensor):
    """Register a forward-pre-hook on each LlamaAttention that adds
    ``sparse_bias`` to the layer's ``attention_mask`` kwarg.

    ``sparse_bias`` has shape ``(1, 1, 1, max_kv_len)`` with ``-inf`` at
    dropped visual positions and ``0`` everywhere else. It is broadcast
    over the q dimension of the 4D attention mask the layer receives.
    """
    handles = []

    def make_hook():
        def hook(module, args, kwargs):
            am = kwargs.get("attention_mask", None)
            # Determine the kv_len for this call. When SDPA's fast path
            # drops the mask (am is None), reconstruct it from
            # hidden_states + past_key_value so we still get to inject the
            # sparse bias.
            if am is None:
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                q_len = hidden_states.shape[1]
                past_kv = kwargs.get("past_key_value", None)
                past_len = 0
                if past_kv is not None and getattr(past_kv, "key_cache", None):
                    layer_idx = getattr(module, "layer_idx", 0)
                    if layer_idx < len(past_kv.key_cache):
                        past_len = past_kv.key_cache[layer_idx].shape[-2]
                kv_len = past_len + q_len
                dtype = hidden_states.dtype
                new_am = sparse_bias[..., :kv_len].to(dtype).expand(1, 1, q_len, kv_len).clone()
            else:
                kv_len = am.shape[-1]
                new_am = am + sparse_bias[..., :kv_len].to(am.dtype)
            kwargs["attention_mask"] = new_am
            return args, kwargs
        return hook

    try:
        for layer in model.model.layers:
            h = layer.self_attn.register_forward_pre_hook(
                make_hook(), with_kwargs=True,
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
) -> torch.Tensor:
    """Greedy SparseVILA generation for LLaVA-1.5 single-image, batch=1.

    Returns only the *generated* tokens (excludes the prompt).
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

    # ---- 1. PREFILL ----
    cache = SparseCache()
    prefill_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        past_key_values=cache,
        use_cache=True,
        output_attentions=need_retrieval,
        return_dict=True,
    )
    cache = prefill_out.past_key_values
    full_len = cache.get_seq_length()

    # ---- 2. Build sparse bias from salience (only if retrieval is on) ----
    sparse_bias = torch.zeros(
        (1, 1, 1, full_len + max_new_tokens + 1),
        device=device, dtype=next(model.parameters()).dtype,
    )
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

        salience: Optional[torch.Tensor] = None
        for layer_attn in prefill_out.attentions:
            contrib = layer_attn[:, :, q_start:q_end, v_start:v_end]
            contrib = contrib.float().mean(dim=(0, 1, 2))   # (V,)
            salience = contrib if salience is None else (salience + contrib)
        salience = salience / len(prefill_out.attentions)

        keep_count = max(1, int(round((1.0 - decode_retrieval_ratio) * K_visual)))
        kept_idx = salience.topk(keep_count).indices
        drop_mask = torch.ones(K_visual, dtype=torch.bool, device=device)
        drop_mask[kept_idx] = False
        absolute_drop = drop_mask.nonzero(as_tuple=True)[0] + v_start
        sparse_bias[..., absolute_drop] = torch.finfo(sparse_bias.dtype).min

    # Save next-token logits before discarding prefill_out (attentions are huge).
    first_logits = prefill_out.logits[0, -1, :].detach().clone()
    del prefill_out

    # ---- 3. First sampled token (from prefill logits) ----
    generated_ids: List[int] = []
    next_token = first_logits.argmax()
    generated_ids.append(int(next_token.item()))
    if int(next_token.item()) == eos_token_id:
        return torch.tensor([generated_ids], device=device)

    # ---- 4. Decode loop with sparse bias hooks ----
    with _install_sparse_attention_bias(model, sparse_bias):
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
