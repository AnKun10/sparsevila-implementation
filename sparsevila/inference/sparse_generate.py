"""Top-level SparseVILA-aware generation for LLaVA-1.5.

Wires encoder-side query-agnostic pruning (already plumbed by
``load_sparse_llava``) together with decode-side query-aware retrieval
(Algorithm 3 of the paper) into a single function.

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


IMAGE_TOKEN_INDEX = -200    # LLaVA constant


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
        output_attentions=need_retrieval,
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

        sparse_biases: List[torch.Tensor] = []
        for layer_attn in prefill_out.attentions:
            # layer_attn: (1, H, L, L); take Q -> visual_K slice
            contrib = layer_attn[:, :, q_start:q_end, v_start:v_end]
            salience = contrib.float().mean(dim=(0, 1, 2))   # (V,)
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

    # Save next-token logits before discarding prefill_out (attentions are huge).
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
