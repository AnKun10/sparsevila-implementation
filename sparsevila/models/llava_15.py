"""Adapter for LLaVA-1.5-7B (haotian-liu/LLaVA).

Wraps the CLIPVisionTower so that its forward(images) returns an
`EncoderSalienceOutput` carrying pruned hidden_states + kept indices + salience.
The downstream `prepare_inputs_labels_for_multimodal` will need to be taught
to consume `pruned_hidden` in place of the regular encoder output (Task 15).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List
import torch
import torch.nn as nn

from ..config import SparseVILAConfig
from ..pruning.salience_estimator import compute_salience_from_attn
from ..pruning.encoder_pruner import prune_visual_tokens
from ..rope.unified_rope import build_compressed_position_ids
from ..cache.sparse_cache import SparseCache
from ..retrieval.decode_retriever import select_packed_kv_for_layer
from ..retrieval.packed_kv import PackedKV
from .base import VLMAdapter


@dataclass
class EncoderSalienceOutput:
    pruned_hidden: torch.Tensor    # (B, K, D)
    kept_idx: torch.Tensor         # (K,)
    salience: torch.Tensor         # (S,) original


@dataclass
class SparseLlavaContext:
    """Carries per-query state that the LLM-side wrapper needs."""
    visual_range: tuple[int, int]
    position_ids: torch.Tensor
    cache: SparseCache
    q_proj_per_layer: List[torch.Tensor]   # one (B, H, |Q|, D) per layer
    config: SparseVILAConfig


class _WrappedCLIPTower(nn.Module):
    def __init__(self, inner: nn.Module, config: SparseVILAConfig):
        super().__init__()
        self.inner = inner
        self.config = config
        # When True, return a plain tensor for LLaVA compatibility.
        # The kept_idx is stashed on `self.last_kept_idx` for the LLM-side
        # wrapper to read when building position_ids and visual_range.
        self.compat_mode: bool = True
        self.last_kept_idx: torch.Tensor | None = None
        self.last_salience: torch.Tensor | None = None

    def forward(self, images: torch.Tensor):
        # Expect inner.forward to return (hidden_states, attn_list)
        hs, attns = self.inner(images)
        # hs:    (B, S+1, D)  — CLS at index 0 for LLaVA-1.5 CLIP
        # attns: list of (B, H, S+1, S+1)
        attn = attns[self.config.salience_layer_idx]
        salience = compute_salience_from_attn(
            attn, strategy=self.config.salience_strategy,
        )
        patches = hs[:, 1:, :]      # strip CLS
        pruned, kept_idx = prune_visual_tokens(
            patches, salience, ratio=self.config.encoder_prune_ratio
        )
        self.last_kept_idx = kept_idx
        self.last_salience = salience
        if self.compat_mode:
            return pruned
        return EncoderSalienceOutput(
            pruned_hidden=pruned, kept_idx=kept_idx, salience=salience
        )


class LlavaFifteenAdapter(VLMAdapter):
    def wrap_encoder(self, vision_tower: nn.Module, config: SparseVILAConfig) -> nn.Module:
        return _WrappedCLIPTower(vision_tower, config)

    def wrap_llm(self, llm: Any, config: SparseVILAConfig) -> Any:
        """Install per-layer attention monkey-patches.

        Wires each `LlamaAttention.forward` to consult a `SparseLlavaContext`
        stashed on the model (`llm._sparsevila_ctx`). When the context is None
        (no current query), the original forward runs. When set, the wrapper
        routes attention through `packed_kvs[layer_idx]` instead of the full cache.
        """
        self._patch_llama_attention(llm, config)
        llm._sparsevila_ctx = None
        llm._sparsevila_packed_kvs = None
        return llm

    def _patch_llama_attention(self, llm: Any, config: SparseVILAConfig) -> None:
        for layer_idx, layer in enumerate(llm.model.layers):
            attn = layer.self_attn
            original_forward = attn.forward

            def make_forward(layer_idx=layer_idx, original=original_forward):
                def patched_forward(hidden_states, *args, **kwargs):
                    ctx = getattr(llm, "_sparsevila_ctx", None)
                    packed = getattr(llm, "_sparsevila_packed_kvs", None)
                    if ctx is None or packed is None:
                        return original(hidden_states, *args, **kwargs)
                    pk = packed[layer_idx]
                    from transformers.cache_utils import DynamicCache
                    tmp = DynamicCache()
                    tmp.key_cache.append(pk.k)
                    tmp.value_cache.append(pk.v)
                    kwargs = dict(kwargs)
                    kwargs["past_key_value"] = tmp
                    return original(hidden_states, *args, **kwargs)
                return patched_forward

            attn.forward = make_forward()

    def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
        """Find the single contiguous run of IMAGE_TOKEN_INDEX (-200) in input_ids."""
        IMAGE_TOKEN_INDEX = -200
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("v1 supports batch_size=1")
        mask = (input_ids[0] == IMAGE_TOKEN_INDEX)
        idxs = mask.nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            raise ValueError("No IMAGE_TOKEN_INDEX in input_ids")
        if idxs.numel() != 1:
            raise NotImplementedError("v1 supports exactly one image marker (one image)")
        v_start = int(idxs[0].item())
        v_end = v_start + 1   # caller expands once K is known
        return (v_start, v_end)

    def build_packed_kvs(self, ctx: SparseLlavaContext) -> List[PackedKV]:
        """Run Algorithm 3 for every layer; return one PackedKV per layer."""
        packed = []
        n_layers = len(ctx.cache.key_cache)
        if n_layers != len(ctx.q_proj_per_layer):
            raise ValueError(
                f"q_proj_per_layer length {len(ctx.q_proj_per_layer)} != "
                f"cache layers {n_layers}"
            )
        for l in range(n_layers):
            pk = select_packed_kv_for_layer(
                layer_idx=l,
                q_proj=ctx.q_proj_per_layer[l],
                cache=ctx.cache,
                visual_range=ctx.visual_range,
                position_ids=ctx.position_ids,
                decode_ratio=ctx.config.decode_retrieval_ratio,
                head_aggregation=ctx.config.decode_head_aggregation,
                use_flash=ctx.config.use_flash_kernel,
            )
            packed.append(pk)
        return packed

    def build_position_ids(self, sys_len: int, vis_len: int, text_len: int) -> torch.Tensor:
        return build_compressed_position_ids(sys_len, vis_len, text_len)


class SparseInferenceSession:
    """Context manager that wires `packed_kvs` onto the LLM for one generation."""

    def __init__(self, llm: Any, adapter: LlavaFifteenAdapter, ctx: SparseLlavaContext):
        self.llm = llm
        self.adapter = adapter
        self.ctx = ctx

    def __enter__(self):
        packed = self.adapter.build_packed_kvs(self.ctx)
        self.llm._sparsevila_ctx = self.ctx
        self.llm._sparsevila_packed_kvs = packed
        return self

    def __exit__(self, exc_type, exc, tb):
        self.llm._sparsevila_ctx = None
        self.llm._sparsevila_packed_kvs = None
        return False
