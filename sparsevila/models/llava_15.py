"""Adapter for LLaVA-1.5-7B (haotian-liu/LLaVA).

Wraps the CLIPVisionTower so that its forward(images) returns an
`EncoderSalienceOutput` carrying pruned hidden_states + kept indices + salience.
The downstream `prepare_inputs_labels_for_multimodal` will need to be taught
to consume `pruned_hidden` in place of the regular encoder output (Task 15).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from ..config import SparseVILAConfig
from ..pruning.salience_estimator import compute_salience_from_attn
from ..pruning.encoder_pruner import prune_visual_tokens
from ..rope.unified_rope import build_compressed_position_ids
from .base import VLMAdapter


@dataclass
class EncoderSalienceOutput:
    pruned_hidden: torch.Tensor    # (B, K, D)
    kept_idx: torch.Tensor         # (K,)
    salience: torch.Tensor         # (S,) original


class _WrappedCLIPTower(nn.Module):
    def __init__(self, inner: nn.Module, config: SparseVILAConfig):
        super().__init__()
        self.inner = inner
        self.config = config

    def forward(self, images: torch.Tensor) -> EncoderSalienceOutput:
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
        return EncoderSalienceOutput(
            pruned_hidden=pruned, kept_idx=kept_idx, salience=salience
        )


class LlavaFifteenAdapter(VLMAdapter):
    def wrap_encoder(self, vision_tower: nn.Module, config: SparseVILAConfig) -> nn.Module:
        return _WrappedCLIPTower(vision_tower, config)

    def wrap_llm(self, llm: Any, config: SparseVILAConfig) -> Any:
        raise NotImplementedError("wrap_llm implemented in Task 15")

    def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
        raise NotImplementedError("get_visual_span implemented in Task 15")

    def build_position_ids(self, sys_len: int, vis_len: int, text_len: int) -> torch.Tensor:
        return build_compressed_position_ids(sys_len, vis_len, text_len)
