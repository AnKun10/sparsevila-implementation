"""Abstract adapter — per-model SparseVILA injection points."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch


class VLMAdapter(ABC):
    """Each supported VLM provides one of these."""

    @abstractmethod
    def wrap_encoder(self, vision_tower: Any, config: Any) -> Any:
        """Return the (possibly wrapped) vision tower that exposes salience."""

    @abstractmethod
    def wrap_llm(self, llm: Any, config: Any) -> Any:
        """Return the (possibly wrapped) LLM with decode-time retrieval injected."""

    @abstractmethod
    def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
        """Return (v_start, v_end) in the input_ids sequence."""

    @abstractmethod
    def build_position_ids(
        self, sys_len: int, vis_len: int, text_len: int
    ) -> torch.Tensor:
        """Return adjusted position_ids (1, S)."""
