"""Top-level entry point: load_sparse_llava."""
from __future__ import annotations
from typing import Any
import torch

from .config import SparseVILAConfig
from .models.llava_15 import LlavaFifteenAdapter


def _load_llava_pretrained(model_path: str, dtype: torch.dtype, device: str):
    """Thin wrapper around llava.model.builder.load_pretrained_model.

    Isolated for mocking in tests.
    """
    from llava.model.builder import load_pretrained_model
    return load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=False,
        load_4bit=False,
        device_map=device,
        torch_dtype=dtype,
    )


def load_sparse_llava(
    model_path: str,
    config: SparseVILAConfig | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load LLaVA-1.5 with SparseVILA wiring applied.

    Returns (model, processor). The model object is the LLaVA model with:
    - Vision tower wrapped to return EncoderSalienceOutput.
    - LLM attention layers patched to route through packed_kvs when set.
    """
    if config is None:
        config = SparseVILAConfig()
    if not 0.0 <= config.encoder_prune_ratio < 1.0:
        raise ValueError(f"Invalid encoder_prune_ratio: {config.encoder_prune_ratio}")

    tokenizer, model, processor, ctx_len = _load_llava_pretrained(model_path, dtype, device)

    adapter = LlavaFifteenAdapter()
    vt = model.get_vision_tower()
    wrapped_vt = adapter.wrap_encoder(vt, config)
    if hasattr(model, "set_vision_tower"):
        model.set_vision_tower(wrapped_vt)
    else:
        model.model.vision_tower = wrapped_vt

    adapter.wrap_llm(model, config)
    model._sparsevila_adapter = adapter
    model._sparsevila_config = config

    return model, processor
