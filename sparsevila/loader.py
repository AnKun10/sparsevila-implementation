"""Top-level entry point: load_sparse_llava."""
from __future__ import annotations
from typing import Any
import torch

from .config import SparseVILAConfig
from .models.llava_15 import LlavaFifteenAdapter


def _load_llava_pretrained(
    model_path: str, dtype: torch.dtype, device: str,
    load_8bit: bool = False, load_4bit: bool = False,
):
    """Thin wrapper around llava.model.builder.load_pretrained_model.

    Isolated for mocking in tests. When ``load_4bit`` is True, LLaVA's
    builder routes through bitsandbytes' NF4 path
    (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=fp16, bnb_4bit_use_double_quant=True)). Roughly
    matches the W4A16 memory + compute profile of the paper's AWQ pipeline
    (different quant scheme — NF4 vs INT4 with activation-aware scales —
    but the same activation/weight bit widths and dequant-at-matmul flow).
    """
    from llava.model.builder import load_pretrained_model
    kwargs = {}
    # LLaVA's builder sets torch_dtype=fp16 internally when not quantized;
    # only forward it when quantization is off, so the bnb code-path keeps
    # bnb_4bit_compute_dtype=fp16 without conflict.
    if not (load_8bit or load_4bit):
        kwargs["torch_dtype"] = dtype
    return load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device,
        **kwargs,
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

    The LLM backbone can optionally be quantized via
    ``config.quantize_llm``: ``"4bit-bnb"`` (paper-equivalent W4A16) or
    ``"8bit-bnb"``. The vision tower stays FP16 in either case — paper
    uses W8A8 SmoothQuant on it but that requires a calibration pass and
    a custom INT8 vision implementation; left as future work.
    """
    if config is None:
        config = SparseVILAConfig()
    if not 0.0 <= config.encoder_prune_ratio < 1.0:
        raise ValueError(f"Invalid encoder_prune_ratio: {config.encoder_prune_ratio}")

    load_4bit = config.quantize_llm == "4bit-bnb"
    load_8bit = config.quantize_llm == "8bit-bnb"
    tokenizer, model, processor, ctx_len = _load_llava_pretrained(
        model_path, dtype, device,
        load_8bit=load_8bit, load_4bit=load_4bit,
    )

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
