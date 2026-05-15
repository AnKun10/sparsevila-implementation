# tests/test_llava15_inline_shim.py
import torch
from unittest.mock import MagicMock
from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import (
    LlavaFifteenAdapter, _WrappedCLIPTower, EncoderSalienceOutput,
)


class _FakeCLIPTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8

    def forward(self, images):
        b = images.shape[0]
        hs = torch.randn(b, 5, 8)   # 4 patches + CLS
        attn = torch.softmax(torch.randn(b, 1, 5, 5), dim=-1)
        return hs, [attn]


def test_wrapped_tower_compat_mode_returns_tensor_when_requested():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    # In compat mode, wrapped tower returns a plain tensor (LLaVA-compatible).
    wrapped.compat_mode = True
    out = wrapped(images)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 8)   # K=2 (after pruning)


def test_wrapped_tower_native_mode_returns_dataclass():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)


def test_wrapped_tower_records_last_kept_idx():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    wrapped.compat_mode = True
    wrapped(torch.randn(1, 3, 16, 16))
    assert wrapped.last_kept_idx is not None
    assert wrapped.last_kept_idx.numel() == 2
