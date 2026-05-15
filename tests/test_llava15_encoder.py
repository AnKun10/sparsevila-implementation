"""Tests for LlavaFifteenAdapter.wrap_encoder + EncoderSalienceOutput.

The fake CLIPVisionTower mimics the minimum surface that the wrapper actually
reaches into:
  - .vision_tower(images, output_hidden_states=True, output_attentions=True)
    returning a BaseModelOutput-like with .hidden_states and .attentions tuples
  - .select_layer (int, e.g. -2)
  - .select_feature ("patch" or "cls_patch")
  - .dtype / .device properties

With ratio=0.5 and CLS-row attention [0, 0.4, 0.6, 0] over 4 patch keys, the
quantile threshold is 0.2 and the wrapper keeps patches 1 and 2.
"""
from types import SimpleNamespace
import torch

from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import LlavaFifteenAdapter, EncoderSalienceOutput


class _FakeHFCLIPVisionModel(torch.nn.Module):
    """Minimal stand-in for transformers.CLIPVisionModel."""

    def __init__(self, num_patches: int = 4, hidden_size: int = 8, num_heads: int = 1):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # Single dummy parameter so .to(...) / .dtype / .device work
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images, output_hidden_states=False, output_attentions=False):
        b = images.shape[0]
        S1 = self.num_patches + 1  # +1 for CLS

        # Two hidden state tensors so [-2] is the penultimate layer (LLaVA default).
        hs_penult = torch.randn(b, S1, self.hidden_size)
        hs_last = torch.randn(b, S1, self.hidden_size)

        # One attention map; CLS (row 0) attends two patches non-zero.
        attn = torch.zeros(b, self.num_heads, S1, S1)
        attn[..., 0, 2] = 0.4   # CLS -> patch 1
        attn[..., 0, 3] = 0.6   # CLS -> patch 2

        return SimpleNamespace(
            hidden_states=(hs_penult, hs_last) if output_hidden_states else None,
            attentions=(attn,) if output_attentions else None,
        )


class _FakeCLIPTower(torch.nn.Module):
    """Mimics LLaVA CLIPVisionTower surface used by the wrapper."""

    def __init__(self):
        super().__init__()
        self.vision_tower = _FakeHFCLIPVisionModel()
        self.select_layer = -2
        self.select_feature = "patch"

    @property
    def dtype(self):
        return self.vision_tower.weight.dtype

    @property
    def device(self):
        return self.vision_tower.weight.device


def test_wrap_encoder_returns_salience_output():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)
    assert out.salience.shape == (4,)
    assert out.pruned_hidden.shape == (1, 2, 8)
    assert out.kept_idx.numel() == 2
    assert 2 in out.kept_idx.tolist()


def test_wrap_encoder_zero_ratio_bypasses():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.0)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert out.pruned_hidden.shape == (1, 4, 8)
    assert out.kept_idx.numel() == 4
