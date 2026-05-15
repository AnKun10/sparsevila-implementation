# tests/test_llava15_inline_shim.py
from types import SimpleNamespace
import torch

from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import (
    LlavaFifteenAdapter, _WrappedCLIPTower, EncoderSalienceOutput,
)


class _FakeHFCLIPVisionModel(torch.nn.Module):
    def __init__(self, num_patches=4, hidden_size=8, num_heads=1):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images, output_hidden_states=False, output_attentions=False):
        b = images.shape[0]
        S1 = self.num_patches + 1
        hs_penult = torch.randn(b, S1, self.hidden_size)
        hs_last = torch.randn(b, S1, self.hidden_size)
        attn = torch.zeros(b, self.num_heads, S1, S1)
        attn[..., 0, 2] = 0.4   # CLS -> patch 1
        attn[..., 0, 3] = 0.6   # CLS -> patch 2
        return SimpleNamespace(
            hidden_states=(hs_penult, hs_last) if output_hidden_states else None,
            attentions=(attn,) if output_attentions else None,
        )


class _FakeCLIPTower(torch.nn.Module):
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


def test_wrapped_tower_compat_mode_returns_tensor_when_requested():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5, use_flash_kernel=False)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    # In compat mode, wrapped tower returns a plain tensor (LLaVA-compatible).
    wrapped.compat_mode = True
    out = wrapped(images)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 8)   # K=2 (after pruning)


def test_wrapped_tower_native_mode_returns_dataclass():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5, use_flash_kernel=False)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)


def test_wrapped_tower_records_last_kept_idx():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5, use_flash_kernel=False)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    wrapped.compat_mode = True
    wrapped(torch.randn(1, 3, 16, 16))
    assert wrapped.last_kept_idx is not None
    assert wrapped.last_kept_idx.numel() == 2
