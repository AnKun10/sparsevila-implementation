"""Tests for LlavaFifteenAdapter.wrap_encoder + EncoderSalienceOutput.

The fake attention is designed so two patch tokens have non-zero salience,
ensuring prune_visual_tokens (quantile strategy) keeps exactly 2 of 4 with
ratio=0.5.

Adapted from the task spec: original spec used only one hot patch
(attn[..., 0, 3] = 1.0), which with strategy="cls" yields salience
[0, 0, 1, 0]. quantile(0.5) = 0, keep_mask = salience > 0 → only 1 token
kept, contradicting the numel()==2 assertion. Fix (Option 1): set two patches
with non-zero attention so salience has two non-zero entries; quantile(0.5)
then lies strictly between 0 and the peak, keeping exactly 2 tokens.
"""
import torch
import pytest
from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import LlavaFifteenAdapter, EncoderSalienceOutput


class _FakeCLIPTower(torch.nn.Module):
    """Mimics LLaVA CLIPVisionTower minimally."""
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.num_patches = 4   # 2x2 patches for the test

    def forward(self, images):
        # Return hidden_states (B, S+1, D) with CLS at 0
        b = images.shape[0]
        hs = torch.randn(b, self.num_patches + 1, self.hidden_size)
        # Fake attention of shape (B, H=1, S+1, S+1)
        # Two patches get non-zero CLS attention so salience has two
        # non-zero entries.  With ratio=0.5 and salience=[0, 0.4, 0.6, 0],
        # quantile(0.5)=0.2, keep_mask = salience>0.2 → patches 1 and 2 kept.
        attn = torch.zeros(b, 1, self.num_patches + 1, self.num_patches + 1)
        attn[..., 0, 2] = 0.4   # CLS -> patch 1 (attn col index 2 = patch idx 1)
        attn[..., 0, 3] = 0.6   # CLS -> patch 2 (attn col index 3 = patch idx 2)
        return hs, [attn]        # (hidden_states, [attn per layer])


def test_wrap_encoder_returns_salience_output():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)
    # All 4 patches present in salience score vector
    assert out.salience.shape == (4,)
    # Pruning: keep 2 of 4
    assert out.pruned_hidden.shape == (1, 2, 8)
    assert out.kept_idx.numel() == 2
    # The patch with the highest signal (idx 2) should be kept
    assert 2 in out.kept_idx.tolist()


def test_wrap_encoder_zero_ratio_bypasses():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.0)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    # No tokens pruned
    assert out.pruned_hidden.shape == (1, 4, 8)
    assert out.kept_idx.numel() == 4
