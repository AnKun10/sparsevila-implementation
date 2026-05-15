import pytest
import torch
from sparsevila.pruning.encoder_pruner import prune_visual_tokens


def test_ratio_zero_returns_all():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.0)
    assert pruned.shape == hidden.shape
    assert torch.equal(idx, torch.arange(100))


def test_ratio_half_returns_half():
    hidden = torch.randn(1, 100, 16)
    sal = torch.arange(100, dtype=torch.float32)   # monotonic
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.5)
    assert pruned.shape[1] == 50
    # Higher salience indices should be kept (50..99)
    assert idx.min().item() >= 50


def test_kept_idx_sorted_ascending():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    _, idx = prune_visual_tokens(hidden, sal, ratio=0.3)
    assert torch.all(idx[1:] >= idx[:-1])


def test_ratio_one_raises():
    hidden = torch.randn(1, 10, 4)
    sal = torch.rand(10)
    with pytest.raises(ValueError):
        prune_visual_tokens(hidden, sal, ratio=1.0)


def test_all_zero_salience_falls_back_to_topk():
    hidden = torch.randn(1, 10, 4)
    sal = torch.zeros(10)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.5)
    # quantile path gives 0 tokens -> fallback gives ~5
    assert pruned.shape[1] == 5
    assert idx.numel() == 5


def test_pruned_hidden_matches_gathered():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.4)
    assert torch.equal(pruned[:, :, :], hidden[:, idx, :])


def test_fp16_salience_supported():
    # Real LLaVA-1.5 inference produces FP16 salience, but torch.quantile
    # only accepts float/double. Ensure the pruner handles FP16 gracefully.
    hidden = torch.randn(1, 100, 16, dtype=torch.float16)
    sal = torch.rand(100, dtype=torch.float16)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.5)
    assert pruned.shape[1] == 50
    assert pruned.dtype == torch.float16
