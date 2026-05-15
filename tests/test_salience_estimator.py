import torch
import pytest
from sparsevila.pruning.salience_estimator import compute_salience_from_attn


def test_cls_strategy_shape():
    # (B, H, S+1, S+1) attention with CLS at index 0
    attn = torch.softmax(torch.randn(1, 8, 577, 577), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="cls")
    assert sal.shape == (576,)


def test_cls_strategy_excludes_cls_self():
    # Build attn where CLS attends purely to itself; then to one specific token.
    attn = torch.zeros(1, 1, 5, 5)
    attn[0, 0, 0, 0] = 0.5     # CLS -> CLS
    attn[0, 0, 0, 2] = 0.5     # CLS -> token at idx 2 (-> patch index 1)
    sal = compute_salience_from_attn(attn, strategy="cls")
    assert sal.shape == (4,)
    # patch idx 0,2,3 should have 0 salience; idx 1 should be 0.5
    assert sal[0].item() == 0.0
    assert sal[1].item() == 0.5
    assert sal[2].item() == 0.0
    assert sal[3].item() == 0.0


def test_cls_strategy_mean_over_heads():
    # Two heads, head 0 attends CLS->idx1, head 1 attends CLS->idx2
    attn = torch.zeros(1, 2, 5, 5)
    attn[0, 0, 0, 2] = 1.0
    attn[0, 1, 0, 3] = 1.0
    sal = compute_salience_from_attn(attn, strategy="cls")
    # idx1 = 0.5 (head 0 contrib / 2), idx2 = 0.5
    assert abs(sal[1].item() - 0.5) < 1e-6
    assert abs(sal[2].item() - 0.5) < 1e-6


def test_invalid_strategy_raises():
    attn = torch.softmax(torch.randn(1, 1, 5, 5), dim=-1)
    with pytest.raises(ValueError):
        compute_salience_from_attn(attn, strategy="bogus")


def test_summary_strategy_shape():
    # 4 summary tokens at front, 100 patch tokens
    attn = torch.softmax(torch.randn(1, 4, 104, 104), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="summary", num_summary_tokens=4)
    assert sal.shape == (100,)


def test_summary_strategy_mean_of_summary_columns():
    # 2 summary tokens, 3 patches. Build attn so summary tokens attend equally
    # to all patches with known weights.
    attn = torch.zeros(1, 1, 5, 5)
    # summary 0 -> patch 0 (col 2) = 0.4
    attn[0, 0, 0, 2] = 0.4
    # summary 1 -> patch 0 = 0.2
    attn[0, 0, 1, 2] = 0.2
    sal = compute_salience_from_attn(attn, strategy="summary", num_summary_tokens=2)
    # patch 0 salience = mean over 2 summaries = (0.4 + 0.2)/2 = 0.3
    assert abs(sal[0].item() - 0.3) < 1e-6


def test_mean_intra_strategy_shape():
    attn = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="mean_intra", num_summary_tokens=0)
    assert sal.shape == (16,)


def test_mean_intra_uniform_attention_gives_uniform_salience():
    # Uniform attention -> all tokens equally salient
    attn = torch.full((1, 1, 8, 8), 1.0 / 8)
    sal = compute_salience_from_attn(attn, strategy="mean_intra", num_summary_tokens=0)
    assert torch.allclose(sal, sal[0].expand_as(sal), atol=1e-6)
