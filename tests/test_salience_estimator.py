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
