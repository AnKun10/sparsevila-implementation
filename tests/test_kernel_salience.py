import torch
import pytest
from sparsevila.kernels.salience import column_salience


def test_naive_path_shape():
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 16, 4)
    out = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    assert out.shape == (1, 2, 16)


def test_naive_matches_manual_softmax_sum():
    q = torch.randn(1, 1, 3, 4, dtype=torch.float32)
    k = torch.randn(1, 1, 5, 4, dtype=torch.float32)
    out = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    # Manual reference
    import math
    scale = 1.0 / math.sqrt(4)
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    ref = attn.sum(dim=2)        # sum over queries -> (B, H, N)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.cuda
def test_flash_matches_naive_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q = torch.randn(1, 2, 16, 8, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 8, device="cuda", dtype=torch.float16)
    out_f = column_salience(q, k, reduction="sum", is_causal=False, use_flash=True)
    out_n = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    assert torch.allclose(out_f, out_n, atol=1e-2, rtol=1e-2)
