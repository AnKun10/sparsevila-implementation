"""Unified wrapper around flash_colreduce (Triton) and a naive PyTorch fallback.

Provides one function `column_salience` matching the signature of the kernel
plus a CPU-friendly path used for unit testing and debug.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _naive(
    q: torch.Tensor, k: torch.Tensor, reduction: str, is_causal: bool, scale: float
) -> torch.Tensor:
    m, n = q.shape[2], k.shape[2]
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        c = max(n - m, 0)
        qi = torch.arange(m, device=q.device).view(1, 1, -1, 1)
        ki = torch.arange(n, device=q.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(qi + c < ki, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if reduction == "sum":
        return probs.sum(dim=2)
    if reduction == "mean":
        out = probs.sum(dim=2)
        if is_causal:
            c = max(n - m, 0)
            out[..., :c] /= m
            out[..., c:] /= torch.arange(n - c, 0, -1, device=q.device)
        else:
            out /= m
        return out
    if reduction == "max":
        return probs.max(dim=2).values
    raise ValueError(f"Invalid reduction: {reduction}")


def column_salience(
    q: torch.Tensor,
    k: torch.Tensor,
    reduction: str = "sum",
    is_causal: bool = False,
    scale: float | None = None,
    use_flash: bool = True,
) -> torch.Tensor:
    """Column-wise softmax-attention reduction along the query axis.

    Returns shape (B, H, N) where N = k.shape[2].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if use_flash and q.is_cuda and k.is_cuda:
        try:
            from flash_colreduce import flash_colreduce
        except ImportError:
            return _naive(q, k, reduction, is_causal, scale)
        return flash_colreduce(q, k, reduction=reduction, is_causal=is_causal, scale=scale)

    return _naive(q, k, reduction, is_causal, scale)
