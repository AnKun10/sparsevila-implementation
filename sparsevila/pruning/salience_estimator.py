"""Salience score estimators (Algorithm 2 in the paper).

Three strategies cover the encoder families used by the paper:
- "cls":        CLIP-style, single CLS summary token at index 0.
- "summary":    RADIO-style, multiple summary tokens at the start.
- "mean_intra": SigLIP/QwenVL-style, no summary token.

Input: attention tensor (B, H, S_full, S_full) where S_full includes summary
tokens (if any). Output: salience vector (S_patches,) over the PATCH tokens
only (summary positions excluded).
"""
from __future__ import annotations
import torch


_VALID = {"cls", "summary", "mean_intra"}


def compute_salience_from_attn(
    attn: torch.Tensor,
    strategy: str = "cls",
    num_summary_tokens: int = 1,
) -> torch.Tensor:
    if strategy not in _VALID:
        raise ValueError(f"strategy must be one of {_VALID}, got {strategy}")

    if strategy == "cls":
        # attn: (B, H, S+1, S+1). CLS at index 0 as QUERY -> patch tokens as KEYS.
        cls_row = attn[..., 0, 1:]                  # (B, H, S)
        sal = cls_row.mean(dim=1).sum(dim=0)        # mean heads, sum batch -> (S,)
        return sal

    if strategy == "summary":
        # First `num_summary_tokens` positions are summary tokens.
        # Each summary acts as QUERY (row), attending to patch tokens (cols).
        n = num_summary_tokens
        sum_rows = attn[..., :n, n:]                # (B, H, n, S)
        sal_per_head = sum_rows.mean(dim=2)         # (B, H, S) — mean over summaries
        sal = sal_per_head.mean(dim=1).sum(dim=0)   # mean heads, sum batch
        return sal

    if strategy == "mean_intra":
        # No summary; salience = how often each patch is attended to (column-mean
        # across all queries), averaged over heads.
        n = num_summary_tokens
        intra = attn[..., n:, n:]                   # (B, H, S, S)
        sal_per_head = intra.mean(dim=2)            # (B, H, S) — col-mean
        sal = sal_per_head.mean(dim=1).sum(dim=0)
        return sal
