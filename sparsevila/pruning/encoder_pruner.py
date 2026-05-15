"""Encoder-stage query-agnostic visual token pruning (Algorithm 2)."""
from __future__ import annotations
import torch


def prune_visual_tokens(
    hidden_states: torch.Tensor,
    salience: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune the lowest-salience tokens.

    Args:
        hidden_states: (B, S, D) visual token embeddings (no CLS).
        salience:      (S,) salience score per token.
        ratio:         fraction to prune in [0, 1).

    Returns:
        pruned_hidden: (B, K, D) where K = round((1-ratio)*S).
        kept_idx:      (K,) indices into the original S range, sorted ascending.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    if ratio == 0.0:
        return hidden_states, torch.arange(salience.shape[0], device=salience.device)

    s = salience.shape[0]
    threshold = salience.quantile(ratio)
    keep_mask = salience > threshold
    if keep_mask.sum().item() == 0:
        # Degenerate case: all salience ties at the threshold (e.g., all zeros).
        k = max(1, round((1.0 - ratio) * s))
        topk = salience.topk(k).indices
        keep_mask = torch.zeros_like(salience, dtype=torch.bool)
        keep_mask[topk] = True

    kept_idx = keep_mask.nonzero(as_tuple=True)[0].sort().values
    pruned = hidden_states.index_select(1, kept_idx)
    return pruned, kept_idx
