import torch


def build_compressed_position_ids(
    sys_len: int,
    kept_visual_count: int,
    text_len: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Contiguous position IDs spanning [0, sys_len + kept_visual_count + text_len).

    Implements paper Section 3.2 'compressed' strategy: after pruning visual
    tokens at the encoder, the LLM-side position grid is rebuilt as if the
    visual span only ever had `kept_visual_count` tokens. Subsequent text
    positions continue immediately after.

    Returns shape (1, total_len), dtype long.
    """
    total = sys_len + kept_visual_count + text_len
    return torch.arange(0, total, dtype=torch.long, device=device).unsqueeze(0)
