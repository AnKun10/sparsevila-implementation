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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Same as transformers' Llama rotate_half: split last dim, swap halves
    with a sign flip on the first half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rerotate_keys(
    keys: torch.Tensor,
    delta_positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """Re-apply RoPE on a K tensor to shift each token by ``delta_positions``.

    Given keys that were originally rotated at positions ``p_old`` and a
    target position ``p_new = p_old + delta``, the equivalent rotation is
    by angle ``delta * theta`` per pair-dim (RoPE rotations compose
    additively along the position axis):

        K_at_p_new = K_at_p_old @ Rot(delta)

    so we can re-rotate by calling apply_rotary at ``delta`` directly,
    handling negative deltas via the sin sign flip
    (cos is even, sin is odd in position).

    Args:
        keys:             (B, H, S, D) original RoPE'd K tensor.
        delta_positions:  (S,) long; per-token shift. Positive = move later,
                          negative = move earlier.
        rotary_emb:       LlamaRotaryEmbedding instance with cached cos/sin.

    Returns:
        (B, H, S, D) K tensor as if RoPE'd at p_old + delta.
    """
    if keys.ndim != 4:
        raise ValueError(f"keys must be (B, H, S, D), got {tuple(keys.shape)}")
    if delta_positions.ndim != 1 or delta_positions.shape[0] != keys.shape[2]:
        raise ValueError(
            f"delta_positions must be (S,) with S={keys.shape[2]}, "
            f"got {tuple(delta_positions.shape)}"
        )

    abs_delta = delta_positions.abs().to(torch.long).to(keys.device)
    max_needed = int(abs_delta.max().item()) + 1
    # Ensure the rotary cache covers max_needed.
    if max_needed > getattr(rotary_emb, "max_seq_len_cached", 0):
        rotary_emb._set_cos_sin_cache(
            seq_len=max_needed, device=keys.device, dtype=keys.dtype,
        )
    cos_full = rotary_emb.cos_cached.to(keys.dtype)   # (M, D)
    sin_full = rotary_emb.sin_cached.to(keys.dtype)
    cos = cos_full[abs_delta]                          # (S, D)
    sin = sin_full[abs_delta]                          # (S, D)
    # sin is odd in position -> negate for negative deltas.
    sign = torch.where(
        delta_positions >= 0,
        torch.ones((), dtype=keys.dtype, device=keys.device),
        -torch.ones((), dtype=keys.dtype, device=keys.device),
    )
    sin = sin * sign.unsqueeze(-1)
    # Broadcast over (B, H) -> (1, 1, S, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return keys * cos + _rotate_half(keys) * sin
