from dataclasses import dataclass, replace
import torch


@dataclass
class PackedKV:
    """A non-destructive view onto a layer's KV with visual-span subset selected.

    Holds tensors that share NO storage with the source cache after construction
    (constructed via index_select + cat, which always allocate).
    """
    k: torch.Tensor                  # (B, H, S', D)
    v: torch.Tensor                  # (B, H, S', D)
    layer: int
    visual_keep: torch.Tensor        # (K',) indices into ORIGINAL visual span
    visual_range: tuple[int, int]    # (v_start, v_end) in original cache coords
    position_ids: torch.Tensor       # (B, S')

    def append_new_kv(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "PackedKV":
        """Return a new PackedKV with `k_new`/`v_new` concatenated on the seq dim.

        Used during the generation loop to extend the packed view with each
        new token's KV.
        """
        new_len = self.k.shape[2] + k_new.shape[2]
        next_pos = torch.arange(
            self.position_ids[0, -1].item() + 1,
            self.position_ids[0, -1].item() + 1 + k_new.shape[2],
            dtype=self.position_ids.dtype,
            device=self.position_ids.device,
        ).unsqueeze(0)
        return replace(
            self,
            k=torch.cat([self.k, k_new], dim=2),
            v=torch.cat([self.v, v_new], dim=2),
            position_ids=torch.cat([self.position_ids, next_pos], dim=1),
        )
