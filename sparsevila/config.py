from dataclasses import dataclass
from typing import Literal

_VALID_STRATEGIES = {"cls", "summary", "mean_intra"}
_VALID_AGGREGATIONS = {"mean", "max"}


@dataclass
class SparseVILAConfig:
    encoder_prune_ratio: float = 0.0
    salience_strategy: Literal["cls", "summary", "mean_intra"] = "cls"
    salience_layer_idx: int = -1

    decode_retrieval_ratio: float = 0.0
    decode_head_aggregation: Literal["mean", "max"] = "mean"

    use_flash_kernel: bool = True
    log_kept_indices: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.encoder_prune_ratio < 1.0:
            raise ValueError(
                f"encoder_prune_ratio must be in [0, 1), got {self.encoder_prune_ratio}"
            )
        if not 0.0 <= self.decode_retrieval_ratio < 1.0:
            raise ValueError(
                f"decode_retrieval_ratio must be in [0, 1), got {self.decode_retrieval_ratio}"
            )
        if self.salience_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"salience_strategy must be one of {_VALID_STRATEGIES}, got {self.salience_strategy}"
            )
        if self.decode_head_aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"decode_head_aggregation must be one of {_VALID_AGGREGATIONS}, "
                f"got {self.decode_head_aggregation}"
            )
