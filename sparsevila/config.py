from dataclasses import dataclass
from typing import Literal

_VALID_STRATEGIES = {"cls", "summary", "mean_intra"}
_VALID_AGGREGATIONS = {"mean", "max"}
_VALID_LLM_QUANT = {"none", "4bit-bnb", "8bit-bnb"}


@dataclass
class SparseVILAConfig:
    encoder_prune_ratio: float = 0.0
    salience_strategy: Literal["cls", "summary", "mean_intra"] = "cls"
    salience_layer_idx: int = -1

    decode_retrieval_ratio: float = 0.0
    decode_head_aggregation: Literal["mean", "max"] = "mean"

    use_flash_kernel: bool = True
    log_kept_indices: bool = False

    # LLM weight quantization. The paper specifies W4A16 via AWQ; the closest
    # off-the-shelf option that integrates cleanly with the official LLaVA
    # model class (LlavaLlamaForCausalLM) is bitsandbytes 4-bit NF4 with
    # double-quant + FP16 compute, which gives the same memory profile and
    # same W4A16 compute pattern. AWQ proper (INT4 + activation-aware scales)
    # would require switching to HF Transformers' LlavaForConditionalGeneration
    # plus autoawq — left as future work.
    quantize_llm: Literal["none", "4bit-bnb", "8bit-bnb"] = "none"

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
        if self.quantize_llm not in _VALID_LLM_QUANT:
            raise ValueError(
                f"quantize_llm must be one of {_VALID_LLM_QUANT}, got {self.quantize_llm}"
            )
