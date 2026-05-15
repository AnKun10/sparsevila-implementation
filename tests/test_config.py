import pytest
from sparsevila.config import SparseVILAConfig


def test_defaults_are_zero_ratios():
    cfg = SparseVILAConfig()
    assert cfg.encoder_prune_ratio == 0.0
    assert cfg.decode_retrieval_ratio == 0.0
    assert cfg.salience_strategy == "cls"
    assert cfg.use_flash_kernel is True


def test_invalid_encoder_ratio_raises():
    with pytest.raises(ValueError, match="encoder_prune_ratio"):
        SparseVILAConfig(encoder_prune_ratio=1.0)
    with pytest.raises(ValueError, match="encoder_prune_ratio"):
        SparseVILAConfig(encoder_prune_ratio=-0.1)


def test_invalid_decode_ratio_raises():
    with pytest.raises(ValueError, match="decode_retrieval_ratio"):
        SparseVILAConfig(decode_retrieval_ratio=1.0)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="salience_strategy"):
        SparseVILAConfig(salience_strategy="invalid")


def test_invalid_aggregation_raises():
    with pytest.raises(ValueError, match="decode_head_aggregation"):
        SparseVILAConfig(decode_head_aggregation="median")


def test_default_quant_is_none():
    assert SparseVILAConfig().quantize_llm == "none"


def test_valid_quant_values():
    for v in ("none", "4bit-bnb", "8bit-bnb"):
        assert SparseVILAConfig(quantize_llm=v).quantize_llm == v


def test_invalid_quant_raises():
    with pytest.raises(ValueError, match="quantize_llm"):
        SparseVILAConfig(quantize_llm="int4-awq")
    with pytest.raises(ValueError, match="quantize_llm"):
        SparseVILAConfig(quantize_llm="fp8")
