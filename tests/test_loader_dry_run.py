import pytest
from unittest.mock import MagicMock, patch
from sparsevila import load_sparse_llava, SparseVILAConfig


def test_loader_returns_model_and_processor_with_wrapped_encoder():
    fake_model = MagicMock()
    fake_model.get_vision_tower.return_value = MagicMock()
    fake_model.model.layers = []   # tells wrap_llm there are 0 layers to patch
    fake_processor = MagicMock()
    fake_tokenizer = MagicMock()

    with patch("sparsevila.loader._load_llava_pretrained",
               return_value=(fake_tokenizer, fake_model, fake_processor, 4096)):
        model, processor = load_sparse_llava(
            "fake/llava-v1.5-7b",
            config=SparseVILAConfig(encoder_prune_ratio=0.3),
        )

    assert model is fake_model
    assert processor is fake_processor
    # Encoder should have been replaced
    fake_model.get_vision_tower.assert_called()


def test_loader_invalid_config_raises():
    with pytest.raises(ValueError):
        load_sparse_llava("any", config=SparseVILAConfig(encoder_prune_ratio=1.5))
