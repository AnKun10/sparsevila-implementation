import pytest
from sparsevila.models.base import VLMAdapter


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        VLMAdapter()


def test_subclass_missing_methods_fails():
    class Incomplete(VLMAdapter):
        def wrap_encoder(self, vt, cfg): return vt
    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_instantiates():
    class Complete(VLMAdapter):
        def wrap_encoder(self, vt, cfg): return vt
        def wrap_llm(self, llm, cfg): return llm
        def get_visual_span(self, input_ids): return (0, 1)
        def build_position_ids(self, sys, vis, txt): return None
    Complete()  # should not raise
