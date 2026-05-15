import torch
from sparsevila.config import SparseVILAConfig
from sparsevila.cache.sparse_cache import SparseCache
from sparsevila.models.llava_15 import (
    LlavaFifteenAdapter, SparseInferenceSession, SparseLlavaContext,
)


def test_session_enter_attaches_packed_kvs_to_llm():
    class FakeLLM:
        def __init__(self):
            self.model = type("M", (), {"layers": [None, None]})()
            self._sparsevila_ctx = None
            self._sparsevila_packed_kvs = None

    cfg = SparseVILAConfig(decode_retrieval_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    llm = FakeLLM()
    cache = SparseCache()
    seq_len = 12
    for l in range(2):
        cache.update(torch.randn(1, 2, seq_len, 4), torch.randn(1, 2, seq_len, 4), l)

    ctx = SparseLlavaContext(
        visual_range=(2, 10),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        cache=cache,
        q_proj_per_layer=[torch.randn(1, 2, 3, 4) for _ in range(2)],
        config=cfg,
    )

    with SparseInferenceSession(llm, adapter, ctx) as session:
        assert llm._sparsevila_ctx is ctx
        assert llm._sparsevila_packed_kvs is not None
        assert len(llm._sparsevila_packed_kvs) == 2

    # On exit, both fields cleared
    assert llm._sparsevila_ctx is None
    assert llm._sparsevila_packed_kvs is None
