# SparseVILA Reference Implementation — Design Spec

**Date:** 2026-05-15
**Target paper:** SparseVILA — Decoupling Visual Sparsity for Efficient VLM Inference (ICCV 2025), arXiv:2510.17777
**Status:** Approved by user, ready for plan

## 1. Goal & Scope

### 1.1 Goal
Build a clean reference implementation of SparseVILA's two-stage decoupled sparsity (query-agnostic encoder pruning + query-aware decode retrieval) on LLaVA-1.5-7B, suitable for a thesis (DATN). Priority is correctness, readability, and demonstration that the algorithm works end-to-end. Exact paper latency reproduction is not a goal.

### 1.2 In scope
- Algorithm 2: query-agnostic visual token pruning at encoder output, using CLIP CLS-token salience.
- Algorithm 3: query-aware decode-time KV retrieval, per LLM layer, using `flash_colreduce` kernel.
- Unified RoPE position adjustment for the compressed visual span (LLaVA-NeXT/1.5 style).
- A `SparseCache` subclass of `transformers.DynamicCache` with snapshot/reset/select APIs (the snapshot/reset surface is wired but multi-turn flow is deferred).
- Adapter for LLaVA-1.5-7B loaded via the official `haotian-liu/LLaVA` repo as a git submodule.
- A Colab-friendly demo notebook plus a CLI inference script.
- Unit, integration, smoke, and golden-equivalence tests.

### 1.3 Out of scope (explicitly deferred)
- Multi-turn conversation evaluation (Algorithm 4) — user deferred.
- Accuracy benchmarks (POPE, GQA, etc.) — deferred.
- Latency benchmarks vs vanilla — deferred.
- Quantization (AWQ, SmoothQuant) — deferred, FP16 only.
- LLaVA-NeXT anyres tiling — deferred, adapter pattern leaves room to add it.
- Qwen2.5-VL / multimodal RoPE — deferred.
- vast.ai onstart script + Docker — deferred.

### 1.4 Success criteria
1. With `SparseVILAConfig(encoder_prune_ratio=0, decode_retrieval_ratio=0)`, output token IDs match vanilla LLaVA-1.5 byte-by-byte for greedy decoding on a fixed image+prompt (50 tokens). This is the golden test.
2. With non-zero ratios, the model produces semantically coherent output on a sample image (smoke test). Exact match not required.
3. All unit tests pass. End-to-end smoke test passes on CUDA.
4. Code is loadable as a Python package and runs in a Colab notebook after `pip install -e .`.

## 2. Architecture & component map

### 2.1 Module layout
```
[man] SparseVILA/
├── flash-colreduce/             # existing, untouched
├── sparsevila/
│   ├── __init__.py              # exports load_sparse_llava, SparseVILAConfig
│   ├── config.py                # SparseVILAConfig dataclass
│   ├── kernels/
│   │   └── salience.py          # wraps flash_colreduce + naive fallback
│   ├── pruning/
│   │   ├── salience_estimator.py  # 3 strategies: cls / summary / mean_intra
│   │   └── encoder_pruner.py      # gather hidden_states by kept_idx
│   ├── retrieval/
│   │   ├── decode_retriever.py    # per-layer query-aware selection
│   │   └── packed_kv.py           # PackedKV dataclass + append_new_kv
│   ├── cache/
│   │   └── sparse_cache.py        # subclass DynamicCache
│   ├── rope/
│   │   └── unified_rope.py        # compressed-span position_ids
│   └── models/
│       ├── base.py                # VLMAdapter ABC
│       └── llava_15.py            # LLaVA-1.5 specific adapter
├── third_party/
│   └── LLaVA/                   # git submodule, pinned commit
├── notebooks/
│   └── colab_demo.ipynb
├── scripts/
│   ├── setup_env.sh             # skeleton for later vast.ai use
│   └── run_inference.py
├── tests/
│   ├── test_salience_estimator.py
│   ├── test_encoder_pruner.py
│   ├── test_decode_retriever.py
│   ├── test_sparse_cache.py
│   ├── test_unified_rope.py
│   └── test_end_to_end_smoke.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-15-sparsevila-impl-design.md   # this file
├── pyproject.toml
├── README.md
└── .gitignore
```

### 2.2 Component responsibilities (one job per unit)

| Component | Input | Output | Depends on |
|---|---|---|---|
| `salience_estimator` | encoder attn maps, strategy | salience scores per token | torch |
| `encoder_pruner` | hidden_states + scores + ratio | pruned hidden_states + kept_idx | salience_estimator |
| `decode_retriever` | q_proj of new query, k_cache visual slice, ratio | PackedKV per layer | flash_colreduce |
| `sparse_cache` | inherits DynamicCache | + `select_visual` + `snapshot/reset_to` | transformers |
| `unified_rope` | sys_len, kept_visual_count, text_len | adjusted position_ids | torch |
| `llava_15` adapter | SparseVILAConfig | wrapped model with sparsity injected | all above |

### 2.3 Invariants between units
- `salience_estimator` computes scores only — does not decide what to prune.
- `encoder_pruner` decides + gathers — does not know about model internals beyond `(B, S, D)` shape.
- `decode_retriever` produces a `PackedKV` view — does not mutate the cache.
- `sparse_cache` provides truncation and selection APIs — knows nothing about salience.
- `unified_rope` does position math — knows nothing about the model.

Each unit is testable in isolation with small synthetic fixtures.

## 3. Prefill stage data flow (query-agnostic pruning)

### 3.1 Sequence
```
image (B, 3, 336, 336)
    │
    ▼
CLIPVisionTower.forward(image, output_attentions=True)
    │   hidden_states  (B, 577, 1024)        # 576 patches + 1 CLS
    │   attn_weights   list of (B, H, 577, 577) per layer
    │
    ▼
salience_estimator.from_clip(attn_weights[salience_layer_idx])
    │   - extract CLS-as-query row: attn[:, :, 0, 1:]
    │     (how much CLS attends to each patch token → contribution to global embedding)
    │   - mean over heads → (B, 576)
    │   - sum over batch (B=1 typically) → (576,)
    │
    ▼
encoder_pruner.prune(hidden_states[:, 1:, :], salience, ratio)
    │   - threshold = salience.quantile(ratio)
    │   - kept_mask = salience > threshold
    │   - if kept_mask.sum() == 0 → fallback top-k
    │   - kept_idx = kept_mask.nonzero().squeeze(-1).sort().values
    │   - pruned_hidden = hidden_states[:, kept_idx + 1, :]   # +1 to skip CLS row
    │
    ▼
mm_projector(pruned_hidden) → (B, K, 4096)
    │
    ▼
LLM input: concat(sys_embed, pruned_visual, query_embed)
position_ids = unified_rope.adjust(sys_len, K, query_len)
    │
    ▼
LlamaForCausalLM.forward(... position_ids=..., past_key_values=SparseCache())
    │
    ▼
cache.snapshot() → anchor   # snapshot saved even though v1 doesn't use it
```

### 3.2 Decision: which encoder layer for salience?
- Paper: final-layer attention.
- LLaVA-1.5 convention: the projector consumes hidden states from layer `-2` (penultimate). The salience signal we use comes from the attention map at the layer specified by `config.salience_layer_idx` (default `-1`, paper's choice).
- These two layers can differ — that is intentional and matches the paper.

### 3.3 Decision: flash kernel vs naive on encoder side
- v1 default: `use_flash_kernel=True`. When `True`, we hook `CLIPAttention.forward` to call `flash_colreduce(q, k, reduction="mean", is_causal=False)` and skip materializing the full attention matrix.
- When `False`, fall back to `output_attentions=True` and compute salience from the dense attention tensor with PyTorch ops.
- Both paths must produce identical salience scores (`atol=1e-3`) — enforced by a test.

### 3.4 Sample shapes (LLaVA-1.5-7B)
```
Image:                   (1, 3, 336, 336)
Encoder output:          (1, 577, 1024)         # +1 CLS
Salience:                (576,)
Kept @ ratio=0.5:        (288, 1024)
After projector:         (288, 4096)
Visual span in LLM:      288 tokens (vs 576 vanilla)
```

### 3.5 Edge cases
- `ratio == 0`: bypass pruning entirely. Output identical to vanilla.
- `ratio >= 1`: raised in config `__post_init__`.
- Quantile produces zero tokens (all salience tied): fallback to top-k with `k = max(1, round((1-ratio)*S))`.
- Batch size B > 1: assume same kept count per sample (paper default), but indices may differ. v1 supports B=1 only; B>1 raises NotImplementedError.

## 4. Decode stage data flow (query-aware retrieval)

### 4.1 Sequence
```
user query tokens Q
    │
    ▼
LLM prefill(Q) appends to cache  → cache = [sys | visual_pruned | Q]
                                    v_start = sys_len
                                    v_end   = sys_len + K
    │
    ▼
For each layer l in range(L):
    q_proj_l  = LlamaAttention[l].q_proj applied to hidden_Q
                shape (B, H, |Q|, D_h)
    k_visual_l = cache.key_cache[l][:, :, v_start:v_end, :]
                shape (B, H, K, D_h)
    │
    salience_l = flash_colreduce(
        q_proj_l, k_visual_l,
        reduction="sum", is_causal=False
    )                                # shape (B, H, K)
    aggregated_l = salience_l.mean(dim=1)        # shape (B, K), head-mean
    │
    threshold = aggregated_l.quantile(decode_ratio)
    keep_mask = aggregated_l > threshold
    if keep_mask.sum() == 0:
        topk_idx = aggregated_l.topk(max(1, round((1-decode_ratio)*K))).indices
        keep_mask.scatter_(1, topk_idx, True)
    keep_idx_l = keep_mask.nonzero(as_tuple=True)[-1].sort().values
    │
    packed_kv[l] = SparseCache.select_visual(
        layer_idx=l,
        visual_range=(v_start, v_end),
        keep_idx=keep_idx_l
    )                                # contains contiguous [sys | kept_visual | Q]

generation loop:
    for step in range(max_new_tokens):
        for l in range(L):
            attn_out_l = scaled_dot_product_attention(
                q_new_token_l, packed_kv[l].k, packed_kv[l].v)
            # Append new_token KV both to cache.key_cache[l] AND to packed_kv[l]
        next_token = greedy_pick(logits)
```

### 4.2 Decisions
- Per-layer selection (paper Algorithm 3).
- Mean over heads (paper default). `decode_head_aggregation="max"` allowed but not v1 default.
- Quantile threshold with top-k fallback (matches paper Algorithm 3 + handles degenerate case).
- Append generated-token KV to **both** the original cache and each `packed_kv[l]` so multi-turn API stays consistent.
- Visual span assumed contiguous in input_ids. Multi-image inputs deferred.

### 4.3 RoPE during decode
- K-tensors in the cache already have RoPE applied (during prefill at adjusted positions).
- `index_select` preserves the encoded values exactly — no re-RoPE needed.
- Q of the new token gets RoPE applied at position `sys_len + K + |Q| + step` (continuation of the compressed-span positions established during prefill).

### 4.4 Sample shapes
```
V = K = 288 (after encoder pruning)
decode_ratio = 0.75 → keep 72 visual KV per layer
k_visual_l:        (1, 32, 288, 128)
salience_l:        (1, 32, 288)
aggregated_l:      (1, 288)
keep_idx_l:        (72,)
packed_kv[l].k:    (1, 32, sys_len + 72 + |Q|, 128)
```

### 4.5 Edge cases
- `decode_retrieval_ratio == 0`: bypass; packed_kv is full visual slice.
- `|Q| == 0`: raise — cannot compute query-aware salience.
- `K == 0` (all pruned at encoder): bypass retrieval, packed_kv has empty visual block.

## 5. KV cache & RoPE

### 5.1 SparseCache API
Subclasses `transformers.DynamicCache`. Adds:
- `snapshot() -> CacheAnchor`: records per-layer sequence lengths.
- `reset_to(anchor)`: slices each layer's tensors back to the anchor lengths. Non-destructive to layers that already match.
- `select_visual(layer_idx, visual_range, keep_idx) -> PackedKV`: builds a new tensor `[k[:, :, :v_start], k.index_select(2, v_start+keep_idx), k[:, :, v_end:]]` concatenated; returns a `PackedKV`. Never mutates `self`.

`select_visual` returns tensors that share no storage with `self.key_cache[layer_idx]` after the index_select operation (verified by `data_ptr()` check in tests).

### 5.2 PackedKV
```python
@dataclass
class PackedKV:
    k: torch.Tensor            # (B, H, S', D)
    v: torch.Tensor            # (B, H, S', D)
    layer: int
    visual_keep: torch.Tensor  # (K',) indices into original visual span
    visual_range: tuple[int,int]
    position_ids: torch.Tensor # (B, S') matching k/v positions

    def append_new_kv(self, k_new, v_new) -> "PackedKV":
        # Returns a new PackedKV with k_new/v_new concatenated on dim=2
        ...
```

### 5.3 Unified RoPE adjuster
For LLaVA-1.5 (and -NeXT, which we don't target in v1 but the math is identical):

```python
def adjust_position_ids(sys_len, kept_visual_count, text_len):
    return torch.arange(0, sys_len + kept_visual_count + text_len).unsqueeze(0)
```

This implements the "compressed" strategy: the visual span occupies positions `[sys_len, sys_len + kept_visual_count)` regardless of how many tokens were pruned. Text positions continue immediately after.

For generated tokens during decode, the LLM's standard generation loop computes `position_ids = past_length + step`, where `past_length` is read from the cache. Because the cache only ever held the compressed-span positions, `past_length` is already correct.

## 6. Public API

### 6.1 Entry point
```python
from sparsevila import load_sparse_llava, SparseVILAConfig

config = SparseVILAConfig(
    encoder_prune_ratio=0.50,
    decode_retrieval_ratio=0.75,
    salience_strategy="cls",
    use_flash_kernel=True,
)
model, processor = load_sparse_llava(
    "liuhaotian/llava-v1.5-7b",
    config=config,
    dtype=torch.float16,
    device="cuda",
)
inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200)
```

### 6.2 SparseVILAConfig
```python
@dataclass
class SparseVILAConfig:
    encoder_prune_ratio: float = 0.0
    salience_strategy: Literal["cls", "summary", "mean_intra"] = "cls"
    salience_layer_idx: int = -1

    decode_retrieval_ratio: float = 0.0
    decode_head_aggregation: Literal["mean", "max"] = "mean"

    use_flash_kernel: bool = True
    log_kept_indices: bool = False

    def __post_init__(self):
        if not 0 <= self.encoder_prune_ratio < 1:
            raise ValueError(...)
        if not 0 <= self.decode_retrieval_ratio < 1:
            raise ValueError(...)
        # ... strategy + aggregation enum checks
```

All ratios default to 0 → loading with default config reproduces vanilla LLaVA output (golden-test invariant).

### 6.3 Adapter ABC
```python
class VLMAdapter(ABC):
    @abstractmethod
    def wrap_encoder(self, vision_tower, config): ...
    @abstractmethod
    def wrap_llm(self, llm, config): ...
    @abstractmethod
    def get_visual_span(self, input_ids) -> tuple[int, int]: ...
    @abstractmethod
    def build_position_ids(self, sys_len, vis_len, text_len): ...
```

`LlavaFifteenAdapter` (in `models/llava_15.py`) is the v1 implementation.

## 7. Testing strategy

### 7.1 Test pyramid
- Unit tests (~20): one fixture per component, no model load.
- Integration tests (3–5): pairs of components with synthetic tensors.
- E2E smoke (1): load real LLaVA-1.5-7B, one forward pass.
- Golden test (1): vanilla LLaVA-1.5 output == SparseVILA with ratio=0 output, 50 tokens, greedy.

### 7.2 Key unit tests
- `test_salience_estimator.py`: shape, CLS exclusion, kernel-vs-naive equivalence, zero-attention edge.
- `test_encoder_pruner.py`: ratio bounds, kept_idx sorted, count matches `(1-ratio)*S`, per-tile consistency (no-op in v1 but tested for future).
- `test_decode_retriever.py`: quantile count, top-k fallback on ties, packed_kv non-aliasing (data_ptr check), per-layer independence.
- `test_sparse_cache.py`: snapshot/reset round-trip, select_visual non-destructive, reset to empty anchor, out-of-range indices raise.
- `test_unified_rope.py`: contiguity, dimensions, dtype.

### 7.3 Golden test
```python
@pytest.mark.slow
def test_zero_config_matches_vanilla():
    # Load vanilla LLaVA-1.5 and SparseVILA with all-zero config.
    # Run greedy decode 50 tokens on a fixed image+prompt.
    # Assert output_ids identical.
```

This is the single most important regression guard.

### 7.4 Error handling
- Validate at config construction time (fail loud, early).
- Internal boundaries trust each other — no re-validation.
- One sanctioned silent-fallback: quantile producing zero tokens → top-k. Logs a warning once per session.
- No broad `except Exception:` anywhere.

### 7.5 Logging
- Logger name: `sparsevila` and sub-loggers per submodule.
- Default level: WARNING.
- `log_kept_indices=True` raises sparsevila.pruning + sparsevila.retrieval to INFO.

## 8. Setup & deployment

### 8.1 Dependencies
```toml
[project]
dependencies = [
    "torch>=2.1",
    "transformers>=4.40,<4.50",
    "triton>=3.0",
    "flash-colreduce",
    "Pillow",
    "sentencepiece",
    "accelerate>=0.27",
    "einops",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-mock", "ruff"]
notebook = ["jupyter", "ipywidgets"]
```

LLaVA is pulled as a git submodule under `third_party/LLaVA`, not via PyPI. A README section documents the pinned commit.

### 8.2 Colab notebook flow
```
Cell 1:  !git clone <repo-url>; %cd "[man] SparseVILA"
Cell 2:  !git submodule update --init --recursive
Cell 3:  !pip install -e ./flash-colreduce
Cell 4:  !pip install -e .
Cell 5:  from sparsevila import load_sparse_llava, SparseVILAConfig
Cell 6:  # Load model, run inference, compare vanilla vs sparse outputs
```

### 8.3 vast.ai (deferred)
Skeleton `scripts/setup_env.sh` reserved for future onstart script + Docker base image selection.

## 9. Risks & open questions

| Risk | Mitigation |
|---|---|
| LLaVA repo deps pin old transformers; conflict with newer features | Pin transformers to `>=4.40,<4.50`. Document compat in README. |
| Windows-native install of LLaVA + Triton may fail | Document WSL/Colab/vast.ai as supported environments. Local Windows not officially supported. |
| Hooking CLIPAttention to swap in flash_colreduce may break gradient flow | We run inference only (`torch.inference_mode()`); no gradients needed. Document this. |
| `output_attentions=True` slow path in naive mode | Naive mode is for debug only; flash kernel is the default. |

## 10. Non-goals reminder

- This is a **reference** implementation. It does not aim to match paper latency.
- Quantization, multi-turn, accuracy benchmarks are all **deferred to follow-up phases**, not negotiated away.
- Code clarity > raw performance throughout.
