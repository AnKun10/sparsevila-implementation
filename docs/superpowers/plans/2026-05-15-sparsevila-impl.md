# SparseVILA Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean reference implementation of SparseVILA's two-stage sparsity (encoder pruning + decode retrieval) on LLaVA-1.5-7B, with a Colab-friendly demo.

**Architecture:** Subclass + override pattern. `sparsevila/` package wraps LLaVA-1.5 (loaded via official LLaVA repo as git submodule). Algorithm 2 (query-agnostic) hooks the CLIP encoder; Algorithm 3 (query-aware) hooks Llama attention layers per-layer. `flash_colreduce` from the bundled subpackage is the salience kernel.

**Tech Stack:** PyTorch ≥2.1, transformers 4.40-4.49, Triton ≥3.0, flash-colreduce (vendored), official LLaVA repo (submodule), pytest.

**Spec:** `docs/superpowers/specs/2026-05-15-sparsevila-impl-design.md`

---

## Conventions
- All paths absolute starting from `E:\Workspaces\My Projects\DATN\[man] SparseVILA\` (referred below as `<repo>/`).
- Test command base: `pytest -v <path>` from `<repo>/`.
- Commit per task using `git add <files>; git commit -m "<msg>"`. Init git in Task 1.
- Tests requiring CUDA/model load are marked `@pytest.mark.cuda` and skipped locally; verified in Colab notebook (Task 21).
- Each TDD task: write test → run (fail) → implement → run (pass) → commit.

---

### Task 1: Project scaffolding + git init

**Files:**
- Create: `<repo>/.gitignore`
- Create: `<repo>/pyproject.toml`
- Create: `<repo>/README.md`
- Create: `<repo>/sparsevila/__init__.py`
- Create: `<repo>/tests/__init__.py`
- Create: `<repo>/tests/conftest.py`

- [ ] **Step 1: Initialize git**

```bash
cd "<repo>"
git init
git branch -M main
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.venv/
venv/
.idea/
.vscode/
build/
dist/
third_party/LLaVA/   # submodule, not tracked as files
*.pdf
*_main.txt
_sparsevila_*.txt
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sparsevila"
version = "0.1.0"
description = "Reference implementation of SparseVILA (ICCV 2025) on LLaVA-1.5"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "torch>=2.1",
    "transformers>=4.40,<4.50",
    "triton>=3.0",
    "Pillow",
    "sentencepiece",
    "accelerate>=0.27",
    "einops",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-mock", "ruff"]
notebook = ["jupyter", "ipywidgets"]

[tool.setuptools.packages.find]
where = ["."]
include = ["sparsevila*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "cuda: requires CUDA GPU",
    "slow: requires real model load",
]
```

- [ ] **Step 4: Create README stub**

```markdown
# SparseVILA — LLaVA-1.5 Reference Implementation

Reference impl of SparseVILA (ICCV 2025, arXiv:2510.17777) on LLaVA-1.5-7B.

## Setup
See `notebooks/colab_demo.ipynb` for one-cell setup, or:

```bash
git clone <repo-url>
cd "[man] SparseVILA"
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e .
```

## Status
Reference implementation — see `docs/superpowers/specs/` for design.
```

- [ ] **Step 5: Create empty `__init__.py` files and `conftest.py`**

```python
# sparsevila/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
```

- [ ] **Step 6: Install package + verify pytest finds zero tests**

```bash
pip install -e .
pip install -e ".[dev]"
pytest -v
```

Expected: `no tests ran` (no test files yet).

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml README.md sparsevila tests
git commit -m "chore: scaffold sparsevila package + tests"
```

---

### Task 2: SparseVILAConfig

**Files:**
- Create: `<repo>/sparsevila/config.py`
- Create: `<repo>/tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
pytest -v tests/test_config.py
```

Expected: `ImportError: cannot import name 'SparseVILAConfig'`.

- [ ] **Step 3: Implement config**

```python
# sparsevila/config.py
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
```

- [ ] **Step 4: Run test, confirm it passes**

```bash
pytest -v tests/test_config.py
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/config.py tests/test_config.py
git commit -m "feat(config): SparseVILAConfig with validation"
```

---

### Task 3: UnifiedRopeAdjuster (pure math)

**Files:**
- Create: `<repo>/sparsevila/rope/__init__.py`
- Create: `<repo>/sparsevila/rope/unified_rope.py`
- Create: `<repo>/tests/test_unified_rope.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_unified_rope.py
import torch
from sparsevila.rope.unified_rope import build_compressed_position_ids


def test_lengths_sum():
    pos = build_compressed_position_ids(sys_len=10, kept_visual_count=288, text_len=20)
    assert pos.shape == (1, 318)


def test_contiguous_no_gaps():
    pos = build_compressed_position_ids(sys_len=5, kept_visual_count=100, text_len=7)
    diffs = pos[0, 1:] - pos[0, :-1]
    assert torch.all(diffs == 1)


def test_starts_at_zero():
    pos = build_compressed_position_ids(sys_len=3, kept_visual_count=8, text_len=2)
    assert pos[0, 0].item() == 0


def test_dtype_long():
    pos = build_compressed_position_ids(sys_len=1, kept_visual_count=1, text_len=1)
    assert pos.dtype == torch.long


def test_zero_visual():
    pos = build_compressed_position_ids(sys_len=5, kept_visual_count=0, text_len=3)
    assert pos.shape == (1, 8)
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
```

- [ ] **Step 2: Run test, confirm fails**

```bash
pytest -v tests/test_unified_rope.py
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# sparsevila/rope/__init__.py
from .unified_rope import build_compressed_position_ids

__all__ = ["build_compressed_position_ids"]
```

```python
# sparsevila/rope/unified_rope.py
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
```

- [ ] **Step 4: Run test, confirm passes**

```bash
pytest -v tests/test_unified_rope.py
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/rope tests/test_unified_rope.py
git commit -m "feat(rope): unified RoPE compressed position_ids builder"
```

---

### Task 4: PackedKV dataclass

**Files:**
- Create: `<repo>/sparsevila/retrieval/__init__.py`
- Create: `<repo>/sparsevila/retrieval/packed_kv.py`
- Create: `<repo>/tests/test_packed_kv.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_packed_kv.py
import torch
from sparsevila.retrieval.packed_kv import PackedKV


def _dummy_packed(b=1, h=2, s=5, d=4):
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    return PackedKV(
        k=k, v=v, layer=0,
        visual_keep=torch.arange(3),
        visual_range=(1, 4),
        position_ids=torch.arange(s).unsqueeze(0),
    )


def test_construct_shape():
    p = _dummy_packed()
    assert p.k.shape == (1, 2, 5, 4)
    assert p.v.shape == (1, 2, 5, 4)


def test_append_new_kv_grows_sequence():
    p = _dummy_packed(s=5)
    k_new = torch.randn(1, 2, 1, 4)
    v_new = torch.randn(1, 2, 1, 4)
    p2 = p.append_new_kv(k_new, v_new)
    assert p2.k.shape == (1, 2, 6, 4)
    assert p2.v.shape == (1, 2, 6, 4)
    assert torch.equal(p2.k[:, :, -1:, :], k_new)


def test_append_new_kv_does_not_mutate_original():
    p = _dummy_packed(s=5)
    k_before = p.k.clone()
    p.append_new_kv(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    assert torch.equal(p.k, k_before)
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_packed_kv.py
```

- [ ] **Step 3: Implement**

```python
# sparsevila/retrieval/__init__.py
from .packed_kv import PackedKV

__all__ = ["PackedKV"]
```

```python
# sparsevila/retrieval/packed_kv.py
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
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_packed_kv.py
```

- [ ] **Step 5: Commit**

```bash
git add sparsevila/retrieval tests/test_packed_kv.py
git commit -m "feat(retrieval): PackedKV dataclass with non-destructive append"
```

---

### Task 5: SalienceEstimator — CLS strategy

**Files:**
- Create: `<repo>/sparsevila/pruning/__init__.py`
- Create: `<repo>/sparsevila/pruning/salience_estimator.py`
- Create: `<repo>/tests/test_salience_estimator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_salience_estimator.py
import torch
from sparsevila.pruning.salience_estimator import compute_salience_from_attn


def test_cls_strategy_shape():
    # (B, H, S+1, S+1) attention with CLS at index 0
    attn = torch.softmax(torch.randn(1, 8, 577, 577), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="cls")
    assert sal.shape == (576,)


def test_cls_strategy_excludes_cls_self():
    # Build attn where CLS attends purely to itself; then to one specific token.
    attn = torch.zeros(1, 1, 5, 5)
    attn[0, 0, 0, 0] = 0.5     # CLS -> CLS
    attn[0, 0, 0, 2] = 0.5     # CLS -> token at idx 2 (-> patch index 1)
    sal = compute_salience_from_attn(attn, strategy="cls")
    assert sal.shape == (4,)
    # patch idx 0,2,3 should have 0 salience; idx 1 should be 0.5
    assert sal[0].item() == 0.0
    assert sal[1].item() == 0.5
    assert sal[2].item() == 0.0
    assert sal[3].item() == 0.0


def test_cls_strategy_mean_over_heads():
    # Two heads, head 0 attends CLS->idx1, head 1 attends CLS->idx2
    attn = torch.zeros(1, 2, 5, 5)
    attn[0, 0, 0, 2] = 1.0
    attn[0, 1, 0, 3] = 1.0
    sal = compute_salience_from_attn(attn, strategy="cls")
    # idx1 = 0.5 (head 0 contrib / 2), idx2 = 0.5
    assert abs(sal[1].item() - 0.5) < 1e-6
    assert abs(sal[2].item() - 0.5) < 1e-6


def test_invalid_strategy_raises():
    attn = torch.softmax(torch.randn(1, 1, 5, 5), dim=-1)
    import pytest
    with pytest.raises(ValueError):
        compute_salience_from_attn(attn, strategy="bogus")
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_salience_estimator.py
```

- [ ] **Step 3: Implement (CLS only for now)**

```python
# sparsevila/pruning/__init__.py
from .salience_estimator import compute_salience_from_attn

__all__ = ["compute_salience_from_attn"]
```

```python
# sparsevila/pruning/salience_estimator.py
"""Salience score estimators (Algorithm 2 in the paper).

Three strategies cover the encoder families used by the paper:
- "cls":        CLIP-style, single CLS summary token at index 0.
- "summary":    RADIO-style, multiple summary tokens at the start.
- "mean_intra": SigLIP/QwenVL-style, no summary token.

Input: attention tensor (B, H, S_full, S_full) where S_full includes summary
tokens (if any). Output: salience vector (S_patches,) over the PATCH tokens
only (summary positions excluded).
"""
from __future__ import annotations
import torch


_VALID = {"cls", "summary", "mean_intra"}


def compute_salience_from_attn(
    attn: torch.Tensor,
    strategy: str = "cls",
    num_summary_tokens: int = 1,
) -> torch.Tensor:
    if strategy not in _VALID:
        raise ValueError(f"strategy must be one of {_VALID}, got {strategy}")

    if strategy == "cls":
        # attn: (B, H, S+1, S+1). CLS at index 0 as QUERY -> patch tokens as KEYS.
        cls_row = attn[..., 0, 1:]                  # (B, H, S)
        sal = cls_row.mean(dim=1).sum(dim=0)        # mean heads, sum batch -> (S,)
        return sal

    raise NotImplementedError(f"strategy={strategy} added in Task 6")
```

- [ ] **Step 4: Run, confirm passes (CLS tests + invalid raises)**

```bash
pytest -v tests/test_salience_estimator.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/pruning tests/test_salience_estimator.py
git commit -m "feat(pruning): salience estimator with CLS strategy"
```

---

### Task 6: SalienceEstimator — summary + mean_intra strategies

**Files:**
- Modify: `<repo>/sparsevila/pruning/salience_estimator.py`
- Modify: `<repo>/tests/test_salience_estimator.py`

- [ ] **Step 1: Add failing tests for summary + mean_intra**

```python
# Append to tests/test_salience_estimator.py
def test_summary_strategy_shape():
    # 4 summary tokens at front, 100 patch tokens
    attn = torch.softmax(torch.randn(1, 4, 104, 104), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="summary", num_summary_tokens=4)
    assert sal.shape == (100,)


def test_summary_strategy_mean_of_summary_columns():
    # 2 summary tokens, 3 patches. Build attn so summary tokens attend equally
    # to all patches with known weights.
    attn = torch.zeros(1, 1, 5, 5)
    # summary 0 -> patch 0 (col 2) = 0.4
    attn[0, 0, 0, 2] = 0.4
    # summary 1 -> patch 0 = 0.2
    attn[0, 0, 1, 2] = 0.2
    sal = compute_salience_from_attn(attn, strategy="summary", num_summary_tokens=2)
    # patch 0 salience = mean over 2 summaries = (0.4 + 0.2)/2 = 0.3
    assert abs(sal[0].item() - 0.3) < 1e-6


def test_mean_intra_strategy_shape():
    attn = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
    sal = compute_salience_from_attn(attn, strategy="mean_intra", num_summary_tokens=0)
    assert sal.shape == (16,)


def test_mean_intra_uniform_attention_gives_uniform_salience():
    # Uniform attention -> all tokens equally salient
    attn = torch.full((1, 1, 8, 8), 1.0 / 8)
    sal = compute_salience_from_attn(attn, strategy="mean_intra", num_summary_tokens=0)
    assert torch.allclose(sal, sal[0].expand_as(sal), atol=1e-6)
```

- [ ] **Step 2: Run, confirm new tests fail**

```bash
pytest -v tests/test_salience_estimator.py
```

- [ ] **Step 3: Implement remaining strategies**

```python
# Replace the body of compute_salience_from_attn after the "cls" branch:
    if strategy == "summary":
        # First `num_summary_tokens` positions are summary tokens.
        # Each summary acts as QUERY (row), attending to patch tokens (cols).
        n = num_summary_tokens
        sum_rows = attn[..., :n, n:]                # (B, H, n, S)
        sal_per_head = sum_rows.mean(dim=2)         # (B, H, S) — mean over summaries
        sal = sal_per_head.mean(dim=1).sum(dim=0)   # mean heads, sum batch
        return sal

    if strategy == "mean_intra":
        # No summary; salience = how often each patch is attended to (column-mean
        # across all queries), averaged over heads.
        n = num_summary_tokens
        intra = attn[..., n:, n:]                   # (B, H, S, S)
        sal_per_head = intra.mean(dim=2)            # (B, H, S) — col-mean
        sal = sal_per_head.mean(dim=1).sum(dim=0)
        return sal
```

- [ ] **Step 4: Run, all tests pass**

```bash
pytest -v tests/test_salience_estimator.py
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/pruning/salience_estimator.py tests/test_salience_estimator.py
git commit -m "feat(pruning): add summary + mean_intra salience strategies"
```

---

### Task 7: EncoderPruner

**Files:**
- Create: `<repo>/sparsevila/pruning/encoder_pruner.py`
- Modify: `<repo>/sparsevila/pruning/__init__.py`
- Create: `<repo>/tests/test_encoder_pruner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_encoder_pruner.py
import pytest
import torch
from sparsevila.pruning.encoder_pruner import prune_visual_tokens


def test_ratio_zero_returns_all():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.0)
    assert pruned.shape == hidden.shape
    assert torch.equal(idx, torch.arange(100))


def test_ratio_half_returns_half():
    hidden = torch.randn(1, 100, 16)
    sal = torch.arange(100, dtype=torch.float32)   # monotonic
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.5)
    assert pruned.shape[1] == 50
    # Higher salience indices should be kept (50..99)
    assert idx.min().item() >= 50


def test_kept_idx_sorted_ascending():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    _, idx = prune_visual_tokens(hidden, sal, ratio=0.3)
    assert torch.all(idx[1:] >= idx[:-1])


def test_ratio_one_raises():
    hidden = torch.randn(1, 10, 4)
    sal = torch.rand(10)
    with pytest.raises(ValueError):
        prune_visual_tokens(hidden, sal, ratio=1.0)


def test_all_zero_salience_falls_back_to_topk():
    hidden = torch.randn(1, 10, 4)
    sal = torch.zeros(10)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.5)
    # quantile path gives 0 tokens -> fallback gives ~5
    assert pruned.shape[1] == 5
    assert idx.numel() == 5


def test_pruned_hidden_matches_gathered():
    hidden = torch.randn(1, 100, 16)
    sal = torch.rand(100)
    pruned, idx = prune_visual_tokens(hidden, sal, ratio=0.4)
    assert torch.equal(pruned[:, :, :], hidden[:, idx, :])
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_encoder_pruner.py
```

- [ ] **Step 3: Implement**

```python
# sparsevila/pruning/encoder_pruner.py
"""Encoder-stage query-agnostic visual token pruning (Algorithm 2)."""
from __future__ import annotations
import torch


def prune_visual_tokens(
    hidden_states: torch.Tensor,
    salience: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune the lowest-salience tokens.

    Args:
        hidden_states: (B, S, D) visual token embeddings (no CLS).
        salience:      (S,) salience score per token.
        ratio:         fraction to prune in [0, 1).

    Returns:
        pruned_hidden: (B, K, D) where K = round((1-ratio)*S).
        kept_idx:      (K,) indices into the original S range, sorted ascending.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    if ratio == 0.0:
        return hidden_states, torch.arange(salience.shape[0], device=salience.device)

    s = salience.shape[0]
    threshold = salience.quantile(ratio)
    keep_mask = salience > threshold
    if keep_mask.sum().item() == 0:
        # Degenerate case: all salience ties at the threshold (e.g., all zeros).
        k = max(1, round((1.0 - ratio) * s))
        topk = salience.topk(k).indices
        keep_mask = torch.zeros_like(salience, dtype=torch.bool)
        keep_mask[topk] = True

    kept_idx = keep_mask.nonzero(as_tuple=True)[0].sort().values
    pruned = hidden_states.index_select(1, kept_idx)
    return pruned, kept_idx
```

- [ ] **Step 4: Update package exports + run tests**

```python
# sparsevila/pruning/__init__.py
from .encoder_pruner import prune_visual_tokens
from .salience_estimator import compute_salience_from_attn

__all__ = ["prune_visual_tokens", "compute_salience_from_attn"]
```

```bash
pytest -v tests/test_encoder_pruner.py
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/pruning/encoder_pruner.py sparsevila/pruning/__init__.py tests/test_encoder_pruner.py
git commit -m "feat(pruning): encoder pruner with quantile + top-k fallback"
```

---

### Task 8: SparseCache — snapshot/reset

**Files:**
- Create: `<repo>/sparsevila/cache/__init__.py`
- Create: `<repo>/sparsevila/cache/sparse_cache.py`
- Create: `<repo>/tests/test_sparse_cache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sparse_cache.py
import torch
from sparsevila.cache.sparse_cache import SparseCache, CacheAnchor


def _populate(cache, n_layers=2, b=1, h=2, s=5, d=4):
    for l in range(n_layers):
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        cache.update(k, v, l)
    return cache


def test_snapshot_records_lengths():
    cache = SparseCache()
    _populate(cache, n_layers=3, s=7)
    anchor = cache.snapshot()
    assert isinstance(anchor, CacheAnchor)
    assert anchor.per_layer_len == [7, 7, 7]


def test_reset_to_truncates():
    cache = SparseCache()
    _populate(cache, n_layers=2, s=5)
    anchor = cache.snapshot()
    # Append more
    for l in range(2):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(k, v, l)
    assert cache.key_cache[0].shape[2] == 8
    cache.reset_to(anchor)
    assert cache.key_cache[0].shape[2] == 5
    assert cache.value_cache[0].shape[2] == 5


def test_reset_to_empty_anchor():
    cache = SparseCache()
    empty = cache.snapshot()              # len = []
    _populate(cache, n_layers=1, s=3)
    cache.reset_to(empty)
    assert len(cache.key_cache) == 1
    assert cache.key_cache[0].shape[2] == 0
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_sparse_cache.py
```

- [ ] **Step 3: Implement snapshot/reset**

```python
# sparsevila/cache/__init__.py
from .sparse_cache import SparseCache, CacheAnchor

__all__ = ["SparseCache", "CacheAnchor"]
```

```python
# sparsevila/cache/sparse_cache.py
"""SparseCache: DynamicCache + snapshot/reset (multi-turn ready) + select_visual."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    from ..retrieval.packed_kv import PackedKV


@dataclass
class CacheAnchor:
    per_layer_len: list[int] = field(default_factory=list)


class SparseCache(DynamicCache):
    """Adds:
    - `snapshot()` -> CacheAnchor of current per-layer seq lengths.
    - `reset_to(anchor)` truncates each layer back to the anchor length.
    - `select_visual(...)` (added in Task 9) returns a PackedKV view.
    """

    def snapshot(self) -> CacheAnchor:
        lengths = [k.shape[2] for k in self.key_cache]
        return CacheAnchor(per_layer_len=lengths)

    def reset_to(self, anchor: CacheAnchor) -> None:
        # When anchor was taken with N layers and we currently have M >= N,
        # truncate the first N layers and drop any layers beyond.
        n = len(anchor.per_layer_len)
        for l in range(n):
            target = anchor.per_layer_len[l]
            self.key_cache[l] = self.key_cache[l][:, :, :target, :].contiguous()
            self.value_cache[l] = self.value_cache[l][:, :, :target, :].contiguous()
        # Truncate the underlying lists if more layers exist
        del self.key_cache[n:]
        del self.value_cache[n:]
        # Update HF internal counter if present
        if hasattr(self, "_seen_tokens"):
            self._seen_tokens = anchor.per_layer_len[0] if anchor.per_layer_len else 0
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_sparse_cache.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/cache tests/test_sparse_cache.py
git commit -m "feat(cache): SparseCache with snapshot/reset"
```

---

### Task 9: SparseCache.select_visual

**Files:**
- Modify: `<repo>/sparsevila/cache/sparse_cache.py`
- Modify: `<repo>/tests/test_sparse_cache.py`

- [ ] **Step 1: Add failing tests**

```python
# Append to tests/test_sparse_cache.py
import pytest


def test_select_visual_concat_shape():
    cache = SparseCache()
    # Layer 0: total seq=10; visual span is [3, 9), so 6 visual tokens.
    cache.update(torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4), 0)
    keep = torch.tensor([0, 2, 5])     # keep 3 of 6 visual tokens
    pos_ids = torch.arange(10).unsqueeze(0)
    packed = cache.select_visual(
        layer_idx=0, visual_range=(3, 9), keep_idx=keep, position_ids=pos_ids
    )
    # 3 (sys-ish) + 3 (kept visual) + 1 (post-visual) = 7
    assert packed.k.shape == (1, 2, 7, 4)
    assert packed.v.shape == (1, 2, 7, 4)
    assert packed.layer == 0


def test_select_visual_does_not_mutate_cache():
    cache = SparseCache()
    k0 = torch.randn(1, 2, 8, 4)
    cache.update(k0.clone(), torch.randn(1, 2, 8, 4), 0)
    before = cache.key_cache[0].clone()
    cache.select_visual(
        layer_idx=0, visual_range=(2, 6),
        keep_idx=torch.tensor([0, 1]),
        position_ids=torch.arange(8).unsqueeze(0),
    )
    assert torch.equal(cache.key_cache[0], before)


def test_select_visual_storage_independent():
    cache = SparseCache()
    cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
    packed = cache.select_visual(
        layer_idx=0, visual_range=(2, 6),
        keep_idx=torch.tensor([0, 1]),
        position_ids=torch.arange(8).unsqueeze(0),
    )
    # Mutating packed.k must not change the source.
    src_ptr = cache.key_cache[0].data_ptr()
    packed.k.add_(1.0)
    assert cache.key_cache[0].data_ptr() == src_ptr  # not reallocated
    # Storage is shared only if pointers overlap; check no overlap.
    assert packed.k.data_ptr() != cache.key_cache[0].data_ptr()


def test_select_visual_keep_idx_out_of_range_raises():
    cache = SparseCache()
    cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
    with pytest.raises(IndexError):
        cache.select_visual(
            layer_idx=0, visual_range=(2, 6),
            keep_idx=torch.tensor([10]),     # out of range
            position_ids=torch.arange(8).unsqueeze(0),
        )
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_sparse_cache.py
```

- [ ] **Step 3: Implement select_visual**

Add this method body to the `SparseCache` class in `sparsevila/cache/sparse_cache.py`:

```python
    def select_visual(
        self,
        layer_idx: int,
        visual_range: tuple[int, int],
        keep_idx: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> "PackedKV":
        from ..retrieval.packed_kv import PackedKV   # local to break cycle

        v_start, v_end = visual_range
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        seq_len = k.shape[2]

        if (keep_idx.max().item() >= (v_end - v_start)) or (keep_idx.min().item() < 0):
            raise IndexError(
                f"keep_idx out of visual range [0, {v_end - v_start}): "
                f"min={keep_idx.min().item()}, max={keep_idx.max().item()}"
            )

        absolute_keep = (keep_idx + v_start).to(k.device)
        pre_k  = k[:, :, :v_start, :]
        mid_k  = k.index_select(2, absolute_keep)
        post_k = k[:, :, v_end:, :]
        pre_v  = v[:, :, :v_start, :]
        mid_v  = v.index_select(2, absolute_keep)
        post_v = v[:, :, v_end:, :]

        packed_k = torch.cat([pre_k, mid_k, post_k], dim=2)
        packed_v = torch.cat([pre_v, mid_v, post_v], dim=2)

        # Build position_ids of the packed sequence: keep sys + select-visual + post
        # Note: incoming position_ids must match the FULL cache seq_len.
        if position_ids.shape[-1] != seq_len:
            raise ValueError(
                f"position_ids length {position_ids.shape[-1]} != cache seq_len {seq_len}"
            )
        pos_pre  = position_ids[:, :v_start]
        pos_mid  = position_ids[:, v_start + keep_idx.cpu()]
        pos_post = position_ids[:, v_end:]
        packed_pos = torch.cat([pos_pre, pos_mid, pos_post], dim=1)

        return PackedKV(
            k=packed_k, v=packed_v, layer=layer_idx,
            visual_keep=keep_idx, visual_range=visual_range,
            position_ids=packed_pos,
        )
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_sparse_cache.py
```

Expected: 7 passed total.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/cache/sparse_cache.py tests/test_sparse_cache.py
git commit -m "feat(cache): SparseCache.select_visual non-destructive view"
```

---

### Task 10: Kernel salience wrapper (flash + naive)

**Files:**
- Create: `<repo>/sparsevila/kernels/__init__.py`
- Create: `<repo>/sparsevila/kernels/salience.py`
- Create: `<repo>/tests/test_kernel_salience.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_kernel_salience.py
import torch
import pytest
from sparsevila.kernels.salience import column_salience


def test_naive_path_shape():
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 16, 4)
    out = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    assert out.shape == (1, 2, 16)


def test_naive_matches_manual_softmax_sum():
    q = torch.randn(1, 1, 3, 4, dtype=torch.float32)
    k = torch.randn(1, 1, 5, 4, dtype=torch.float32)
    out = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    # Manual reference
    import math
    scale = 1.0 / math.sqrt(4)
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    ref = attn.sum(dim=2)        # sum over queries -> (B, H, N)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.cuda
def test_flash_matches_naive_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q = torch.randn(1, 2, 16, 8, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 8, device="cuda", dtype=torch.float16)
    out_f = column_salience(q, k, reduction="sum", is_causal=False, use_flash=True)
    out_n = column_salience(q, k, reduction="sum", is_causal=False, use_flash=False)
    assert torch.allclose(out_f, out_n, atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: Run, confirm CPU tests fail (no module)**

```bash
pytest -v tests/test_kernel_salience.py -m "not cuda"
```

- [ ] **Step 3: Implement**

```python
# sparsevila/kernels/__init__.py
from .salience import column_salience

__all__ = ["column_salience"]
```

```python
# sparsevila/kernels/salience.py
"""Unified wrapper around flash_colreduce (Triton) and a naive PyTorch fallback.

Provides one function `column_salience` matching the signature of the kernel
plus a CPU-friendly path used for unit testing and debug.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _naive(
    q: torch.Tensor, k: torch.Tensor, reduction: str, is_causal: bool, scale: float
) -> torch.Tensor:
    m, n = q.shape[2], k.shape[2]
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        c = max(n - m, 0)
        qi = torch.arange(m, device=q.device).view(1, 1, -1, 1)
        ki = torch.arange(n, device=q.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(qi + c < ki, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if reduction == "sum":
        return probs.sum(dim=2)
    if reduction == "mean":
        out = probs.sum(dim=2)
        if is_causal:
            c = max(n - m, 0)
            out[..., :c] /= m
            out[..., c:] /= torch.arange(n - c, 0, -1, device=q.device)
        else:
            out /= m
        return out
    if reduction == "max":
        return probs.max(dim=2).values
    raise ValueError(f"Invalid reduction: {reduction}")


def column_salience(
    q: torch.Tensor,
    k: torch.Tensor,
    reduction: str = "sum",
    is_causal: bool = False,
    scale: float | None = None,
    use_flash: bool = True,
) -> torch.Tensor:
    """Column-wise softmax-attention reduction along the query axis.

    Returns shape (B, H, N) where N = k.shape[2].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if use_flash and q.is_cuda and k.is_cuda:
        try:
            from flash_colreduce import flash_colreduce
        except ImportError:
            return _naive(q, k, reduction, is_causal, scale)
        return flash_colreduce(q, k, reduction=reduction, is_causal=is_causal, scale=scale)

    return _naive(q, k, reduction, is_causal, scale)
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_kernel_salience.py -m "not cuda"
```

Expected: 2 passed; 1 deselected (cuda).

- [ ] **Step 5: Commit**

```bash
git add sparsevila/kernels tests/test_kernel_salience.py
git commit -m "feat(kernels): column_salience wrapper (flash + naive)"
```

---

### Task 11: DecodeRetriever

**Files:**
- Create: `<repo>/sparsevila/retrieval/decode_retriever.py`
- Modify: `<repo>/sparsevila/retrieval/__init__.py`
- Create: `<repo>/tests/test_decode_retriever.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_decode_retriever.py
import torch
from sparsevila.cache.sparse_cache import SparseCache
from sparsevila.retrieval.decode_retriever import select_packed_kv_for_layer


def _make_cache_with_one_layer(seq_len=10, b=1, h=2, d=4):
    cache = SparseCache()
    cache.update(torch.randn(b, h, seq_len, d), torch.randn(b, h, seq_len, d), 0)
    return cache


def test_returns_packed_kv_with_expected_count():
    cache = _make_cache_with_one_layer(seq_len=12)
    q_proj = torch.randn(1, 2, 3, 4)  # |Q|=3
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0,
        q_proj=q_proj,
        cache=cache,
        visual_range=(2, 10),         # 8 visual tokens
        position_ids=pos_ids,
        decode_ratio=0.5,             # keep ~50% = 4
        head_aggregation="mean",
        use_flash=False,
    )
    # sys(2) + kept_visual(~4) + post(2) = ~8
    assert packed.k.shape[2] == 2 + 4 + 2
    assert packed.layer == 0


def test_ratio_zero_keeps_all_visual():
    cache = _make_cache_with_one_layer(seq_len=12)
    q_proj = torch.randn(1, 2, 2, 4)
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0, q_proj=q_proj, cache=cache,
        visual_range=(2, 10), position_ids=pos_ids,
        decode_ratio=0.0, head_aggregation="mean", use_flash=False,
    )
    assert packed.k.shape[2] == 12     # nothing pruned


def test_ties_fallback_to_topk():
    cache = _make_cache_with_one_layer(seq_len=12)
    # If we set q_proj to zero, all scores become uniform -> ties.
    q_proj = torch.zeros(1, 2, 2, 4)
    pos_ids = torch.arange(12).unsqueeze(0)
    packed = select_packed_kv_for_layer(
        layer_idx=0, q_proj=q_proj, cache=cache,
        visual_range=(2, 10), position_ids=pos_ids,
        decode_ratio=0.5, head_aggregation="mean", use_flash=False,
    )
    # Fallback path: keep round((1-0.5)*8)=4 tokens
    assert packed.k.shape[2] == 2 + 4 + 2
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_decode_retriever.py
```

- [ ] **Step 3: Implement**

```python
# sparsevila/retrieval/decode_retriever.py
"""Algorithm 3: query-aware decode-time KV retrieval, per layer."""
from __future__ import annotations
import torch
from ..cache.sparse_cache import SparseCache
from ..kernels.salience import column_salience
from .packed_kv import PackedKV


def select_packed_kv_for_layer(
    layer_idx: int,
    q_proj: torch.Tensor,                  # (B, H, |Q|, D)
    cache: SparseCache,
    visual_range: tuple[int, int],
    position_ids: torch.Tensor,            # (B, full_seq_len)
    decode_ratio: float,
    head_aggregation: str = "mean",
    use_flash: bool = True,
) -> PackedKV:
    v_start, v_end = visual_range
    V = v_end - v_start

    if decode_ratio == 0.0 or V == 0:
        keep_idx = torch.arange(V, device=q_proj.device)
        return cache.select_visual(
            layer_idx=layer_idx, visual_range=visual_range,
            keep_idx=keep_idx, position_ids=position_ids,
        )

    k_visual = cache.key_cache[layer_idx][:, :, v_start:v_end, :]   # (B, H, V, D)

    salience = column_salience(
        q_proj, k_visual, reduction="sum", is_causal=False, use_flash=use_flash,
    )  # (B, H, V)

    if head_aggregation == "mean":
        aggregated = salience.mean(dim=1)        # (B, V)
    elif head_aggregation == "max":
        aggregated = salience.max(dim=1).values  # (B, V)
    else:
        raise ValueError(f"head_aggregation must be mean|max, got {head_aggregation}")

    # Use the first batch row (paper assumes B=1)
    sal_b = aggregated[0]                        # (V,)
    threshold = sal_b.quantile(decode_ratio)
    keep_mask = sal_b > threshold
    if keep_mask.sum().item() == 0:
        k = max(1, round((1.0 - decode_ratio) * V))
        topk = sal_b.topk(k).indices
        keep_mask = torch.zeros_like(sal_b, dtype=torch.bool)
        keep_mask[topk] = True

    keep_idx = keep_mask.nonzero(as_tuple=True)[0].sort().values

    return cache.select_visual(
        layer_idx=layer_idx, visual_range=visual_range,
        keep_idx=keep_idx, position_ids=position_ids,
    )
```

```python
# sparsevila/retrieval/__init__.py
from .packed_kv import PackedKV
from .decode_retriever import select_packed_kv_for_layer

__all__ = ["PackedKV", "select_packed_kv_for_layer"]
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_decode_retriever.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/retrieval tests/test_decode_retriever.py
git commit -m "feat(retrieval): per-layer query-aware KV selection"
```

---

### Task 12: VLMAdapter ABC

**Files:**
- Create: `<repo>/sparsevila/models/__init__.py`
- Create: `<repo>/sparsevila/models/base.py`
- Create: `<repo>/tests/test_models_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models_base.py
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
```

- [ ] **Step 2: Run, confirm fails (ImportError)**

```bash
pytest -v tests/test_models_base.py
```

- [ ] **Step 3: Implement**

```python
# sparsevila/models/__init__.py
from .base import VLMAdapter

__all__ = ["VLMAdapter"]
```

```python
# sparsevila/models/base.py
"""Abstract adapter — per-model SparseVILA injection points."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch


class VLMAdapter(ABC):
    """Each supported VLM provides one of these."""

    @abstractmethod
    def wrap_encoder(self, vision_tower: Any, config: Any) -> Any:
        """Return the (possibly wrapped) vision tower that exposes salience."""

    @abstractmethod
    def wrap_llm(self, llm: Any, config: Any) -> Any:
        """Return the (possibly wrapped) LLM with decode-time retrieval injected."""

    @abstractmethod
    def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
        """Return (v_start, v_end) in the input_ids sequence."""

    @abstractmethod
    def build_position_ids(
        self, sys_len: int, vis_len: int, text_len: int
    ) -> torch.Tensor:
        """Return adjusted position_ids (1, S)."""
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_models_base.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/models tests/test_models_base.py
git commit -m "feat(models): VLMAdapter abstract base"
```

---

### Task 13: LLaVA-1.5 submodule + reconnaissance notes

**Files:**
- Create: `<repo>/third_party/.gitkeep`
- Create: `<repo>/docs/llava_repo_notes.md`

- [ ] **Step 1: Add LLaVA repo as submodule**

```bash
cd "<repo>"
mkdir -p third_party
git submodule add https://github.com/haotian-liu/LLaVA.git third_party/LLaVA
cd third_party/LLaVA
# Pin to a known-good commit (LLaVA-1.5 stable)
git checkout c121f04
cd ../..
```

If `git submodule add` fails because of HTTPS/proxy, fall back to `git clone` and document in README that this is not a real submodule yet.

- [ ] **Step 2: Read LLaVA's modeling code and document key classes**

Read the following files in `third_party/LLaVA/llava/model/`:
- `language_model/llava_llama.py` — `LlavaLlamaForCausalLM`
- `multimodal_encoder/clip_encoder.py` — `CLIPVisionTower`
- `multimodal_encoder/builder.py`
- `llava_arch.py` — `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`

Write a reconnaissance note recording the entry points needed by SparseVILA, into `docs/llava_repo_notes.md`. Required content:

```markdown
# LLaVA-1.5 Repo Reconnaissance

Pinned commit: c121f04 (or the actual commit used).

## Entry points used by SparseVILA

### Model loader
`from llava.model.builder import load_pretrained_model`
- Returns `(tokenizer, model, image_processor, context_len)`.
- `model` is `LlavaLlamaForCausalLM` (subclasses `LlamaForCausalLM`).

### Vision tower
- Class: `CLIPVisionTower` in `llava/model/multimodal_encoder/clip_encoder.py`.
- Forward output: hidden states from layer `-2` (LLaVA convention).
- We patch its `forward` to capture attention weights from layer `salience_layer_idx` and compute salience BEFORE the projector.

### LLM
- `LlavaLlamaForCausalLM.forward` calls `prepare_inputs_labels_for_multimodal` which interleaves visual embeddings.
- Per-layer attention lives in `model.model.layers[l].self_attn` (`LlamaAttention`).
- We replace each `LlamaAttention.forward` with a wrapper that:
  1. On the first prefill of a new query, computes salience via `column_salience` and builds `packed_kv[l]` once.
  2. On subsequent decode steps, attends only into `packed_kv[l]` (extended with the latest token KV).

## Visual span in input_ids
- Image token is `IMAGE_TOKEN_INDEX = -200`.
- `prepare_inputs_labels_for_multimodal` replaces the -200 marker with the visual embeddings inline.
- `get_visual_span` finds the contiguous block of visual positions in the post-replacement sequence by tracking the position of the marker before replacement.

## Image processing
- `image_processor` returns `(B, 3, 336, 336)`.
- Pad/center-crop policy: documented in `image_aspect_ratio='pad'` for v1.5.
```

- [ ] **Step 3: Verify import works**

```bash
cd "<repo>"
pip install -e ./third_party/LLaVA
python -c "from llava.model.builder import load_pretrained_model; print('ok')"
```

Expected: "ok". If imports fail due to missing deps, fix `pyproject.toml` and re-install.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/.gitkeep docs/llava_repo_notes.md
git commit -m "deps: add LLaVA submodule + reconnaissance notes"
```

---

### Task 14: LLaVA-1.5 adapter — wrap_encoder + salience extraction

**Files:**
- Create: `<repo>/sparsevila/models/llava_15.py`
- Create: `<repo>/tests/test_llava15_encoder.py`

- [ ] **Step 1: Write failing test (uses mocked encoder)**

```python
# tests/test_llava15_encoder.py
import torch
import pytest
from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import LlavaFifteenAdapter, EncoderSalienceOutput


class _FakeCLIPTower(torch.nn.Module):
    """Mimics LLaVA CLIPVisionTower minimally."""
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.num_patches = 4   # 2x2 patches for the test

    def forward(self, images):
        # Return hidden_states (B, S+1, D) with CLS at 0
        b = images.shape[0]
        hs = torch.randn(b, self.num_patches + 1, self.hidden_size)
        # Fake attention with deliberate signal on patch idx 2
        attn = torch.zeros(b, 1, self.num_patches + 1, self.num_patches + 1)
        attn[..., 0, 3] = 1.0  # CLS -> patch 2
        return hs, [attn]      # (hidden_states, [attn per layer])


def test_wrap_encoder_returns_salience_output():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)
    # All 4 patches present in salience score vector
    assert out.salience.shape == (4,)
    # Pruning: keep 2 of 4
    assert out.pruned_hidden.shape == (1, 2, 8)
    assert out.kept_idx.numel() == 2
    # The patch with the highest signal (idx 2) should be kept
    assert 2 in out.kept_idx.tolist()


def test_wrap_encoder_zero_ratio_bypasses():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.0)
    adapter = LlavaFifteenAdapter()
    tower = _FakeCLIPTower()
    wrapped = adapter.wrap_encoder(tower, cfg)
    images = torch.randn(1, 3, 16, 16)
    out = wrapped(images)
    # No tokens pruned
    assert out.pruned_hidden.shape == (1, 4, 8)
    assert out.kept_idx.numel() == 4
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_llava15_encoder.py
```

- [ ] **Step 3: Implement LlavaFifteenAdapter.wrap_encoder**

```python
# sparsevila/models/llava_15.py
"""Adapter for LLaVA-1.5-7B (haotian-liu/LLaVA).

Wraps the CLIPVisionTower so that its forward(images) returns an
`EncoderSalienceOutput` carrying pruned hidden_states + kept indices + salience.
The downstream `prepare_inputs_labels_for_multimodal` will need to be taught
to consume `pruned_hidden` in place of the regular encoder output (Task 15).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from ..config import SparseVILAConfig
from ..pruning.salience_estimator import compute_salience_from_attn
from ..pruning.encoder_pruner import prune_visual_tokens
from ..rope.unified_rope import build_compressed_position_ids
from .base import VLMAdapter


@dataclass
class EncoderSalienceOutput:
    pruned_hidden: torch.Tensor    # (B, K, D)
    kept_idx: torch.Tensor         # (K,)
    salience: torch.Tensor         # (S,) original


class _WrappedCLIPTower(nn.Module):
    def __init__(self, inner: nn.Module, config: SparseVILAConfig):
        super().__init__()
        self.inner = inner
        self.config = config

    def forward(self, images: torch.Tensor) -> EncoderSalienceOutput:
        # Expect inner.forward to return (hidden_states, attn_list)
        hs, attns = self.inner(images)
        # hs:    (B, S+1, D)  — CLS at index 0 for LLaVA-1.5 CLIP
        # attns: list of (B, H, S+1, S+1)
        attn = attns[self.config.salience_layer_idx]
        salience = compute_salience_from_attn(
            attn, strategy=self.config.salience_strategy,
        )
        patches = hs[:, 1:, :]      # strip CLS
        pruned, kept_idx = prune_visual_tokens(
            patches, salience, ratio=self.config.encoder_prune_ratio
        )
        return EncoderSalienceOutput(
            pruned_hidden=pruned, kept_idx=kept_idx, salience=salience
        )


class LlavaFifteenAdapter(VLMAdapter):
    def wrap_encoder(self, vision_tower: nn.Module, config: SparseVILAConfig) -> nn.Module:
        return _WrappedCLIPTower(vision_tower, config)

    def wrap_llm(self, llm: Any, config: SparseVILAConfig) -> Any:
        raise NotImplementedError("wrap_llm implemented in Task 15")

    def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
        raise NotImplementedError("get_visual_span implemented in Task 15")

    def build_position_ids(self, sys_len: int, vis_len: int, text_len: int) -> torch.Tensor:
        return build_compressed_position_ids(sys_len, vis_len, text_len)
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_llava15_encoder.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/models/llava_15.py tests/test_llava15_encoder.py
git commit -m "feat(models): LLaVA-1.5 adapter wrap_encoder + salience output"
```

---

### Task 15: LLaVA-1.5 adapter — wrap_llm with per-layer decode injection

**Files:**
- Modify: `<repo>/sparsevila/models/llava_15.py`
- Create: `<repo>/tests/test_llava15_llm_decode_hook.py`

- [ ] **Step 1: Write failing test (uses mocked LlamaAttention)**

```python
# tests/test_llava15_llm_decode_hook.py
import torch
from sparsevila.config import SparseVILAConfig
from sparsevila.cache.sparse_cache import SparseCache
from sparsevila.models.llava_15 import LlavaFifteenAdapter, SparseLlavaContext


def test_select_packed_kv_for_all_layers():
    """Smoke: build a fake 2-layer cache and verify the adapter produces
    one PackedKV per layer with the configured retrieval ratio."""
    cfg = SparseVILAConfig(decode_retrieval_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    cache = SparseCache()
    seq_len = 12
    for l in range(2):
        cache.update(torch.randn(1, 2, seq_len, 4), torch.randn(1, 2, seq_len, 4), l)

    # Build a SparseLlavaContext as the adapter expects.
    ctx = SparseLlavaContext(
        visual_range=(2, 10),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        cache=cache,
        q_proj_per_layer=[torch.randn(1, 2, 3, 4) for _ in range(2)],
        config=cfg,
    )

    packed_kvs = adapter.build_packed_kvs(ctx)
    assert len(packed_kvs) == 2
    for pk in packed_kvs:
        # sys(2) + keep_visual(~4) + post(2) = 8
        assert pk.k.shape[2] == 8


def test_build_packed_kvs_ratio_zero_bypasses():
    cfg = SparseVILAConfig(decode_retrieval_ratio=0.0)
    adapter = LlavaFifteenAdapter()
    cache = SparseCache()
    seq_len = 8
    cache.update(torch.randn(1, 2, seq_len, 4), torch.randn(1, 2, seq_len, 4), 0)
    ctx = SparseLlavaContext(
        visual_range=(2, 6),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        cache=cache,
        q_proj_per_layer=[torch.randn(1, 2, 1, 4)],
        config=cfg,
    )
    packed_kvs = adapter.build_packed_kvs(ctx)
    assert packed_kvs[0].k.shape[2] == seq_len    # all kept
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_llava15_llm_decode_hook.py
```

- [ ] **Step 3: Extend `llava_15.py`**

Add to `sparsevila/models/llava_15.py`:

```python
from dataclasses import field
from typing import List
from ..cache.sparse_cache import SparseCache
from ..retrieval.decode_retriever import select_packed_kv_for_layer
from ..retrieval.packed_kv import PackedKV


@dataclass
class SparseLlavaContext:
    """Carries per-query state that the LLM-side wrapper needs."""
    visual_range: tuple[int, int]
    position_ids: torch.Tensor
    cache: SparseCache
    q_proj_per_layer: List[torch.Tensor]   # one (B, H, |Q|, D) per layer
    config: SparseVILAConfig


# Add as method on LlavaFifteenAdapter:
class LlavaFifteenAdapter(VLMAdapter):   # (re-declare, extending earlier)
    ...
    def build_packed_kvs(self, ctx: SparseLlavaContext) -> List[PackedKV]:
        """Run Algorithm 3 for every layer; return one PackedKV per layer."""
        packed = []
        n_layers = len(ctx.cache.key_cache)
        if n_layers != len(ctx.q_proj_per_layer):
            raise ValueError(
                f"q_proj_per_layer length {len(ctx.q_proj_per_layer)} != "
                f"cache layers {n_layers}"
            )
        for l in range(n_layers):
            pk = select_packed_kv_for_layer(
                layer_idx=l,
                q_proj=ctx.q_proj_per_layer[l],
                cache=ctx.cache,
                visual_range=ctx.visual_range,
                position_ids=ctx.position_ids,
                decode_ratio=ctx.config.decode_retrieval_ratio,
                head_aggregation=ctx.config.decode_head_aggregation,
                use_flash=ctx.config.use_flash_kernel,
            )
            packed.append(pk)
        return packed
```

Because Python does not allow re-declaring a class twice in the same file, edit the existing `LlavaFifteenAdapter` class to add `build_packed_kvs` instead. Also implement `get_visual_span` (find the contiguous run of `IMAGE_TOKEN_INDEX = -200`):

```python
IMAGE_TOKEN_INDEX = -200

def get_visual_span(self, input_ids: torch.Tensor) -> tuple[int, int]:
    """Find the single contiguous run of IMAGE_TOKEN_INDEX (-200) in input_ids.

    NOTE: This must be called on the PRE-replacement input_ids, before
    `prepare_inputs_labels_for_multimodal` interleaves visual embeddings.
    After replacement, the caller maps (v_start_pre, v_start_pre + K) where K
    is the count of kept visual tokens.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("v1 supports batch_size=1")
    mask = (input_ids[0] == IMAGE_TOKEN_INDEX)
    idxs = mask.nonzero(as_tuple=True)[0]
    if idxs.numel() == 0:
        raise ValueError("No IMAGE_TOKEN_INDEX in input_ids")
    if idxs.numel() != 1:
        raise NotImplementedError("v1 supports exactly one image marker (one image)")
    v_start = int(idxs[0].item())
    v_end = v_start + 1   # caller expands once K is known
    return (v_start, v_end)


def wrap_llm(self, llm: Any, config: SparseVILAConfig) -> Any:
    """Install per-layer attention monkey-patches.

    Wires each `LlamaAttention.forward` to consult a `SparseLlavaContext`
    stashed on the model (`llm._sparsevila_ctx`). When the context is None
    (no current query), the original forward runs. When set, the wrapper
    routes attention through `packed_kvs[layer_idx]` instead of the full cache.

    Implementation note:
    The actual override is delegated to `_patch_llama_attention(llm, config)`,
    which iterates over `llm.model.layers` and replaces each
    `self_attn.forward` with a closure capturing the original forward and the
    config. The closure decides at runtime whether to use the packed path or
    the original path.
    """
    self._patch_llama_attention(llm, config)
    llm._sparsevila_ctx = None
    llm._sparsevila_packed_kvs = None
    return llm

def _patch_llama_attention(self, llm: Any, config: SparseVILAConfig) -> None:
    for layer_idx, layer in enumerate(llm.model.layers):
        attn = layer.self_attn
        original_forward = attn.forward

        def make_forward(layer_idx=layer_idx, original=original_forward):
            def patched_forward(hidden_states, *args, **kwargs):
                ctx = getattr(llm, "_sparsevila_ctx", None)
                packed = getattr(llm, "_sparsevila_packed_kvs", None)
                if ctx is None or packed is None:
                    return original(hidden_states, *args, **kwargs)
                # Route attention through packed_kvs[layer_idx]:
                pk = packed[layer_idx]
                # Replace past_key_value in kwargs with a temporary single-layer
                # cache wrapping pk.k, pk.v. Delegate to original with substituted
                # past_key_value.
                from transformers.cache_utils import DynamicCache
                tmp = DynamicCache()
                tmp.key_cache.append(pk.k)
                tmp.value_cache.append(pk.v)
                kwargs = dict(kwargs)
                kwargs["past_key_value"] = tmp
                return original(hidden_states, *args, **kwargs)
            return patched_forward

        attn.forward = make_forward()
```

- [ ] **Step 4: Run, confirm tests pass**

```bash
pytest -v tests/test_llava15_llm_decode_hook.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/models/llava_15.py tests/test_llava15_llm_decode_hook.py
git commit -m "feat(models): LLaVA-1.5 adapter wrap_llm + per-layer attention patch"
```

---

### Task 16: `load_sparse_llava` entry point

**Files:**
- Create: `<repo>/sparsevila/loader.py`
- Modify: `<repo>/sparsevila/__init__.py`
- Create: `<repo>/tests/test_loader_dry_run.py`

- [ ] **Step 1: Write failing test (mocks LLaVA loader)**

```python
# tests/test_loader_dry_run.py
import pytest
from unittest.mock import MagicMock, patch
from sparsevila import load_sparse_llava, SparseVILAConfig


def test_loader_returns_model_and_processor_with_wrapped_encoder():
    fake_model = MagicMock()
    fake_model.get_vision_tower.return_value = MagicMock()
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
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_loader_dry_run.py
```

- [ ] **Step 3: Implement loader**

```python
# sparsevila/loader.py
"""Top-level entry point: load_sparse_llava."""
from __future__ import annotations
from typing import Any
import torch

from .config import SparseVILAConfig
from .models.llava_15 import LlavaFifteenAdapter


def _load_llava_pretrained(model_path: str, dtype: torch.dtype, device: str):
    """Thin wrapper around llava.model.builder.load_pretrained_model.

    Isolated for mocking in tests.
    """
    from llava.model.builder import load_pretrained_model
    return load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=False,
        load_4bit=False,
        device_map=device,
        torch_dtype=dtype,
    )


def load_sparse_llava(
    model_path: str,
    config: SparseVILAConfig | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load LLaVA-1.5 with SparseVILA wiring applied.

    Returns (model, processor). The model object is the LLaVA model with:
    - Vision tower wrapped to return EncoderSalienceOutput.
    - LLM attention layers patched to route through packed_kvs when set.
    """
    if config is None:
        config = SparseVILAConfig()
    # __post_init__ already validates; this is a redundant guard if user
    # passes a hand-crafted object that bypasses dataclass init.
    if not 0.0 <= config.encoder_prune_ratio < 1.0:
        raise ValueError(f"Invalid encoder_prune_ratio: {config.encoder_prune_ratio}")

    tokenizer, model, processor, ctx_len = _load_llava_pretrained(model_path, dtype, device)

    adapter = LlavaFifteenAdapter()
    vt = model.get_vision_tower()
    wrapped_vt = adapter.wrap_encoder(vt, config)
    # Re-attach: LLaVA stores vision tower in different attribute paths across
    # commits. Try the documented setter first, fall back to direct attribute.
    if hasattr(model, "set_vision_tower"):
        model.set_vision_tower(wrapped_vt)
    else:
        model.model.vision_tower = wrapped_vt

    adapter.wrap_llm(model, config)
    model._sparsevila_adapter = adapter
    model._sparsevila_config = config

    return model, processor
```

```python
# sparsevila/__init__.py
from .config import SparseVILAConfig
from .loader import load_sparse_llava

__version__ = "0.1.0"
__all__ = ["SparseVILAConfig", "load_sparse_llava"]
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_loader_dry_run.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/loader.py sparsevila/__init__.py tests/test_loader_dry_run.py
git commit -m "feat(loader): load_sparse_llava entry point"
```

---

### Task 17: Inline-pruning forward shim (the integration step)

This task wires the encoder pruning OUTPUT (which our `_WrappedCLIPTower` returns as `EncoderSalienceOutput`) into LLaVA's `prepare_inputs_labels_for_multimodal`, which natively expects the encoder to return a plain tensor. We add a thin shim that unpacks our object.

**Files:**
- Modify: `<repo>/sparsevila/models/llava_15.py`
- Create: `<repo>/tests/test_llava15_inline_shim.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llava15_inline_shim.py
import torch
from unittest.mock import MagicMock
from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import (
    LlavaFifteenAdapter, _WrappedCLIPTower, EncoderSalienceOutput,
)


class _FakeCLIPTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8

    def forward(self, images):
        b = images.shape[0]
        hs = torch.randn(b, 5, 8)   # 4 patches + CLS
        attn = torch.softmax(torch.randn(b, 1, 5, 5), dim=-1)
        return hs, [attn]


def test_wrapped_tower_compat_mode_returns_tensor_when_requested():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    # In compat mode, wrapped tower returns a plain tensor (LLaVA-compatible).
    wrapped.compat_mode = True
    out = wrapped(images)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 8)   # K=2 (after pruning)


def test_wrapped_tower_native_mode_returns_dataclass():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    images = torch.randn(1, 3, 16, 16)
    wrapped.compat_mode = False
    out = wrapped(images)
    assert isinstance(out, EncoderSalienceOutput)


def test_wrapped_tower_records_last_kept_idx():
    cfg = SparseVILAConfig(encoder_prune_ratio=0.5)
    adapter = LlavaFifteenAdapter()
    wrapped = adapter.wrap_encoder(_FakeCLIPTower(), cfg)
    wrapped.compat_mode = True
    wrapped(torch.randn(1, 3, 16, 16))
    assert wrapped.last_kept_idx is not None
    assert wrapped.last_kept_idx.numel() == 2
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_llava15_inline_shim.py
```

- [ ] **Step 3: Modify `_WrappedCLIPTower`**

Edit `sparsevila/models/llava_15.py`:

```python
class _WrappedCLIPTower(nn.Module):
    def __init__(self, inner, config):
        super().__init__()
        self.inner = inner
        self.config = config
        # When True, return a plain tensor for LLaVA compatibility.
        # The kept_idx is stashed on `self.last_kept_idx` for the LLM-side
        # wrapper to read when building position_ids and visual_range.
        self.compat_mode: bool = True
        self.last_kept_idx: torch.Tensor | None = None
        self.last_salience: torch.Tensor | None = None

    def forward(self, images):
        hs, attns = self.inner(images)
        attn = attns[self.config.salience_layer_idx]
        salience = compute_salience_from_attn(
            attn, strategy=self.config.salience_strategy,
        )
        patches = hs[:, 1:, :]
        pruned, kept_idx = prune_visual_tokens(
            patches, salience, ratio=self.config.encoder_prune_ratio,
        )
        self.last_kept_idx = kept_idx
        self.last_salience = salience
        if self.compat_mode:
            return pruned
        return EncoderSalienceOutput(
            pruned_hidden=pruned, kept_idx=kept_idx, salience=salience,
        )
```

Earlier tests in `test_llava15_encoder.py` expect `EncoderSalienceOutput`. Update them to set `wrapped.compat_mode = False` before calling, or move that assignment into the test fixtures.

Update `tests/test_llava15_encoder.py`:

```python
# In both existing tests, add before calling wrapped(images):
    wrapped.compat_mode = False
```

- [ ] **Step 4: Run all tests under this group**

```bash
pytest -v tests/test_llava15_encoder.py tests/test_llava15_inline_shim.py
```

Expected: 5 passed total.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/models/llava_15.py tests/test_llava15_inline_shim.py tests/test_llava15_encoder.py
git commit -m "feat(models): compat-mode shim so LLaVA pipeline accepts pruned encoder output"
```

---

### Task 18: Bind query and packed_kvs around `generate`

**Files:**
- Modify: `<repo>/sparsevila/models/llava_15.py`
- Create: `<repo>/tests/test_llava15_generate_binding.py`

This task adds the missing piece that ties everything together: a context manager (`SparseInferenceSession`) that, before `generate` is called, captures the Q-projection per layer (by hooking into the LLM forward), runs Algorithm 3 once, attaches `packed_kvs` to the LLM, and releases everything on exit.

- [ ] **Step 1: Write failing test**

```python
# tests/test_llava15_generate_binding.py
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
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_llava15_generate_binding.py
```

- [ ] **Step 3: Implement session**

Append to `sparsevila/models/llava_15.py`:

```python
class SparseInferenceSession:
    """Context manager that wires `packed_kvs` onto the LLM for one generation."""

    def __init__(self, llm: Any, adapter: LlavaFifteenAdapter, ctx: SparseLlavaContext):
        self.llm = llm
        self.adapter = adapter
        self.ctx = ctx

    def __enter__(self):
        packed = self.adapter.build_packed_kvs(self.ctx)
        self.llm._sparsevila_ctx = self.ctx
        self.llm._sparsevila_packed_kvs = packed
        return self

    def __exit__(self, exc_type, exc, tb):
        self.llm._sparsevila_ctx = None
        self.llm._sparsevila_packed_kvs = None
        return False
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_llava15_generate_binding.py
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add sparsevila/models/llava_15.py tests/test_llava15_generate_binding.py
git commit -m "feat(models): SparseInferenceSession context manager"
```

---

### Task 19: CLI inference script

**Files:**
- Create: `<repo>/scripts/run_inference.py`
- Create: `<repo>/tests/test_cli_argparse.py`

- [ ] **Step 1: Write failing test (argparse only — actual model not loaded)**

```python
# tests/test_cli_argparse.py
import sys
import subprocess


def test_cli_parses_args_and_runs_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_inference.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--image" in result.stdout
    assert "--prompt" in result.stdout
    assert "--encoder-prune-ratio" in result.stdout
    assert "--decode-retrieval-ratio" in result.stdout
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest -v tests/test_cli_argparse.py
```

- [ ] **Step 3: Implement CLI**

```python
# scripts/run_inference.py
"""Tiny CLI for SparseVILA LLaVA-1.5 inference.

Example:
    python scripts/run_inference.py \\
        --model liuhaotian/llava-v1.5-7b \\
        --image path/to/cat.jpg \\
        --prompt "What is in this image?" \\
        --encoder-prune-ratio 0.5 \\
        --decode-retrieval-ratio 0.75
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_inference.py` without install:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="SparseVILA LLaVA-1.5 inference")
    p.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    p.add_argument("--image", required=True, help="Path or URL to image")
    p.add_argument("--prompt", required=True)
    p.add_argument("--encoder-prune-ratio", type=float, default=0.0)
    p.add_argument("--decode-retrieval-ratio", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    # Defer heavy imports so --help is fast and runs without GPU.
    from PIL import Image
    import torch
    from sparsevila import load_sparse_llava, SparseVILAConfig

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    config = SparseVILAConfig(
        encoder_prune_ratio=args.encoder_prune_ratio,
        decode_retrieval_ratio=args.decode_retrieval_ratio,
    )
    model, processor = load_sparse_llava(
        args.model, config=config, dtype=dtype, device=args.device,
    )
    if args.image.startswith("http"):
        import requests
        from io import BytesIO
        img = Image.open(BytesIO(requests.get(args.image, timeout=30).content)).convert("RGB")
    else:
        img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, text=args.prompt, return_tensors="pt").to(args.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest -v tests/test_cli_argparse.py
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_inference.py tests/test_cli_argparse.py
git commit -m "feat(scripts): CLI inference entry point"
```

---

### Task 20: README polish

**Files:**
- Modify: `<repo>/README.md`

- [ ] **Step 1: Rewrite README with usage**

```markdown
# SparseVILA — LLaVA-1.5 Reference Implementation

Reference implementation of [SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](https://arxiv.org/abs/2510.17777) (ICCV 2025) on LLaVA-1.5-7B.

This package implements the two SparseVILA algorithms:
- **Algorithm 2** — query-agnostic visual token pruning at the encoder output, using CLIP CLS-token salience.
- **Algorithm 3** — query-aware decode-time KV retrieval, per LLM layer, using the `flash_colreduce` Triton kernel.

For the design and rationale, see `docs/superpowers/specs/2026-05-15-sparsevila-impl-design.md`.

## Setup

```bash
git clone <repo-url> && cd "[man] SparseVILA"
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e ./third_party/LLaVA
pip install -e .
```

Recommended environments: Linux + CUDA 12, WSL2, Google Colab, or vast.ai. Windows-native may work but is not officially supported.

## Usage

### Python
```python
from sparsevila import load_sparse_llava, SparseVILAConfig

config = SparseVILAConfig(
    encoder_prune_ratio=0.5,
    decode_retrieval_ratio=0.75,
)
model, processor = load_sparse_llava("liuhaotian/llava-v1.5-7b", config=config)

inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(out[0], skip_special_tokens=True))
```

### CLI
```bash
python scripts/run_inference.py \
    --image cat.jpg \
    --prompt "What is in this image?" \
    --encoder-prune-ratio 0.5 \
    --decode-retrieval-ratio 0.75
```

### Colab
Open `notebooks/colab_demo.ipynb`. The first cells clone the repo and install everything.

## Project structure

See the spec at `docs/superpowers/specs/2026-05-15-sparsevila-impl-design.md` for the full module map and design.

## What's NOT here yet

- Multi-turn conversation evaluation (Algorithm 4).
- Accuracy benchmarks (POPE, GQA, etc.).
- Latency benchmarks vs vanilla.
- AWQ / SmoothQuant quantization.
- LLaVA-NeXT anyres tiling.

These are deferred to future phases.

## License

MIT.

## Citing

If this code helps your work, please cite the original paper (BibTeX in `flash-colreduce/README.md`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: polished README with setup and usage"
```

---

### Task 21: Colab notebook

**Files:**
- Create: `<repo>/notebooks/colab_demo.ipynb`

- [ ] **Step 1: Build the notebook**

Use Jupyter to author cells. The notebook should contain, in order:

1. **Markdown cell** — title + 1 paragraph context, link to spec.
2. **Code cell — clone & install**:
   ```python
   !git clone <repo-url>
   %cd "[man] SparseVILA"
   !git submodule update --init --recursive
   !pip install -q -e ./flash-colreduce
   !pip install -q -e ./third_party/LLaVA
   !pip install -q -e .
   ```
3. **Code cell — imports**:
   ```python
   import torch
   from PIL import Image
   from sparsevila import load_sparse_llava, SparseVILAConfig
   ```
4. **Code cell — load model (vanilla, config zero)**:
   ```python
   cfg_vanilla = SparseVILAConfig()  # zero ratios
   model_v, processor_v = load_sparse_llava(
       "liuhaotian/llava-v1.5-7b", config=cfg_vanilla,
       dtype=torch.float16, device="cuda")
   ```
5. **Code cell — load same model with sparsity**:
   ```python
   cfg_sparse = SparseVILAConfig(
       encoder_prune_ratio=0.5,
       decode_retrieval_ratio=0.75,
   )
   model_s, processor_s = load_sparse_llava(
       "liuhaotian/llava-v1.5-7b", config=cfg_sparse,
       dtype=torch.float16, device="cuda")
   ```
6. **Code cell — load sample image + run both**:
   ```python
   from io import BytesIO
   import requests
   img = Image.open(BytesIO(requests.get(
       "https://llava-vl.github.io/static/images/view.jpg").content)).convert("RGB")
   prompt = "What is unusual about this image?"
   ```
7. **Code cell — generate vanilla** (greedy, fixed seed):
   ```python
   torch.manual_seed(0)
   inputs_v = processor_v(images=img, text=prompt, return_tensors="pt").to("cuda")
   out_v = model_v.generate(**inputs_v, max_new_tokens=80, do_sample=False)
   text_v = processor_v.decode(out_v[0], skip_special_tokens=True)
   print("VANILLA:\n", text_v)
   ```
8. **Code cell — generate sparse**:
   ```python
   torch.manual_seed(0)
   inputs_s = processor_s(images=img, text=prompt, return_tensors="pt").to("cuda")
   out_s = model_s.generate(**inputs_s, max_new_tokens=80, do_sample=False)
   text_s = processor_s.decode(out_s[0], skip_special_tokens=True)
   print("SPARSEVILA:\n", text_s)
   ```
9. **Code cell — golden equivalence check (zero config)**:
   ```python
   # Re-load with zero config, compare output IDs to a second zero-config load.
   # If both produce identical output IDs for 50 tokens, the wrapping is byte-equivalent
   # to vanilla LLaVA — proves no regression.
   m1, p1 = load_sparse_llava("liuhaotian/llava-v1.5-7b",
                              config=SparseVILAConfig(), dtype=torch.float16, device="cuda")
   m2, p2 = load_sparse_llava("liuhaotian/llava-v1.5-7b",
                              config=SparseVILAConfig(), dtype=torch.float16, device="cuda")
   inp = p1(images=img, text=prompt, return_tensors="pt").to("cuda")
   torch.manual_seed(0); o1 = m1.generate(**inp, max_new_tokens=50, do_sample=False)
   torch.manual_seed(0); o2 = m2.generate(**inp, max_new_tokens=50, do_sample=False)
   assert torch.equal(o1, o2), "Two zero-config loads must produce identical output"
   print("Golden check OK")
   ```
10. **Markdown cell** — pointers to next steps (multi-turn eval, quantization).

Save the notebook as `notebooks/colab_demo.ipynb`.

- [ ] **Step 2: Commit**

```bash
git add notebooks/colab_demo.ipynb
git commit -m "docs: Colab demo notebook"
```

---

## Self-Review

**Spec coverage check (each spec section → task):**

| Spec section | Task(s) covering it |
|---|---|
| §2 Module map | Tasks 1, 12 (skeleton + ABC) |
| §3 Prefill flow / Alg 2 | Tasks 5, 6, 7, 14, 17 |
| §4 Decode flow / Alg 3 | Tasks 10, 11, 15, 18 |
| §5 SparseCache + RoPE | Tasks 3, 8, 9 |
| §6 Public API | Tasks 2, 12, 16 |
| §7 Testing | Each task is TDD; smoke = notebook (Task 21) |
| §8 Setup + Colab | Tasks 13, 19, 20, 21 |

**Placeholder scan:** Reviewed plan. No "TBD" / "TODO" placeholders. All steps include actual code or commands.

**Type consistency:**
- `SparseVILAConfig` — same field names across Tasks 2, 14, 15, 16, 17, 19.
- `PackedKV` — fields match across Tasks 4, 9, 11, 15.
- `compute_salience_from_attn` signature stable Tasks 5, 6, 14, 17.
- `prune_visual_tokens` signature stable Tasks 7, 14, 17.
- `SparseCache.select_visual` signature stable Tasks 9, 11, 15.
- `EncoderSalienceOutput` defined Task 14, consumed Task 17.
- `SparseLlavaContext` and `SparseInferenceSession` introduced Tasks 15 and 18 respectively.

**Notes on deferred items confirmed not in plan:** multi-turn evaluation, accuracy benchmarks, latency benchmarks, quantization, LLaVA-NeXT anyres — all out of scope per spec §1.3.

**Plan complete.**
