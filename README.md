# SparseVILA — LLaVA-1.5 Reference Implementation

Reference implementation of [SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](https://arxiv.org/abs/2510.17777) (ICCV 2025) on LLaVA-1.5-7B.

## What's implemented

| Paper component | Status | Notes |
|---|---|---|
| **Algorithm 2** — encoder query-agnostic pruning | ✓ | CLIP CLS-token salience over last layer's attention, quantile threshold. Available via `compute_salience_from_attn` (dense path) or `flash_colreduce` (kernel path). |
| **Algorithm 3** — decode query-aware retrieval | ✓ | Per-layer salience from text-Q → visual-K, top-K per layer. Two delivery variants (see below). |
| **flash-colreduce Triton kernel** | ✓ wired | Both encoder ("cls" strategy, M=1 case) and decode (M=`|Q_text|`, N=full_seq_len, `is_causal=True` to match Llama prefill). Falls back to naive PyTorch on CPU. |
| **Compressed RoPE** (paper §3.2) | ✓ | `rerotate_keys` re-applies RoPE on surviving K to bring them onto contiguous compressed positions; used by the cache-packing variant. |
| **W4A16 quantization (LLM)** | ~ | `quantize_llm="4bit-bnb"` loads via bitsandbytes NF4 + FP16 compute (paper-equivalent W4A16 memory profile, different scheme — NF4 vs AWQ INT4 with activation-aware scales). Verified ~60% VRAM reduction (14 GB → 5.5 GB). True AWQ via `autoawq` blocked by env compat (see below). |
| **W8A8 vision encoder (SmoothQuant)** | ✗ | Requires calibration + custom INT8 CLIP — deferred. |
| **Multi-turn protocol** (Algorithm 4) | ✗ | `SparseCache.snapshot()`/`reset_to()` API exists but not wired into a multi-turn `sparse_generate`. Deferred. |
| **Long-context / video benchmarks** | ✗ | Single-image inference only. Paper's main speedup claim is at 200–500 frames; not exercised. |

## Two decode-retrieval variants

```python
from sparsevila import sparse_generate, sparse_generate_packed
```

* **`sparse_generate`** — attention-mask strategy. Cache stays full; dropped visual positions get a `-inf` additive bias so softmax zeroes them. Same output math; **no real speedup** but no RoPE recomputation.
* **`sparse_generate_packed`** — cache-packing strategy (paper §4.1 latency claim). Each layer's KV cache is physically sliced to the kept visual subset, surviving K tensors are RoPE-recompressed to contiguous positions via `rerotate_keys`, decode runs against the shorter cache. Per-layer attention is routed through `PerLayerProxyCache` so updates write back into each layer's `PackedKV`.

Both verified to produce identical (or numerically equivalent) output on the LLaVA-1.5 pier demo image at decode_retrieval_ratio ∈ {0.0, 0.5, 0.9}.

## Setup

```bash
git clone https://github.com/AnKun10/sparsevila-implementation
cd sparsevila-implementation
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e .
```

The LLaVA submodule (`third_party/LLaVA`) is on the Python path; no separate install needed.

**Pinned versions** (per `pyproject.toml`): `transformers==4.37.2` + `tokenizers==0.15.1` to match LLaVA-repo's expectations. Newer transformers (4.40+) trip on LLaVA's pre-4.38 `forward` signature via `cache_position`.

## Usage

```python
import torch
from PIL import Image
from transformers import AutoTokenizer
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from sparsevila import load_sparse_llava, SparseVILAConfig, sparse_generate_packed

cfg = SparseVILAConfig(
    encoder_prune_ratio=0.5,        # Algorithm 2 ratio
    decode_retrieval_ratio=0.5,     # Algorithm 3 ratio
    use_flash_kernel=True,          # use flash-colreduce; False for naive PyTorch
    quantize_llm="none",            # or "4bit-bnb" / "8bit-bnb"
)
model, image_processor = load_sparse_llava(
    "liuhaotian/llava-v1.5-7b", config=cfg, dtype=torch.float16, device="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)

# Build a LLaVA-1.5 prompt and tokenize with IMAGE_TOKEN_INDEX in place.
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nWhat is in this image?")
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
).unsqueeze(0).cuda()

image = Image.open("path/to/image.jpg").convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).to("cuda", dtype=torch.float16)
attention_mask = torch.ones_like(input_ids)

out = sparse_generate_packed(
    model, tokenizer, input_ids, image_tensor, attention_mask,
    decode_retrieval_ratio=0.5, max_new_tokens=80,
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

For the simpler `sparse_generate` (attention-mask, no RoPE recomputation), swap the call. Both have identical signatures.

## Colab

Open `notebooks/colab_demo.ipynb`. The first cells clone the repo and install dependencies. Verified on Colab L4.

## Tests

```bash
pytest                    # full suite (~70 unit tests, CPU only)
pytest -m slow            # end-to-end smoke test (requires CUDA + LLaVA weights)
```

Coverage includes:
- Salience estimator (`cls`/`summary`/`mean_intra` strategies)
- Encoder pruner (quantile + top-K fallback, FP16-safe)
- RoPE re-rotation (matches fresh rotation; identity at delta=0; +d/-d round-trip)
- Per-layer proxy cache (decoupled layer index, append semantics)
- Flash-vs-attn-map salience equivalence (both encoder and decode)
- SparseCache snapshot/reset/select_visual non-destructive view

## On AWQ proper

The paper specifies INT4 AWQ + activation-aware scales for the LLM (TinyChat pipeline). We attempted to load `ybelkada/llava-1.5-7b-hf-awq` (publicly available, AWQ-packed):

* The checkpoint loads correctly as `transformers.LlavaForConditionalGeneration` with `WQLinear_GEMM` layers carrying INT4 `qweight` + per-group FP16 `scales` + `qzeros` — the structure matches the paper's spec exactly. VRAM for LLM weights is ~3.5 GB.
* **Inference is blocked by environment compat.** `autoawq` 0.2.9 is officially deprecated (Aug 2024); its Triton kernel does not compile against Triton 3.6 (Colab's current), and the prebuilt `awq_ext` / `awq_v2_ext` CUDA wheels have undefined symbols against PyTorch 2.10's C++ ABI. Successor is vLLM's [llm-compressor](https://github.com/vllm-project/llm-compressor) + `compressed-tensors` — not exercised in this implementation.

So our W4A16 demo uses bitsandbytes NF4 (`quantize_llm="4bit-bnb"`), which gives the same memory profile and dequant-at-matmul pattern as AWQ, with a different quantization scheme. Functionally a fair substitute; for paper-fidelity quantization-quality comparison, the AWQ path needs a working runtime.

## Layout

```
sparsevila/
├── config.py                       # SparseVILAConfig
├── loader.py                       # load_sparse_llava
├── cache/
│   ├── sparse_cache.py             # DynamicCache + snapshot/reset/select_visual
│   └── proxy_cache.py              # PerLayerProxyCache for attention patch
├── kernels/
│   └── salience.py                 # column_salience: flash_colreduce + naive fallback
├── pruning/
│   ├── salience_estimator.py       # CLS / summary / mean_intra strategies
│   └── encoder_pruner.py           # Algorithm 2 prune
├── retrieval/
│   ├── packed_kv.py                # PackedKV dataclass
│   └── decode_retriever.py         # select_packed_kv_for_layer (legacy path)
├── rope/
│   └── unified_rope.py             # compressed position_ids + rerotate_keys
├── inference/
│   ├── decode_salience.py          # per_layer_salience_{via_flash,from_attn_maps}
│   └── sparse_generate.py          # sparse_generate + sparse_generate_packed
└── models/
    ├── base.py                     # VLMAdapter ABC
    └── llava_15.py                 # LlavaFifteenAdapter

flash-colreduce/                    # Triton kernel (submodule of z-lab/flash-colreduce)
third_party/LLaVA/                  # haotian-liu/LLaVA submodule
tests/                              # ~70 unit tests
```

## License

MIT.

## Citing

If this code helps your work, please cite the original paper.
