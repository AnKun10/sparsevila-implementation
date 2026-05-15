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
pip install -e ./third_party/LLaVA --no-deps
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
