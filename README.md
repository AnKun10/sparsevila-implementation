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
