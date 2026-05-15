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
