#!/usr/bin/env python3

"""Prepare an RWKV-7 F16 safetensors model for the HarmonyOS NNRT backend.

The first embedding layer norm is constant for every vocabulary row. Computing it
offline removes one precision-sensitive operation from the Kirin NPU graph while
leaving all recurrent and token-dependent operations on the NPU.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


MARKER = "rwkv.nnrt.emb_ln0_precomputed"
CONVERSION_THREADS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute embedding ln0 for the RWKV HarmonyOS NNRT backend."
    )
    parser.add_argument("input", type=Path, help="Source RWKV-7 F16 safetensors file")
    parser.add_argument("output", type=Path, help="Destination NNRT safetensors file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    torch.set_num_threads(CONVERSION_THREADS)
    input_sha256 = sha256_file(args.input)
    tensors = load_file(str(args.input), device="cpu")
    required = ("emb.weight", "blocks.0.ln0.weight", "blocks.0.ln0.bias")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"missing required tensors: {', '.join(missing)}")
    if MARKER in tensors:
        raise ValueError("input is already prepared for NNRT")
    non_f16 = [name for name, tensor in tensors.items() if tensor.dtype != torch.float16]
    if non_f16:
        raise ValueError(f"NNRT model requires F16 tensors; first mismatch: {non_f16[0]}")

    embedding = tensors["emb.weight"].to(dtype=torch.float32)
    scale = tensors["blocks.0.ln0.weight"].to(dtype=torch.float32)
    bias = tensors["blocks.0.ln0.bias"].to(dtype=torch.float32)
    tensors["emb.weight"] = F.layer_norm(
        embedding,
        (embedding.shape[-1],),
        scale,
        bias,
        1.0e-5,
    ).to(dtype=torch.float16).contiguous()
    tensors[MARKER] = torch.ones((1,), dtype=torch.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "format": "rwkv-harmony-nnrt-v1",
            "embedding_ln0": "precomputed-fp32-to-fp16",
        },
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "input_bytes": args.input.stat().st_size,
                "input_sha256": input_sha256,
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "marker": MARKER,
                "torch_threads": CONVERSION_THREADS,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "safetensors": safetensors.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
