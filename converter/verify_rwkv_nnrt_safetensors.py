#!/usr/bin/env python3

"""Verify an NNRT-prepared RWKV-7 safetensors file against its generic source."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


MARKER = "rwkv.nnrt.emb_ln0_precomputed"
CONVERSION_THREADS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--expected-output-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source.resolve() == args.prepared.resolve():
        raise ValueError("source and prepared files must be different")

    torch.set_num_threads(CONVERSION_THREADS)
    source_tensors = load_file(str(args.source), device="cpu")
    prepared_tensors = load_file(str(args.prepared), device="cpu")
    required = ("emb.weight", "blocks.0.ln0.weight", "blocks.0.ln0.bias")
    missing = [name for name in required if name not in source_tensors]
    if missing:
        raise ValueError(f"source is missing required tensors: {', '.join(missing)}")
    if set(prepared_tensors) != set(source_tensors) | {MARKER}:
        raise ValueError("prepared tensor names do not match source plus NNRT marker")

    marker = prepared_tensors[MARKER]
    if marker.dtype != torch.float16 or marker.shape != (1,) or marker.item() != 1.0:
        raise ValueError("invalid NNRT marker tensor")

    for name, tensor in source_tensors.items():
        if tensor.dtype != torch.float16:
            raise ValueError(f"source tensor is not F16: {name}")
        if name == "emb.weight":
            continue
        if not torch.equal(tensor, prepared_tensors[name]):
            raise ValueError(f"non-embedding tensor changed: {name}")

    embedding = source_tensors["emb.weight"].to(dtype=torch.float32)
    scale = source_tensors["blocks.0.ln0.weight"].to(dtype=torch.float32)
    bias = source_tensors["blocks.0.ln0.bias"].to(dtype=torch.float32)
    expected_embedding = F.layer_norm(
        embedding,
        (embedding.shape[-1],),
        scale,
        bias,
        1.0e-5,
    ).to(dtype=torch.float16).contiguous()
    if not torch.equal(expected_embedding, prepared_tensors["emb.weight"]):
        raise ValueError("prepared embedding does not match the FP32 LayerNorm reference")

    output_sha256 = sha256_file(args.prepared)
    if (
        args.expected_output_sha256 is not None
        and output_sha256 != args.expected_output_sha256
    ):
        raise ValueError(
            f"prepared SHA-256 mismatch: {output_sha256} "
            f"!= {args.expected_output_sha256}"
        )

    print(
        json.dumps(
            {
                "source": args.source.name,
                "source_bytes": args.source.stat().st_size,
                "source_sha256": sha256_file(args.source),
                "prepared": args.prepared.name,
                "prepared_bytes": args.prepared.stat().st_size,
                "prepared_sha256": output_sha256,
                "tensor_count_source": len(source_tensors),
                "tensor_count_prepared": len(prepared_tensors),
                "marker_valid": True,
                "all_non_embedding_tensors_exact": True,
                "embedding_fp32_layer_norm_exact": True,
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
