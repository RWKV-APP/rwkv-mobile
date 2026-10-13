"""Export RWKV7 recurrent state and separate decode/prefill functions to Core AI.

Requires Python 3.12, torch 2.11, coreai-torch 0.4.2 and macOS 27 for runtime
validation. Artifacts must live outside the source tree.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import coreai_torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rwkv-coreml"))
from rwkv_src.rwkv_modeling import RWKV_RNN, RWKV_RNN_Stateful


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens)[:, -1:, :]


class MixedLinear(torch.nn.Module):
    """Use half matmul inputs/weights, with FP32 residuals and recurrent math."""

    def __init__(self, linear):
        super().__init__()
        self.weight = torch.nn.Parameter(linear.weight.detach().half(), requires_grad=False)
        self.bias = None if linear.bias is None else torch.nn.Parameter(linear.bias.detach().half(), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.half(), self.weight, self.bias).float()


class MixedEmbedding(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.weight = torch.nn.Parameter(embedding.weight.detach().half(), requires_grad=False)

    def forward(self, tokens):
        return torch.nn.functional.embedding(tokens, self.weight).float()


def use_mixed_matmuls(model):
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, MixedLinear(child))
            elif isinstance(child, torch.nn.Embedding):
                setattr(parent, name, MixedEmbedding(child))


def palettize_lut6(program, group_size, weight_num_threshold=1024):
    """Use scalar grouped-channel LUT6, preserving the frequency of each weight.

    coreai-opt 0.2.1 imports the public kmeans1d package, whose cluster API lacks
    the weighted argument it calls. Core ML Tools 9 supplies Apple's compatible
    weighted implementation. Keep the workaround local to this compression call.
    """
    import inspect
    from coreai_opt.coreai_utils import CompressionGranularity, palettize_weights
    from coreai_opt.coreai_utils._utils import palettize_utils

    previous = palettize_utils._kmeans1d, palettize_utils._HAS_KMEANS1D
    if previous[0] is None or "weights" not in inspect.signature(previous[0].cluster).parameters:
        from coremltools._deps import kmeans1d

        palettize_utils._kmeans1d = kmeans1d
        palettize_utils._HAS_KMEANS1D = True
    try:
        return palettize_weights(
            program, None, n_bits=6,
            granularity=CompressionGranularity.PER_GROUPED_CHANNEL,
            group_size=group_size, num_kmeans_workers=1,
            weight_num_threshold=weight_num_threshold,
            enable_fast_kmeans_mode=False, in_place=True,
        )
    finally:
        palettize_utils._kmeans1d, palettize_utils._HAS_KMEANS1D = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefill-length", type=int, default=16)
    parser.add_argument("--prefill-tails", type=int, nargs="*", default=[],
                        help="Additional smaller prefill functions for partial chunks")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float32",
                        help="FP32 reference by default; FP16 requires separate numerical validation")
    parser.add_argument("--quantization", choices=["none", "w8", "w4", "w4-w8", "lut6", "lut6-w8"], default="none",
                        help="Integer or LUT6 weight compression with FP32 recurrent state")
    parser.add_argument("--group-size", type=int, choices=[16, 32, 64, 128], default=32)
    parser.add_argument("--lut-weight-threshold", type=int, default=1048576,
                        help="LUT6+W8: matrices above this element count are LUT candidates")
    parser.add_argument("--int8-large-threshold", type=int,
                        help="LUT6+W8: first keep matrices above this element count in INT8")
    parser.add_argument("--mixed-matmuls", action="store_true",
                        help="Quantized half-precision matmuls with FP32 normalization, residuals and state")
    parser.add_argument("--quantization-recipe", type=Path,
                        help="Source-bound JSON listing FFN modules to retain in W8 within w4-w8")
    parser.add_argument("--fused-wkv", action="store_true", help="Experimental FP32 Metal WKV recurrence")
    parser.add_argument("--fused-decode", action="store_true", help="Also use the experimental WKV kernel for decode")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new artifact directory")
    if args.prefill_length < 2:
        parser.error("prefill-length must be at least 2")
    if (len(args.prefill_tails) != len(set(args.prefill_tails))
            or any(n < 2 or n >= args.prefill_length for n in args.prefill_tails)):
        parser.error("prefill-tails must be unique lengths smaller than prefill-length and at least 2")
    if args.fused_decode and not args.fused_wkv:
        parser.error("fused-decode requires fused-wkv")
    if args.quantization != "none" and args.precision != "float32":
        parser.error("quantized experiments require FP32 computation and recurrent state")
    if args.mixed_matmuls and (args.quantization == "none" or args.precision != "float32"):
        parser.error("mixed-matmuls requires compressed weights and FP32 recurrent state")
    if args.quantization_recipe and (args.quantization != "w4-w8" or not args.mixed_matmuls):
        parser.error("quantization-recipe requires w4-w8 and mixed-matmuls")
    if args.fused_wkv and args.precision != "float32":
        parser.error("fused-wkv requires FP32 recurrent computation")
    if args.lut_weight_threshold < 1024:
        parser.error("lut-weight-threshold must be at least 1024")
    if args.int8_large_threshold is not None and (
        args.quantization != "lut6-w8" or args.int8_large_threshold <= args.lut_weight_threshold
    ):
        parser.error("int8-large-threshold requires lut6-w8 and must exceed lut-weight-threshold")

    with args.model.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    recipe = None
    protected_keys = None
    protected_matches = None
    if args.quantization_recipe:
        recipe = json.loads(args.quantization_recipe.read_text())
        if (recipe.get("format_version") != 1 or recipe.get("source_sha256") != digest
                or recipe.get("group_size") != args.group_size
                or not isinstance(recipe.get("protected_modules"), list)
                or not all(isinstance(name, str) for name in recipe["protected_modules"])):
            parser.error("recipe format, source SHA-256, group size or module list is invalid")
    options = SimpleNamespace(
        MODEL_NAME=str(args.model), USE_CUDA=False,
        fp16=args.precision == "float16", USE_EMBEDDING=True, SKIP_LMHEAD=False,
    )
    full_model = RWKV_RNN(options)
    stateful = RWKV_RNN_Stateful.from_full_model(full_model)
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    model = ExportModel(stateful).to(dtype=dtype).eval()
    if args.mixed_matmuls:
        use_mixed_matmuls(model)
    if recipe:
        from quantization import protected_weight_keys
        protected_keys = protected_weight_keys(stateful, recipe["protected_modules"])
    if args.fused_wkv:
        from wkv_kernel import use_fused_wkv
        use_fused_wkv(stateful, decode=args.fused_decode)
    converter = coreai_torch.TorchConverter()
    if args.fused_wkv:
        from wkv_kernel import WKV_KERNEL
        converter.register_custom_kernels([WKV_KERNEL])
    entrypoints = [("decode", 1), ("prefill", args.prefill_length)]
    entrypoints.extend((f"prefill_{n}", n) for n in sorted(args.prefill_tails, reverse=True))
    for name, length in entrypoints:
        stateful.state_tokenshift.zero_()
        stateful.state_wkv.zero_()
        tokens = torch.zeros((1, length), dtype=torch.int32)
        print(f"Exporting {name} ({length} tokens)", flush=True)
        exported = torch.export.export(model, (tokens,))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        converter.add_exported_program(
            exported, entrypoint_name=name, input_names=["tokens"],
            state_names=["tokenshift", "wkv"], output_names=["logits"],
        )
    print("Converting Core AI graph", flush=True)
    program = converter.to_coreai()
    program.optimize()
    if protected_keys:
        from quantization import quantize_protected_weights
        program, protected_matches = quantize_protected_weights(program, protected_keys)
        print(f"Protected W8 matrices: {protected_matches}", flush=True)
    if args.quantization in ("lut6", "lut6-w8"):
        if args.int8_large_threshold is not None:
            from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

            print(f"Keeping matrices above {args.int8_large_threshold} elements in INT8", flush=True)
            program = quantize_weights(program, DType.INT8,
                                       granularity=CompressionGranularity.PER_BLOCK,
                                       block_size=32, weight_num_threshold=args.int8_large_threshold,
                                       in_place=True)
        print(f"Compressing LUT6 weights in groups of {args.group_size} output channels", flush=True)
        program = palettize_lut6(program, args.group_size,
                                 args.lut_weight_threshold if args.quantization == "lut6-w8" else 1024)
        if args.quantization == "lut6-w8":
            from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

            print("Compressing remaining eligible matrices to INT8, group 32", flush=True)
            program = quantize_weights(program, DType.INT8,
                                       granularity=CompressionGranularity.PER_BLOCK,
                                       block_size=32, in_place=True)
    elif args.quantization == "w4-w8":
        from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

        # For the 1.5B shape this selects the FFN matrices for W4 and retains
        # vocabulary, attention and low-rank matrices in W8. The manifest
        # records the size policy; other model shapes require a fresh audit.
        for dtype, threshold, group in [(DType.INT8, 67108864, 32),
                                        (DType.INT4, 8388608, args.group_size),
                                        (DType.INT8, 1024, 32)]:
            print(f"Compressing {dtype} matrices above {threshold} elements", flush=True)
            program = quantize_weights(program, dtype,
                                       granularity=CompressionGranularity.PER_BLOCK,
                                       block_size=group, weight_num_threshold=threshold,
                                       in_place=True)
    elif args.quantization != "none":
        from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

        print(f"Compressing {args.quantization} weights in groups of {args.group_size}", flush=True)
        program = quantize_weights(
            program, DType.INT8 if args.quantization == "w8" else DType.INT4,
            granularity=CompressionGranularity.PER_BLOCK,
            block_size=args.group_size, in_place=True,
        )
    args.output.mkdir(parents=True)
    if args.quantization != "none":
        from coreai._compiler.ir import WalkResult

        counts = Counter()
        def audit(op):
            if op.name in ("coreai.lut_to_dense", "coreai.blockwise_shift_scale"):
                counts[op.name] += 1
            if op.name == "coreai.constant":
                counts["constant_" + str(op.result.type.element_type)] += 1
            return WalkResult.ADVANCE

        program._mlir_module.operation.walk(audit)
        if args.quantization in ("lut6", "lut6-w8") and not counts["coreai.lut_to_dense"]:
            raise RuntimeError("LUT6 compression produced no lookup-table operations")
        if args.quantization in ("w4", "w8", "w4-w8", "lut6-w8") and not counts["coreai.blockwise_shift_scale"]:
            raise RuntimeError("Integer compression produced no decompression operations")
        (args.output / "compression.json").write_text(json.dumps(counts, indent=2) + "\n")
    program.save_asset(args.output / "rwkv.aimodel")
    metadata = {
        "format_version": 1, "backend": "coreai", "model_version": 7,
        "source_sha256": digest, "precision": args.precision,
        "quantization": args.quantization, "model_file": "rwkv.aimodel",
        "quantization_scheme": "kmeans_lut" if args.quantization in ("lut6", "lut6-w8") else (
            "symmetric" if args.quantization != "none" else None),
        "quantization_group_size": args.group_size if args.quantization != "none" else None,
        "quantization_granularity": "per_grouped_channel" if args.quantization in ("lut6", "lut6-w8") else (
            "per_block" if args.quantization != "none" else None),
        "activation_precision": "mixed_float16_float32" if args.mixed_matmuls else args.precision,
        "matmul_precision": "float16" if args.mixed_matmuls else args.precision,
        "state_precision": args.precision,
        "wkv_implementation": "metal_row64_fp32" if args.fused_wkv else "graph",
        "decode_wkv_implementation": "metal_row64_fp32" if args.fused_decode else "graph",
        "n_layers": options.n_layer, "hidden_size": options.n_embd,
        "num_heads": options.n_head, "head_size": options.head_size,
        "vocab_size": options.vocab_size, "prefill_length": args.prefill_length,
        "tokenshift_shape": list(stateful.state_tokenshift.shape),
        "wkv_shape": list(stateful.state_wkv.shape),
    }
    if args.prefill_tails:
        metadata["prefill_lengths"] = sorted([args.prefill_length, *args.prefill_tails], reverse=True)
    if args.quantization == "lut6-w8":
        metadata["integer_fallback"] = {"bits": 8, "scheme": "symmetric", "group_size": 32}
        metadata["lut_weight_num_threshold"] = args.lut_weight_threshold
        metadata["integer_prepass_weight_num_threshold"] = args.int8_large_threshold
    if args.quantization == "w4-w8":
        metadata["integer_fallback"] = {"bits": 8, "scheme": "symmetric", "group_size": 32}
        metadata["w4_weight_num_threshold"] = 8388608
        metadata["integer_prepass_weight_num_threshold"] = 67108864
    if recipe:
        metadata["protected_w8_modules"] = recipe["protected_modules"]
        metadata["protected_w8_graph_matches"] = protected_matches
        metadata["quantization_recipe_sha256"] = hashlib.sha256(args.quantization_recipe.read_bytes()).hexdigest()
        (args.output / "quantization-recipe.json").write_bytes(args.quantization_recipe.read_bytes())
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Export complete", flush=True)


if __name__ == "__main__":
    main()
