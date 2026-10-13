"""Exact-weight selection for the pinned Core AI 0.2.1 compression pipeline."""

import hashlib

import numpy as np


def weight_key(weight):
    array = np.ascontiguousarray(weight)
    return (array.shape, array.dtype.str, hashlib.sha256(array.tobytes()).hexdigest())


def protected_weight_keys(model, module_names):
    """Bind explicit FFN names to their exported half-precision weight bytes."""
    modules = dict(model.named_modules())
    keys = {}
    if len(module_names) != len(set(module_names)):
        raise ValueError("protected_modules contains duplicates")
    for name in module_names:
        if name not in modules or not name.startswith("blocks.") or not name.endswith((".ffn.key", ".ffn.value")):
            raise ValueError(f"Not an FFN matrix: {name}")
        weight = modules[name].weight.detach().cpu().numpy()
        for value in (weight, weight.T):
            keys.setdefault(weight_key(value), set()).add(name)
    return keys


def quantize_protected_weights(program, keys):
    """Compress selected constants first, before the general W4/W8 passes.

    coreai-opt's graph pass has no per-module selector. Reuse its numerical
    helpers and IR builder, selecting by exact bytes rather than compiler names.
    The subsequent passes skip these already-compressed weights. Fail closed if
    optimization changed any selected constant so it cannot be identified.
    """
    from coreai_opt.coreai_utils._coreai_imports import (
        InsertionPoint, IntegerType, WalkResult,
        _get_constant_value_as_np_array, compression_types,
    )
    from coreai_opt.coreai_utils._utils.graph_utils import (
        _apply_compression_transform, _infer_quantization_block_sizes, _should_compress_op,
    )
    from coreai_opt.coreai_utils._utils.quantize_utils import _compute_qparams_by_dtype
    from coreai_opt.coreai_utils.common import CompressionGranularity, QScheme
    from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION
    from coreai_opt.coreai_utils.passes.weight_quantization import _create_int_quantized_weight

    matches = {name: 0 for names in keys.values() for name in names}
    shapes = {key[0] for key in keys}

    def replace(op):
        if not _should_compress_op(op, 1024, _OPS_WEIGHT_NEED_COMPRESSION):
            return WalkResult.ADVANCE
        if tuple(op.result.type.shape) not in shapes:
            return WalkResult.ADVANCE
        weight = _get_constant_value_as_np_array(op)
        names = keys.get(weight_key(weight))
        if not names:
            return WalkResult.ADVANCE
        blocks = _infer_quantization_block_sizes(op, weight.shape, CompressionGranularity.PER_BLOCK, 32)
        q, scale, zero = _compute_qparams_by_dtype(
            weight, compression_types.string_to_builtin("int8"), QScheme.SYMMETRIC, list(blocks))
        if zero is None:
            zero = np.zeros_like(scale, dtype=q.dtype)
        with op.context, op.location, InsertionPoint(op):
            value = _create_int_quantized_weight(q, scale, zero, IntegerType.get_signed(8), op.result.type.element_type)
        op.result.replace_all_uses_with(value)
        for name in names:
            matches[name] += 1
        return WalkResult.ADVANCE

    program = _apply_compression_transform(program, replace, True)
    missing = [name for name, count in matches.items() if count == 0]
    if missing:
        raise ValueError(f"Selected weights absent from exported graph: {missing}")
    return program, matches
