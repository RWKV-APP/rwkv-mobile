import unittest

import coreai_torch
import torch

from quantization import protected_weight_keys, quantize_protected_weights


class TinyFFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.ffn = torch.nn.Module()
        block.ffn.key = torch.nn.Linear(64, 64, bias=False)
        block.ffn.value = torch.nn.Linear(64, 64, bias=False)
        self.blocks = torch.nn.ModuleList([block])

    def forward(self, x):
        return self.blocks[0].ffn.value(self.blocks[0].ffn.key(x).relu())


class RecipeTests(unittest.TestCase):
    def test_selected_constant_survives_general_w4_pass(self):
        from coreai._compiler.ir import WalkResult
        from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

        torch.manual_seed(91)
        model = TinyFFN().half().eval()
        keys = protected_weight_keys(model, ["blocks.0.ffn.key"])
        converter = coreai_torch.TorchConverter()
        for name, length in [("decode", 1), ("prefill", 16)]:
            exported = torch.export.export(model, (torch.ones(length, 64).half(),))
            converter.add_exported_program(exported.run_decompositions(coreai_torch.get_decomp_table()), entrypoint_name=name)
        program = converter.to_coreai()
        program.optimize()
        program, matches = quantize_protected_weights(program, keys)
        self.assertEqual(matches["blocks.0.ffn.key"], 2)
        program = quantize_weights(program, DType.INT4, granularity=CompressionGranularity.PER_BLOCK, block_size=32, in_place=True)
        matrices = []

        def audit(op):
            if op.name == "coreai.constant" and list(op.result.type.shape) == [64, 64]:
                matrices.append(str(op.result.type.element_type))
            return WalkResult.ADVANCE

        program._mlir_module.operation.walk(audit)
        self.assertEqual(sorted(matrices), ["si4", "si4", "si8", "si8"])

    def test_unknown_duplicate_and_changed_weights_fail(self):
        model = TinyFFN().half()
        with self.assertRaises(ValueError):
            protected_weight_keys(model, ["blocks.99.ffn.key"])
        with self.assertRaises(ValueError):
            protected_weight_keys(model, ["blocks.0.ffn.key"] * 2)
        keys = protected_weight_keys(model, ["blocks.0.ffn.key"])
        with torch.no_grad():
            model.blocks[0].ffn.key.weight.add_(1)
        converter = coreai_torch.TorchConverter()
        exported = torch.export.export(model, (torch.ones(1, 64).half(),))
        converter.add_exported_program(exported.run_decompositions(coreai_torch.get_decomp_table()))
        program = converter.to_coreai()
        program.optimize()
        with self.assertRaisesRegex(ValueError, "absent"):
            quantize_protected_weights(program, keys)


if __name__ == "__main__":
    unittest.main()
