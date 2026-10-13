"""Projection scaling invariants; device numerical validation is separate."""

import unittest
import torch
from rwkv_src.rwkv_v7_modules import Rwkv7SelfAttention


class AttentionOutputScaleTests(unittest.TestCase):
    def projection(self, scale, dtype=torch.float32):
        torch.manual_seed(7)
        layer = Rwkv7SelfAttention.__new__(Rwkv7SelfAttention)
        torch.nn.Module.__init__(layer)
        layer.output = torch.nn.Linear(128, 128, bias=False, dtype=dtype)
        layer.att_output_scale = scale
        return layer

    def test_scaling_preserves_projection_and_weight_storage(self):
        x = torch.linspace(-0.1, 0.1, 512).reshape(1, 4, 128)
        for scale in (1, 16, 64):
            layer = self.projection(scale)
            weight = layer.output.weight.detach().clone()
            address = layer.output.weight.data_ptr()
            expected = layer.output(x)
            actual = layer._output_projection(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(layer.output.weight, weight, rtol=0, atol=0)
            self.assertEqual(layer.output.weight.data_ptr(), address)

    def test_half_precision_small_inputs_remain_finite(self):
        layer = self.projection(64, torch.float16)
        x = torch.linspace(-1e-4, 1e-4, 512).reshape(1, 4, 128).half()
        expected = torch.nn.functional.linear(x.float(), layer.output.weight.float()).half()
        actual = layer._output_projection(x)
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=6e-8)

    def test_coreml_export_preserves_scaling_and_weights(self):
        try:
            import coremltools as ct
            import numpy as np
        except ImportError:
            self.skipTest('Core ML converter dependencies are not installed')

        class Projection(torch.nn.Module):
            def __init__(self, attention):
                super().__init__()
                self.attention = attention

            def forward(self, x):
                return self.attention._output_projection(x)

        layer = self.projection(64).eval()
        traced = torch.jit.trace(Projection(layer), torch.zeros(1, 1, 128))
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name='x', shape=(1, 1, 128), dtype=np.float16)],
            outputs=[ct.TensorType(name='y', dtype=np.float16)],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
        operations = list(converted._mil_program.functions['main'].operations)
        self.assertEqual([op.op_type for op in operations if op.op_type != 'const'],
                         ['mul', 'linear', 'mul'])
        linear = next(op for op in operations if op.op_type == 'linear')
        np.testing.assert_array_equal(
            linear.weight.val, layer.output.weight.detach().numpy().astype(np.float16),
        )


if __name__ == '__main__':
    unittest.main()
