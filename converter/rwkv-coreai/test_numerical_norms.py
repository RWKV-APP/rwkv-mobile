"""Regression checks for normalization overflow, without loading model weights."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rwkv-coreml"))
from rwkv_src.rwkv_v7_modules import Rwkv7SelfAttention


class NumericalNormTests(unittest.TestCase):
    def setUp(self):
        self.att = Rwkv7SelfAttention.__new__(Rwkv7SelfAttention)
        torch.nn.Module.__init__(self.att)
        self.att.num_heads = 2
        self.att.head_size = 64
        self.att.hidden_size = 128

    def test_group_norm_large_finite_inputs_do_not_collapse(self):
        x = torch.zeros(1, 128, dtype=torch.float16)
        x[0, 0], x[0, 64] = 600, -400
        groups = x.float().reshape(1, 2, 64)
        centered = groups - groups.mean(-1, keepdim=True)
        expected = centered / (centered.square().mean(-1, keepdim=True) + 64e-5).sqrt()
        actual = self.att._head_group_norm(x, 1, 1)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual, expected.reshape(1, 1, 128).half())
        self.assertGreater(actual.abs().max().item(), 7)

    def test_key_norm_handles_zero_tiny_and_large_vectors(self):
        x = torch.tensor([[0., 0.], [1e-4, 1e-4], [600., 600.]], dtype=torch.float16)
        expected = torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6).half()
        actual = self.att._normalize_heads(x)
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual, expected)

    def test_fp32_group_norm_matches_original_formula(self):
        x = torch.arange(256, dtype=torch.float32).reshape(2, 128)
        v = x.reshape(2, 2, 64)
        v = v - v.mean(-1, keepdim=True)
        expected = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 64e-5)).reshape(1, 2, 128)
        torch.testing.assert_close(self.att._head_group_norm(x, 2, 1), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
