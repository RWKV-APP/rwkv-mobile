import json
import tempfile
import unittest
from pathlib import Path

from normalize_mlx_config import prepare_config


class MLXConfigTest(unittest.TestCase):
    def test_partial_overrides_and_existing_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {
                "model_type": "rwkv7",
                "quantization": {
                    "bits": 6, "group_size": 64,
                    "model.embeddings": {"bits": 8},
                    "model.layers.0.attn.a_lora.lora.2": {"bits": 8},
                    "lm_head": {"bits": 6, "group_size": 32},
                    "skip": False,
                },
                "quantization_config": {
                    "bits": 6, "group_size": 128,
                    "model.embeddings": {"bits": 8},
                },
            }
            path.write_text(json.dumps(config))
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                prepare_config(path, check=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(prepare_config(path), 4)
            expected = json.loads(original)
            expected["quantization"]["model.embeddings"]["group_size"] = 64
            expected["quantization"]["model.layers.0.attn.a_lora.lora.2"]["group_size"] = 64
            expected["quantization_config"]["model.embeddings"]["group_size"] = 128
            expected["quantization"]["model.layers.0.attn.a_lora.up_proj"] = {"bits": 8, "group_size": 64}
            self.assertEqual(json.loads(path.read_text()), expected)
            normalized = path.read_bytes()
            self.assertEqual(prepare_config(path, check=True), 0)
            self.assertEqual(prepare_config(path), 0)
            self.assertEqual(path.read_bytes(), normalized)

    def test_swift_aliases_preserve_python_names_and_reject_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field in ("quantization", "quantization_config"):
                with self.subTest(field=field):
                    config = {"model_type": "rwkv7", field: {
                        "bits": 6, "group_size": 64,
                        "model.layers.0.attn.w_lora.lora.0": False,
                        "model.layers.0.attn.w_lora.lora.2": {"bits": 8, "group_size": 64},
                    }}
                    path.write_text(json.dumps(config))
                    self.assertEqual(prepare_config(path), 2)
                    normalized = json.loads(path.read_text())
                    self.assertFalse(normalized[field]["model.layers.0.attn.w_lora.down_proj"])
                    self.assertEqual(normalized[field]["model.layers.0.attn.w_lora.up_proj"],
                                     config[field]["model.layers.0.attn.w_lora.lora.2"])
                    self.assertEqual(prepare_config(path, check=True), 0)
                    normalized[field]["model.layers.0.attn.w_lora.up_proj"]["bits"] = 6
                    path.write_text(json.dumps(normalized))
                    original = path.read_bytes()
                    with self.assertRaises(ValueError):
                        prepare_config(path)
                    self.assertEqual(path.read_bytes(), original)

    def test_invalid_group_size_never_changes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for group_size in (None, 0, -1, True, "64"):
                with self.subTest(group_size=group_size):
                    path.write_text(json.dumps({"quantization": {
                        "group_size": group_size, "bits": 6,
                        "model.embeddings": {"bits": 8},
                    }}))
                    original = path.read_bytes()
                    with self.assertRaises(ValueError):
                        prepare_config(path)
                    self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
