"""Normalize MLX quantization metadata for Python and Swift before packaging."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def normalize_group_sizes(config):
    changed = 0
    for field in ("quantization", "quantization_config"):
        quantization = config.get(field)
        if not isinstance(quantization, dict):
            continue
        default = quantization.get("group_size")
        for name, layer in quantization.items():
            if not isinstance(layer, dict) or "bits" not in layer:
                continue
            group_size = layer.get("group_size", default)
            if type(group_size) is not int or group_size <= 0:
                raise ValueError(f"{field}.{name}: missing or invalid group_size")
            if "group_size" not in layer:
                layer["group_size"] = group_size
                changed += 1
    return changed


def add_rwkv7_swift_aliases(config):
    # Rwkv7.swift sanitizes tensor names, but not the per-layer config keys.
    if config.get("model_type") != "rwkv7":
        return 0
    changed = 0
    for field in ("quantization", "quantization_config"):
        quantization = config.get(field)
        if not isinstance(quantization, dict):
            continue
        for name, layer in list(quantization.items()):
            for suffix, replacement in ((".lora.0", ".down_proj"), (".lora.2", ".up_proj")):
                if not name.endswith(suffix):
                    continue
                alias = name[:-len(suffix)] + replacement
                if alias in quantization:
                    if quantization[alias] != layer:
                        raise ValueError(f"{field}.{alias}: conflicting Python and Swift quantization")
                else:
                    quantization[alias] = layer.copy() if isinstance(layer, dict) else layer
                    changed += 1
    return changed


def prepare_config(path, check=False):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    changed = normalize_group_sizes(config)
    changed += add_rwkv7_swift_aliases(config)
    if check and changed:
        raise ValueError(f"{changed} quantization metadata updates need normalization")
    if not changed:
        return 0
    # Replace only complete JSON; interruption must not truncate the model config.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(config, output, indent=4, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary.chmod(path.stat().st_mode & 0o777)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Converted MLX model's config.json")
    parser.add_argument("--check", action="store_true", help="Fail without writing if normalization is needed")
    args = parser.parse_args()
    print(f"Normalized {prepare_config(args.config, args.check)} quantization metadata entries")
