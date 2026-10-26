"""Expose released G1j WKV state as tensors without requantizing weights.

Only the known wkv-coreml MIL export and CMLD description layout are accepted.
For new exports use convert_rwkv_to_coreml_int4_import.py --state-mode tensor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_mil(text):
    require(not re.search(r'\bstate_wkv_(?:in|out)\b', text),
            'Model already has external WKV state')
    require(re.findall(r'\bfunc (\w+)<', text) == ['decode', 'prefill'], 'Unsupported functions')
    headers = re.findall(r'\bfunc \w+<[^>]+>\(([^\n]+)\) {', text)
    require(all('v_first_in' not in h or h.index('state_wkv') < h.index('v_first_in') for h in headers),
            'Unsupported state argument order')
    types = re.findall(r'state<(tensor<fp16, \[([\d, ]+)\]>)> state_wkv', text)
    require(len(types) == 2 and types[0] == types[1], 'Expected matching decode/prefill WKV states')
    typ, dimensions = types[0]
    shape = [int(n) for n in dimensions.split(',')]
    require(len(shape) == 4 and all(n > 0 for n in shape), 'Invalid WKV shape')
    # Replacing a partial update by its payload would silently lose retained state.
    for name, value in {
        'concat_0': '[0, 0, 0, 0]', 'concat_1': '[0, 0, 0, 0]',
        'state_wkv_internal_tensor_assign_1_stride_0': '[1, 1, 1, 1]',
        'state_wkv_internal_tensor_assign_1_begin_mask_0': '[false, true, true, true]',
        'state_wkv_internal_tensor_assign_1_end_mask_0': '[true, true, true, true]',
        'state_wkv_internal_tensor_assign_1_squeeze_mask_0': '[false, false, false, false]',
    }.items():
        lines = re.findall(r'^.*\b' + name + r' = const\(\).*$', text, re.M)
        require(len(lines) == 2 and all(f'>({value})' in line for line in lines),
                f'Unsupported update window: {name}')
    text = text.replace(f'state<{typ}> state_wkv', f'{typ} state_wkv_in')
    read = f'{typ} read_state_0 = read_state(input = state_wkv)[name = string("read_state_0")];'
    require(text.count(read) == 2, 'Unsupported state read')
    text = text.replace(read, f'{typ} read_state_0 = identity(x = state_wkv_in)[name = string("read_state_0")];')
    update = (re.escape(typ) + r' state_wkv_internal_tensor_assign_1_cast_fp16 = slice_update\('
              r'begin = concat_0, begin_mask = state_wkv_internal_tensor_assign_1_begin_mask_0, '
              r'end = concat_1, end_mask = state_wkv_internal_tensor_assign_1_end_mask_0, '
              r'squeeze_mask = state_wkv_internal_tensor_assign_1_squeeze_mask_0, '
              r'stride = state_wkv_internal_tensor_assign_1_stride_0, update = (\w+), x = read_state_0\)'
              r'\[name = string\("state_wkv_internal_tensor_assign_1_cast_fp16"\)\];\s+'
              r'write_state\(data = state_wkv_internal_tensor_assign_1_cast_fp16, input = state_wkv\)'
              r'\[name = string\("coreml_update_state_\d+_write_state"\)\];')
    text, count = re.subn(update, lambda m: f'{typ} state_wkv_out = identity(x = {m[1]})[name = string("state_wkv_out")];', text)
    require(count == 2, 'Unsupported state write')
    text, count = re.subn(r'^([ \t]+} -> \()(out0, state_tokenshift_out(?:, v_first_out)?)(\);)$',
                          r'\1\2, state_wkv_out\3', text, flags=re.M)
    require(count == 2 and not re.search(r'\b(read_state|write_state)\(', text),
            'Unsupported function outputs or remaining state operations')
    return text, shape


def tensor_description(data, shape):
    from coremltools.proto import Model_pb2
    # shortcut: compiled CMLD is version-specific; reject other layouts and re-export them.
    candidates = []
    for offset in range(min(len(data) - 8, 160)):
        size = struct.unpack_from('<Q', data, offset)[0]
        if not 0 < size <= len(data) - offset - 8:
            continue
        desc = Model_pb2.ModelDescription()
        try:
            desc.ParseFromString(data[offset + 8:offset + 8 + size])
        except Exception:
            continue
        if [f.name for f in desc.functions] == ['decode', 'prefill']:
            candidates.append((offset, size, desc))
    require(data[:8] == struct.pack('<II', 502, 9) and len(candidates) == 1, 'Unsupported CMLD layout')
    offset, size, desc = candidates[0]
    for fn in desc.functions:
        require(len(fn.state) == 1 and fn.state[0].name == 'state_wkv', 'Unexpected states')
        array = fn.state[0].type.stateType.arrayType
        from coremltools.proto import FeatureTypes_pb2
        require(list(array.shape) == shape and array.dataType == FeatureTypes_pb2.ArrayFeatureType.FLOAT16, 'State schema mismatch')
        for features, name in [(fn.input, 'state_wkv_in'), (fn.output, 'state_wkv_out')]:
            require(all(f.name != name for f in features), 'Duplicate state feature')
            feature = features.add()
            feature.name = name
            feature.type.multiArrayType.CopyFrom(array)
            if name == 'state_wkv_in':
                index = next((i for i, f in enumerate(features) if f.name == 'v_first_in'), None)
                if index is not None:
                    item = type(feature)()
                    item.CopyFrom(feature)
                    del features[-1]
                    features.insert(index, item)
        del fn.state[:]
    require(not desc.state and not desc.input and not desc.output, 'Unexpected top-level schema')
    encoded = desc.SerializeToString()
    return data[:offset] + struct.pack('<Q', len(encoded)) + encoded + data[offset + 8 + size:]


def tensor_metadata(data, shape):
    rows = json.loads(data)
    for row in rows:
        for schema in [row, *row.get('functions', [])]:
            require(len(schema.get('stateSchema', [])) == 1 and schema['stateSchema'][0]['name'] == 'state_wkv',
                    'Unexpected metadata state schema')
            schema['stateSchema'] = []
            for key, name in [('inputSchema', 'state_wkv_in'), ('outputSchema', 'state_wkv_out')]:
                entry = {'name': name, 'type': 'MultiArray', 'dataType': 'Float16',
                                    'shape': str(shape), 'formattedType': 'MultiArray (Float16 ' + ' × '.join(map(str, shape)) + ')',
                                    'isOptional': '0', 'hasShapeFlexibility': '0', 'shortDescription': ''}
                index = next((i for i, f in enumerate(schema[key]) if key == 'inputSchema' and f['name'] == 'v_first_in'), len(schema[key]))
                schema[key].insert(index, entry)
        row.pop('mlProgramOperationTypeHistogram', None)
    return json.dumps(rows, indent=2) + '\n'


def sha256(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def repair(source, output):
    require(source.is_dir() and not output.exists(), 'Source must exist; output must be new')
    source = source.resolve()
    require(not output.resolve().is_relative_to(source), 'Output must be outside source')
    require(not any(p.is_symlink() for p in source.rglob('*')), 'Symlink inputs are unsupported')
    config = (source / 'config.yaml').read_text()
    require(len(re.findall(r'^state_mode: wkv-coreml$', config, re.M)) == 1, 'Expected wkv-coreml config')
    basename = re.findall(r'^basename: ([\w.-]+)$', config, re.M)
    chunks = re.findall(r'^num_chunks: (\d+)$', config, re.M)
    require(len(basename) == len(chunks) == 1 and int(chunks[0]) > 0, 'Invalid chunk config')
    changes = {}
    for chunk in range(1, int(chunks[0]) + 1):
        folder = Path(f'{basename[0]}_chunk{chunk}of{chunks[0]}.mlmodelc')
        mil, shape = tensor_mil((source / folder / 'model.mil').read_text())
        changes[folder / 'model.mil'] = mil.encode()
        changes[folder / 'coremldata.bin'] = tensor_description((source / folder / 'coremldata.bin').read_bytes(), shape)
        changes[folder / 'metadata.json'] = tensor_metadata((source / folder / 'metadata.json').read_text(), shape).encode()
    weights = {str(p.relative_to(source)): sha256(p) for p in source.glob('*.mlmodelc/weights/*') if p.is_file()}
    require(bool(weights), 'Missing model weights')
    shutil.copytree(source, output)
    for file, data in changes.items():
        (output / file).write_bytes(data)
    (output / 'config.yaml').write_text(config.replace('state_mode: wkv-coreml', 'state_mode: tensor'))
    require(all(sha256(output / file) == digest for file, digest in weights.items()), 'Weight copy mismatch')
    receipt = {'state_mode': 'tensor', 'weights_sha256': weights,
               'changed_files': [str(p) for p in changes] + ['config.yaml']}
    (output / 'tensor-state-repair.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output), indent=2))
