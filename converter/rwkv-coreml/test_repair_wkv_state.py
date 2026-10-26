import json
from pathlib import Path
import struct
import tempfile
import unittest

from coremltools.proto import FeatureTypes_pb2, Model_pb2
from repair_wkv_state import repair, tensor_description, tensor_mil


def fixture(shape=(24, 32, 64, 64), v_first=False):
    typ = f'tensor<fp16, {list(shape)}>'
    body = f'    func decode<ios18>({typ} in0, state<{typ}> state_wkv) {{\n'
    constants = {
        'concat_0': ('int32', [0, 0, 0, 0]), 'concat_1': ('int32', [0, 0, 0, 0]),
        'stride_0': ('int32', [1, 1, 1, 1]),
        'begin_mask_0': ('bool', '[false, true, true, true]'),
        'end_mask_0': ('bool', '[true, true, true, true]'),
        'squeeze_mask_0': ('bool', '[false, false, false, false]'),
    }
    for name, (kind, value) in constants.items():
        if not name.startswith('concat_'):
            name = 'state_wkv_internal_tensor_assign_1_' + name
        body += f'        tensor<{kind}, [4]> {name} = const()[name = string("{name}"), val = tensor<{kind}, [4]>({value})];\n'
    body += f'        {typ} read_state_0 = read_state(input = state_wkv)[name = string("read_state_0")];\n'
    body += (f'        {typ} state_wkv_internal_tensor_assign_1_cast_fp16 = slice_update('
             'begin = concat_0, begin_mask = state_wkv_internal_tensor_assign_1_begin_mask_0, '
             'end = concat_1, end_mask = state_wkv_internal_tensor_assign_1_end_mask_0, '
             'squeeze_mask = state_wkv_internal_tensor_assign_1_squeeze_mask_0, '
             'stride = state_wkv_internal_tensor_assign_1_stride_0, update = new_state, x = read_state_0)'
             '[name = string("state_wkv_internal_tensor_assign_1_cast_fp16")];\n'
             '        write_state(data = state_wkv_internal_tensor_assign_1_cast_fp16, input = state_wkv)'
             '[name = string("coreml_update_state_0_write_state")];\n')
    body += '        } -> (out0, state_tokenshift_out' + (', v_first_out' if v_first else '') + ');\n'
    return 'program(1.3)\n{\n' + body + body.replace('func decode', 'func prefill') + '}\n'


def description(shape, v_first=False):
    desc = Model_pb2.ModelDescription()
    desc.defaultFunctionName = 'decode'
    for name in ['decode', 'prefill']:
        fn = desc.functions.add(name=name)
        fn.input.add(name='in0')
        fn.input.add(name='state_tokenshift_in')
        if v_first:
            fn.input.add(name='v_first_in')
        state = fn.state.add(name='state_wkv')
        state.type.stateType.arrayType.shape.extend(shape)
        state.type.stateType.arrayType.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT16
    encoded = desc.SerializeToString()
    return struct.pack('<II', 502, 9) + bytes(67) + struct.pack('<Q', len(encoded)) + encoded + b'unchanged-tail'


class RepairTests(unittest.TestCase):
    def test_single_and_multi_chunk_shapes(self):
        for shape, first in [([24, 32, 64, 64], False), ([16, 40, 64, 64], True)]:
            mil, actual = tensor_mil(fixture(shape, first))
            self.assertEqual(actual, shape)
            self.assertNotIn('write_state(', mil)
            self.assertIn('state_wkv_out = identity(x = new_state)', mil)
            self.assertEqual(mil.count(', state_wkv_out);'), 2)
            data = tensor_description(description(shape), shape)
            size = struct.unpack_from('<Q', data, 75)[0]
            desc = Model_pb2.ModelDescription()
            desc.ParseFromString(data[83:83 + size])
            self.assertEqual(len(desc.functions[0].state), 0)
            self.assertEqual(list(desc.functions[0].input[-1].type.multiArrayType.shape), shape)
            self.assertTrue(data.endswith(b'unchanged-tail'))

    def test_reject_partial_or_unknown_graph(self):
        original = fixture()
        for bad in [original.replace('[true, true, true, true]', '[false, true, true, true]'),
                    original.replace('[0, 0, 0, 0]', '[1, 0, 0, 0]', 1),
                    original.replace('update = new_state', 'update = other, extra = true'),
                    original.replace('func prefill', 'func unknown').replace('write_state(', 'unknown_write(')]:
            with self.assertRaises(ValueError):
                tensor_mil(bad)
        with self.assertRaises(ValueError):
            tensor_description(description([24, 32, 64, 64]), [16, 40, 64, 64])

    def test_state_input_order_matches_multi_chunk_mil(self):
        shape = [16, 40, 64, 64]
        data = tensor_description(description(shape, v_first=True), shape)
        desc = Model_pb2.ModelDescription()
        desc.ParseFromString(data[83:83 + struct.unpack_from('<Q', data, 75)[0]])
        for fn in desc.functions:
            self.assertEqual([f.name for f in fn.input], ['in0', 'state_tokenshift_in', 'state_wkv_in', 'v_first_in'])

    def test_repair_preserves_source_and_weights_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'source', Path(temp) / 'output'
            chunk = source / 'g1j_chunk1of1.mlmodelc'
            (chunk / 'weights').mkdir(parents=True)
            (chunk / 'weights/weight.bin').write_bytes(b'unchanged-quantized-weights')
            (chunk / 'model.mil').write_text(fixture())
            (chunk / 'coremldata.bin').write_bytes(description([24, 32, 64, 64]))
            schema = {'stateSchema': [{'name': 'state_wkv'}], 'inputSchema': [], 'outputSchema': []}
            (chunk / 'metadata.json').write_text(json.dumps([dict(schema, functions=[schema, schema])]))
            (source / 'config.yaml').write_text('basename: g1j\nnum_chunks: 1\nstate_mode: wkv-coreml\ndecode_compute_units: cpu-ne\nprefill_compute_units: cpu-ne\n')
            before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            repair(source, output)
            self.assertTrue(all((source / p).read_bytes() == b for p, b in before.items()))
            self.assertEqual((output / chunk.relative_to(source) / 'weights/weight.bin').read_bytes(), b'unchanged-quantized-weights')
            self.assertIn('state_mode: tensor', (output / 'config.yaml').read_text())
            self.assertEqual((output / 'config.yaml').read_text().count('cpu-ne'), 2)
            with self.assertRaises(ValueError):
                repair(source, output)


if __name__ == '__main__':
    unittest.main()
