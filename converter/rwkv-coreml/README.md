# Core ML WKV state compatibility

Use `--state-mode tensor --decode-compute-units cpu-ne --prefill-compute-units cpu-ne`
for new direct INT4/LUT6 exports. Tensor state is the default for
`convert_rwkv_to_coreml_int4_import.py`. The existing native tensor-state path
feeds WKV and TokenShift outputs into the next prediction; the model still
computes on the Neural Engine. State storage and compute device are separate.

On M4 Pro / macOS 27.0.1 (26A434), the released G1j 1.5B and 2.9B first-chunk
`wkv-coreml` graphs fail ANE compilation; Core ML eventually reports a misleading
`functionName` error. Replacing the full `slice_update` with an identity while
keeping `write_state` still fails. Removing the write allows NE compilation.
Supplying embeddings externally while retaining the full mutable WKV write also
works, as does the original 2.9B second chunk with its full write intact. These
controls isolate a compatibility problem in this first-chunk embedding/mutable
state combination; they do not establish that full state writes or arbitrary
CPU/NE partitions are universally unsupported.

Exposing the same WKV update as tensor I/O preserves embedding and weight bytes
and avoids that combination. Both repaired sizes completed 16 decode steps and
four 16-token prefill calls through the native loader on this Mac. Their logits,
WKV and TokenShift outputs matched independent Core ML probes byte for byte;
reset, save/restore and prefill-to-decode checks passed. Every decode/prefill
compute plan assigned operators to NE and none to GPU. This does not establish
iPhone acceptance or identify an Apple compiler source-level defect.

For existing compiled G1j `wkv-coreml` artifacts, avoid another quantization:

```sh
python repair_wkv_state.py /path/to/original-g1j /path/to/new-g1j-tensorstate
python -m unittest test_repair_wkv_state.py
```

The source directory must contain `config.yaml` and all configured `.mlmodelc`
chunks. `coremltools` provides the protobuf schema; no PyTorch checkpoint is
needed. The tool validates both functions and a full-state update, copies into
a new directory, adjusts MIL and feature schemas, and changes only
`state_mode` in the runtime config. Compute-unit settings and weight bytes are
preserved. It rejects partial writes and unknown compiled-description layouts.
Compiled Core ML internals are version-specific; re-export unsupported layouts
with the converter instead. Do not overwrite an installed model or reuse its
download hash when distributing the repaired artifact.

`examples/test_coreml_tensor_state.mm` checks the real native loader, sequential
predictions, and state reset. It asserts CPU+NE for every decode/prefill model,
so the loader's GPU fallback cannot pass this test. Inspect `MLComputePlan` on
the target device as well: a configuration permits a device but does not prove
operator assignment. NE/GPU floating-point results need separate numerical
acceptance; same-device graph equivalence does not imply cross-device equality.
