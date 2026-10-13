# Experimental RWKV7 Core AI backend

This converter exports `decode` and fixed-length `prefill` functions sharing
mutable `tokenshift` and `wkv` state. The native `coreai` backend retains those
arrays between inference calls and copies them only for runtime checkpoints.
It currently supports a single sequence on Apple Silicon macOS 27 with Xcode 27.
The CMake option `ENABLE_COREAI_BACKEND` is off by default. GPU is a compute
preference; it is not proof of actual placement.

Use Python 3.12 with `torch==2.11.0`, `coreai-torch==0.4.2`, and
`coreai-core==1.0.0b2`, and `coreai-opt==0.2.1`. Install Xcode's Metal toolchain before specialization:

```sh
xcodebuild -downloadComponent MetalToolchain
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-directory
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-w8 --quantization w8 --group-size 32
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-w8-mixed --quantization w8 --mixed-matmuls
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-lut6-w8 --quantization lut6-w8 --group-size 128 --mixed-matmuls
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-w4-w8 --quantization w4-w8 --mixed-matmuls
python converter/rwkv-coreai/convert_rwkv_to_coreai.py model.pth output-lut6-ffn-w8 --quantization lut6-w8 --group-size 128 --lut-weight-threshold 8388608 --int8-large-threshold 67108864 --mixed-matmuls
```

The output directory contains `config.json` and the `rwkv.aimodel` asset
directory. Keep generated weights outside the source tree. Metadata records
the source SHA-256, tensor shapes, precision and prefill length. Existing
output directories are never overwritten.

FP32 is the conversion reference default. `--quantization w8` or `w4` applies
Core AI's symmetric per-block integer compression to eligible matrix weights,
including the embedding and output head. Activations, normalization reductions,
and recurrent state remain FP32 (W8A32/S32 or W4A32/S32). This is weight-only
compression, not integer activation/state execution. Inspect the produced
asset and validate its outputs before use; compression may skip incompatible
tensor shapes. These recipes are not byte-equivalent to MLX affine W6 or
Core ML's INT4/LUT6 mixture. W4 requires separate numerical evaluation.

`--mixed-matmuls` uses FP16 inputs and results inside compressed matrix
multiplications, then converts their results to FP32. Normalization, residuals,
recurrent updates and persistent state remain FP32. The manifest labels this
W8 (or W4) with mixed FP16/FP32 activations and FP32 state. It is a separate
candidate requiring both PyTorch and OS Core AI numerical checks; it must not
be labeled as full integer computation or compared as identical to MLX W6.

`--quantization lut6` uses scalar K-means lookup tables with six-bit indices.
Its group size counts output channels, unlike integer quantization's input
channel blocks. `lut6-w8` applies LUT6 to eligible matrices with more than
1,048,576 elements, then compresses remaining eligible matrices to symmetric
INT8 with input group size 32. This preserves higher precision in small
low-rank projections. These are LUT6 / LUT6+W8 recipes, not linear INT6.
Both support mixed matmuls and keep FP32 state. `compression.json` records
the actual compression operations and constant types for structural review;
unsupported shapes may remain uncompressed and need artifact-specific auditing.

For `lut6-w8`, `--lut-weight-threshold` changes the LUT eligibility threshold;
`--int8-large-threshold` first compresses larger matrices to INT8. The last
command above therefore keeps vocabulary matrices in INT8, applies LUT6 to
the 1.5B model's FFN matrices, and uses INT8 for its other eligible matrices.
`w4-w8` uses the same fixed 8,388,608 / 67,108,864 element thresholds with
INT4 for the middle-size matrices and symmetric INT8 group32 elsewhere.
These are explicit size-based recipes; other model shapes require checking
which matrices fall into each range. Reducing size trades away numerical
precision and does not guarantee faster inference.

For a calibrated W4/W8 recipe, add `--quantization-recipe recipe.json` to
`--quantization w4-w8 --mixed-matmuls`. The JSON contains `format_version: 1`,
the exact `source_sha256`, `group_size`, and a `protected_modules` list such as
`["blocks.23.ffn.key", "blocks.23.ffn.value"]`. Names refer to the stateful
PyTorch model. These selected matrices are compressed to W8 before the usual
size-based passes. The converter binds names to exact weight bytes, checks
that the exported graph contains every selected matrix, and records graph
match counts and the recipe digest. Unknown or duplicate names fail. Select
the list using calibration inputs, then evaluate it on separate held-out
inputs; no protected-layer list is universally optimal.

`recipes/g1j-1.5b-sensitive8-g32.json` supplies one source-bound candidate for
G1j 1.5B (20260831). It protects eight of the 48 FFN matrices and leaves the
other 40 in W4. Use it with `--quantization w4-w8 --group-size 32
--mixed-matmuls --quantization-recipe recipes/g1j-1.5b-sensitive8-g32.json`.
The list was selected on 353 calibration tokens; a separate 1,559-token
evaluation (including a 1,024-token sequence) reduced mean checkpoint logits
relative L2 from about 4.81% to 2.80%, with original-model top-1 agreement
increasing from 92/100 to 96/100. These numerical comparisons do not establish
downstream task accuracy or suitability for other checkpoints.

`recipes/g1j-2.9b-sensitive16-g32.json` is separately bound to G1j 2.9B
(20260831). Its 32 layers have 40 heads of width 64. The size thresholds
select the 2560-by-10240 FFN matrices for W4; this recipe retains 16 of the
64 FFN matrices in W8 and leaves 48 in W4. Vocabulary, attention and
low-rank projections use W8 group32. The selection ranks activation-weighted
projection error on 236 calibration tokens, then chooses the lowest
calibration KL among protection counts 8, 16 and 24. The calibration mean
logits relative L2 drops from 6.60% to 2.77% compared with the base W4/W8
recipe. This is a numerical measurement, not an answer-quality score.

Use the 2.9B recipe with `--quantization w4-w8 --group-size 32
--mixed-matmuls --fused-wkv --fused-decode --prefill-length 64
--prefill-tails 32 16 --quantization-recipe
recipes/g1j-2.9b-sensitive16-g32.json`. On macOS 27, the resulting asset was
checked against independently reconstructed identical quantized weights:
44 comparisons across separate 61-, 92- and 512-token inputs and explicit
entrypoint/tail cases kept maximum logits relative L2 below 0.18%, WKV state
below 0.08%, and token-shift state below 0.18%. The asset is about 2.51 GB;
its quantization differs from the distributed MLX W6 and Core ML INT4/LUT6
packages. These observations do not extend to other checkpoints or devices.

`--fused-wkv` optionally embeds an FP32 Metal kernel for prefill recurrence
with 64-element heads. A SIMD group retains each state row throughout the
chunk and computes its reductions with `simd_sum`. By default, decode keeps
the original graph. The converter records the implementation in `config.json`; existing
Core AI loaders can execute the embedded kernel without a native ABI change.
Both numerical agreement over repeated state updates and full-model/App
performance must be checked before adopting this experimental option. Small
kernel timing improvements alone do not establish end-to-end gains.

`--fused-decode` extends the opt-in WKV kernel to the decode graph and requires
`--fused-wkv`. Compare it using the full model; isolated one-step kernel timing
can hide savings from removing multiple recurrent operations per layer.

For larger prefill blocks, `--prefill-length 64 --prefill-tails 32 16` exports
`prefill`, `prefill_32`, and `prefill_16`, alongside `decode`. The native loader
reads optional `prefill_lengths` and selects the largest function that fits
remaining input, without padding tokens or changing state semantics. Existing
single-prefill packages remain supported. Compare both aligned inputs and
partial chunks, including snapshot/restore, before use. Changing the primary
chunk size alone can slow short prompts when the remainder falls back to decode.

For short remainders, use `--prefill-length 64 --prefill-tails 32 16 8 4 2`.
A 15-token remainder then runs as 8+4+2+1 instead of 15 decode calls. To try
larger blocks while preserving those tail paths, use `--prefill-length 128
--prefill-tails 64 32 16 8 4 2`. These options change the exported execution
shapes, not the selected quantization recipe or persistent state precision.
Compare fixed-token logits and both state arrays across the block boundary
and after subsequent decode calls. Measure short and long inputs separately;
larger blocks can improve prefill without improving decode. Host diagnostic
timings do not establish performance in the chat App.

Run `python -m unittest test_quantization_recipe test_numerical_norms` from
this directory for the focused conversion regression checks. These use the
pinned Python/Core AI dependencies and do not require model weights.

LUT conversion uses one K-means worker and disables fast decimal rounding.
With the pinned coreai-opt 0.2.1, install `coremltools==9.0`: its bundled
weighted K-means implementation supplies the `weights` argument absent from
public kmeans1d 0.5.0. The converter temporarily supplies this compatible
implementation only during palettization and restores the library afterwards.
Inspect both numerical drift against the original model and agreement with
the compressed reference; a six-bit storage label alone does not establish
accuracy, speed, or equivalence to another backend's quantization.

The shared PyTorch normalization helpers accumulate squares in FP32 before
casting back: half-precision squaring can overflow for finite attention
outliers and collapse normalized heads. `test_numerical_norms.py` covers this
failure, tiny/zero key vectors, and unchanged FP32 behavior. The legacy
`--precision float16` option remains unvalidated as a full model candidate;
it cannot be combined with these quantization experiments. Export success
alone does not establish numerical equivalence or App performance.

Build the native library with `-DENABLE_COREAI_BACKEND=ON`; existing MLX and
Core ML options may remain enabled. The Swift bridge validates the state
descriptors, supports FP16 and FP32 state snapshots, and rejects nonfinite
logits. Only the final token's logits cross the C ABI during prefill. Calls
must be serialized by the runtime and made off the UI thread.

In the corresponding RWKV Chat macOS Debug integration, use the normal local
folder picker to select the model directory. Model visibility also requires
the native library to report `coreai`. No online catalog entry is required.
Compare conversion numerics and checkpoint behavior before App evaluation;
compare actual chat performance only with explicit model, precision,
quantization, device and thermal conditions.
