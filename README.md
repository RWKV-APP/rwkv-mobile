# rwkv-mobile

An inference runtime with multiple backends supported.

This RWKV-APP fork maintains the native libraries consumed by RWKV Chat. Upstream
is [MollySophia/rwkv-mobile](https://github.com/MollySophia/rwkv-mobile).

## Goal:

- Easy integration on different platforms using flutter or native cpp, including mobile devices.
- Support inference using different hardware like Qualcomm Hexagon NPU, or general CPU/GPU.
- Provide easy-to-use C apis
- Provide an api server compatible with AI00_server(openai api)

## Supported or planned backends:

- [x] WebRWKV (WebGPU): Compatible with most PC graphics cards, as well as macOS Metal. Doesn't work on Qualcomm's proprietary Adreno GPU driver though.
- [x] llama.cpp: Run on Android devices with CPU inference.
- [x] ncnn: Initial support for rwkv v6/v7 unquantized models (suitable for running tiny models everywhere).
- [x] Qualcomm Hexagon NPU: Based on Qualcomm's QNN SDK 2.42.0.
- [x] MLX: Running RWKV on Apple Silicon devices using Apple's MLX framework.
- [x] MediaTek Neuropilot7: Running RWKV on Dimensity 9300 NPU.
- [x] MediaTek Neuropilot9: Running RWKV on Dimensity 9500 NPU through Neuron Adapter/uSDK.
- [x] CoreML: Running RWKV with Apple's CoreML framework on CPU, GPU, or Apple Neural Engine.
- [x] Palm-Infra: CPU inference from `.mollm` packages, enabled with `-DENABLE_PALM_BACKEND=ON`.
- [ ] To be continued...

## How to build:

- Install [rust](https://www.rust-lang.org/tools/install) and [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) (for building the web-rwkv backend)
- Check [.github/workflows/build.yml](.github/workflows/build.yml).

Palm builds fetch the commit in `PALM_INFRA_GIT_TAG`; `PALM_INFRA_DIR` is an
explicit local development override. Linux and Android ARM64 expose Palm only
when the CPU supports dot-product and FP16 arithmetic. i8mm remains optional.
Windows packages include the OpenMP DLL imported by the native library. The
desktop CI loads the packaged library and checks the public backend enumeration;
application and model acceptance are separate release checks.

### Build for Android:
- Install Android NDK r25c (recommended version) (Download from https://dl.google.com/android/repository/android-ndk-r25c-linux.zip)
- Install Ninja
```
git clone https://github.com/MollySophia/rwkv-mobile.git
cd rwkv-mobile
mkdir build && cd build
cmake .. -DENABLE_NCNN_BACKEND=ON -DENABLE_WEBRWKV_BACKEND=ON -DENABLE_QNN_BACKEND=ON -DENABLE_MTK_NP7_BACKEND=ON \
    -DENABLE_TTS=ON -DENABLE_VISION=ON -DENABLE_WHISPER=ON -DBUILD_EXAMPLES=ON -DENABLE_SERVER=ON \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=$HOME/android-ndk-r25c \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/android-ndk-r25c/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -G Ninja
ninja
```

For Dimensity 9500/NP9 builds, use `-DENABLE_MTK_NP9_BACKEND=ON`. NP7 and NP9 can be enabled together; the MediaTek runtime archives are packaged as separate `librwkv_mtk_np7.so` and `librwkv_mtk_np9.so` libraries and loaded with `dlopen`. See [MTK_NP9_SUPPORT.md](MTK_NP9_SUPPORT.md) for the current NP9 integration notes.

## Prepare MLX model packages

After MLX conversion and before creating the model ZIP, normalize and check
per-layer quantization metadata:

```sh
python3 converter/normalize_mlx_config.py /path/to/model/config.json
python3 converter/normalize_mlx_config.py /path/to/model/config.json --check
```

The Swift MLX loader requires each quantized layer's `group_size` explicitly.
The tool fills omitted values from that section's global `group_size` in both
`quantization` and `quantization_config`, preserving explicit overrides, bits,
and tensor files. For RWKV7 it also adds `down_proj` / `up_proj` config aliases
for Python's `lora.0` / `lora.2` names, matching the Swift loader's tensor-name
sanitization while retaining the Python keys. Conflicting aliases fail the check.
It does not re-quantize weights. Package the resulting config
with the unchanged weights, record the new ZIP digest, and verify loading and
generation in the consuming App. This metadata check alone is not runtime
acceptance.

## Core ML loading on macOS 27

Some multifunction ML Programs fail to load with ANE enabled on macOS 27,
reporting that `functionName` must be nil unless the model is an ML Program.
For this specific Core ML error, the runtime retries the same model and selected
function with `cpu-gpu`. It preserves the weights, quantization, and `config.yaml`.
The warning and successful-load logs identify the effective compute units;
this compatibility path does not use ANE. Other loading errors retain the normal
retry behavior, and this workaround does not apply to iOS or earlier macOS versions.

For source exports using `converter/rwkv-coreml/convert_rwkv_to_coreml_int4_import.py`,
`--state-mode tensor` exposes recurrent state as model inputs and outputs.
The optional `--att-output-scale 64` multiplies attention-output activations
before projection and divides the result afterward, retaining the selected
INT4/LUT6 weight recipe. This reduced CPU/NE numerical divergence for the
tested 1.5B model on macOS 27. It defaults to `1`: validate finite intermediate
ranges, fixed-token recurrence, prefill, and compute-unit performance for each
model before enabling it. Re-exporting is not a byte-identical repair of an
existing quantized package, and these options do not update downloaded models.

## TODO:
- [ ] Better tensor abstraction for different backends
- [ ] Batch inference for all backends
