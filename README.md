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

- [x] WebRWKV (WebGPU): Compatible with most PC graphics cards, macOS Metal, and tested Android Adreno/Mali GPUs. Driver support remains device specific.
- [x] llama.cpp: Run on Android devices with CPU inference.
- [x] ncnn: Initial support for rwkv v6/v7 unquantized models (suitable for running tiny models everywhere).
- [x] Qualcomm Hexagon NPU: Based on Qualcomm's QNN SDK 2.42.0.
- [x] MLX: Running RWKV on Apple Silicon devices using Apple's MLX framework.
- [x] MediaTek Neuropilot7: Running RWKV on Dimensity 9300 NPU.
- [x] MediaTek Neuropilot9: Running RWKV on Dimensity 9500 NPU through Neuron Adapter/uSDK.
- [x] CoreML: Running RWKV with Apple Neural Engine. Based on Apple's CoreML framework.
- [x] Palm-Infra: CPU inference from `.mollm` packages, enabled with `-DENABLE_PALM_BACKEND=ON`.
- [ ] To be continued...

## QNN model and Runtime lifetimes

QNN shares one process-owned SDK session across model and Runtime lifetimes.
Model release frees model contexts, tensor metadata, RPC allocations and package
mappings, and removes the active power vote. Reusing the idle session also avoids
SDK 2.42 custom-op file/dmabuf descriptor growth when Runtime objects are recreated.
Cleanup is registered after SDK initialization so it runs before vendor static
destructors at orderly process/library shutdown. Release Runtime handles before
unloading the library. An idle session is replaced when its SDK path changes.
The idle SDK baseline is retained ownership, not zero total process descriptors.
Run `test_qnn_runtime_cycles` with model, SDK and tokenizer paths to check three
Runtime lifetimes in one process, including exact released FD target counts.

If QNN context release fails, the model and Runtime release APIs return an error
and retain the buffer owners for a release retry. A partially released model
rejects inference and state access. Keep the Runtime handle until release succeeds.

The device regression in `examples/test_qnn_runtime_lifecycle.cpp` takes a model,
SDK path, tokenizer and `ownership`, `model` or `runtime` mode. It checks temporary
inventory queries, multiple Runtime owners and real SDK context-release failure
and retry. Run it against the same native image and QNN headers.

## Android ARM64 CPU underflow handling

The RWKV7 F32 CPU operator flushes subnormal arithmetic to zero locally on each
Android ARM64 worker. Decayed recurrent state can otherwise trigger severe
denormal slow paths. The operator restores the incoming FPCR and preserves its
other bits, including FP16 handling; other operators and platforms are unchanged.
This can zero extremely small recurrent-state values earlier. Fixed natural-text
and random-token Q6 checks retain bit-identical full logits and generated IDs.

With `-DBUILD_EXAMPLES=ON`, build `test_android_cpu_fpcr` and
`test_qnn_runtime_lifecycle` for their enabled backends. The CPU test takes a
GGUF model, tokenizer and UTF-8 prompt file. It checks complete 64-step logits
across both incoming FZ modes and verifies caller FPCR preservation after every
evaluation. The QNN test arguments and SDK requirements are described above.

## CPU thread defaults

CPU-only llama.cpp on macOS uses eight compute threads, bounded by the reported
logical CPU count. GPU offload and iOS keep their existing defaults. Android
RWKV7 CPU inference reads capacities of the caller's allowed CPUs and selects
those with at least half the maximum capacity. App processes use up to four
workers on this set, leaving scheduler room for the UI; shell and system
diagnostics use the full selected worker count. Process classification uses
the [AOSP per-user App UID range](https://android.googlesource.com/platform/system/core/+/484acce026c14c8303dee845b8021d5a11e7d1a7/libcutils/include/private/android_filesystem_config.h).
Unknown capacities retain the
existing default; homogeneous and all-performance-core devices keep their
allowed CPUs. Explicit `RWKV_LLAMACPP_THREADS` or `RWKV_LLAMACPP_CPUSET` profiles
take precedence. Automatic placement covers the actual decode caller, including
a different Dart worker, and restores its incoming affinity after each call.
Build `test_cpu_thread_policy` for topology cases and `test_android_cpu_fpcr`
for real-model caller-state preservation. Android model-release logs include
a process FD count after releasing the model; this includes unrelated App and
the retained process-owned SDK descriptors, rather than asserting a zero baseline.

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

## Headless engine measurements

Build with `-DBUILD_EXAMPLES=ON`, then run:

```sh
simple_benchmark model_file backend 128 64 3
```

This resets recurrent state for one warmup and three measured runs, uses fixed
token IDs, and reports wall-clock prefill/decode tokens per second. Checksums
cover finite logits across the entire vocabulary. Sampling and App/UI overhead
are excluded; these measurements do not establish App performance acceptance.
For QNN, set `RWKV_QNN_BACKEND_PATH` to the absolute `libQnnHtp.so` path.

Windows/Linux x64 CPU inference with llama.cpp defaults to half the logical
CPU count (rounded down), with a minimum of four compute threads. Desktop
CPU fallback uses this budget even when GPU layers were requested but no GPU
device is available. Actual GPU offload keeps its existing thread defaults.
Desktop
ARM64 retains its cap of 12 available compute threads. Desktop Palm keeps its
upstream topology-derived default (at most eight compute threads), except on
Windows. Windows Palm uses N workers when N <= 8, otherwise max(8, N - 2),
where N is the total reported logical CPU count; unknown N falls back to four.
Worker counts do not set affinity or reserve exclusive physical cores.
`RWKV_LLAMACPP_THREADS` and `RWKV_PALM_THREADS` override the respective thread
counts. GPU offload keeps its existing defaults. Palm uses four threads on
Android; Android RWKV7 llama.cpp uses the capacity-based defaults above. Other
Android llama.cpp models retain the online CPU count.
Choose mobile overrides from repeated measurements on the intended device.

Compare a frozen baseline executable with a candidate using
`python examples/test_prefill_equivalence.py BASELINE CANDIDATE MODEL BACKEND`.
The check covers 1, 2, 17 and 128 input tokens and the next decode step.

For optional numerical diagnostics, `RWKV_BENCH_DUMP_LOGITS=PREFIX` writes
`PREFIX-0.f32`, `PREFIX-1.f32`, and subsequent full-vocabulary float32 outputs
outside the timed regions.

WebRWKV uses pinned dependency patches for RWKV7 output heads that exceed a
device's storage-binding limit. It splits the existing FP16 head into row
bindings and uses 32-token prefill chunks on those devices. Weight files and
quantization remain unchanged. NF4 decode reads its quantization table directly
from uniform storage on Adreno and Mali; other GPU vendors keep the shared-table
path. Fixed 2.9B measurements and full-vocabulary parity checks cover Adreno 750
and Mali-G1-Ultra; these native checks do not establish App acceptance.

## TODO:
- [ ] Better tensor abstraction for different backends
- [ ] Batch inference for all backends
