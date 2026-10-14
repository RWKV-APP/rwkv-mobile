#include "../src/backends/llama_cpp/cpu_thread_policy.h"

using rwkvmobile::desktop_llama_cpu_threads;
using rwkvmobile::macos_llama_cpu_threads;

static_assert(desktop_llama_cpu_threads(0) == 4);
static_assert(desktop_llama_cpu_threads(1) == 4);
static_assert(desktop_llama_cpu_threads(4) == 4);
static_assert(desktop_llama_cpu_threads(7) == 4);
static_assert(desktop_llama_cpu_threads(8) == 4);
static_assert(desktop_llama_cpu_threads(9) == 4);
static_assert(desktop_llama_cpu_threads(12) == 6);
static_assert(desktop_llama_cpu_threads(24) == 12);
static_assert(desktop_llama_cpu_threads(64) == 32);
static_assert(macos_llama_cpu_threads(0) == 8);
static_assert(macos_llama_cpu_threads(4) == 4);
static_assert(macos_llama_cpu_threads(8) == 8);
static_assert(macos_llama_cpu_threads(14) == 8);

int main() {}
