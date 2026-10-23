#include "../src/backends/llama_cpp/cpu_thread_policy.h"
#include <cstdio>

using rwkvmobile::desktop_llama_cpu_threads;
using rwkvmobile::macos_llama_cpu_threads;

static_assert(rwkvmobile::desktop_llama_cpu_only(0, false));
static_assert(rwkvmobile::desktop_llama_cpu_only(0, true));
static_assert(rwkvmobile::desktop_llama_cpu_only(99, false));
static_assert(!rwkvmobile::desktop_llama_cpu_only(99, true));

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
static_assert(rwkvmobile::android_llama_cpu_threads(0, 10862) == 0);
static_assert(rwkvmobile::android_llama_cpu_threads(1, 10862) == 1);
static_assert(rwkvmobile::android_llama_cpu_threads(2, 10862) == 2);
static_assert(rwkvmobile::android_llama_cpu_threads(4, 10862) == 4);
static_assert(rwkvmobile::android_llama_cpu_threads(6, 10862) == 4);
static_assert(rwkvmobile::android_llama_cpu_threads(8, 10862) == 4);
static_assert(rwkvmobile::android_llama_cpu_threads(6, 110862) == 4);
static_assert(rwkvmobile::android_llama_cpu_threads(6, 99000) == 4);
static_assert(rwkvmobile::android_llama_cpu_threads(6, 2000) == 6);
static_assert(rwkvmobile::android_llama_cpu_threads(6, 102000) == 6);
static_assert(rwkvmobile::android_llama_cpu_threads(8, 0) == 8);
static_assert(rwkvmobile::android_llama_cpu_threads(8, 1000) == 8);

int main() {
    using rwkvmobile::android_llama_performance_cpus;
    using CpuList = std::vector<int>;
    if (android_llama_performance_cpus({{0,379},{1,379},{2,923},{3,923},{4,923},{5,867},{6,867},{7,1024}}) != CpuList({2,3,4,5,6,7})) return 1;
    if (android_llama_performance_cpus({{0,640},{1,640},{2,640},{3,640},{4,1024},{5,1024},{6,1024},{7,1024}}) != CpuList({0,1,2,3,4,5,6,7})) return 1;
    if (android_llama_performance_cpus({{2,512},{5,512}}) != CpuList({2,5})) return 1;
    if (android_llama_performance_cpus({{0,379},{1,379}}) != CpuList({0,1})) return 1;
    if (!android_llama_performance_cpus({{0,0},{1,1024}}).empty()) return 1;
    if (!android_llama_performance_cpus({}).empty()) return 1;
    std::puts("CPU_THREAD_POLICY_TESTS_PASSED");
}
