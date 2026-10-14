#pragma once

#include <algorithm>

namespace rwkvmobile {

constexpr int desktop_llama_cpu_threads(unsigned int logical_cpus) {
    return static_cast<int>(std::max(4u, logical_cpus / 2));
}

constexpr int macos_llama_cpu_threads(unsigned int logical_cpus) {
    return static_cast<int>(logical_cpus == 0 ? 8u : std::min(8u, logical_cpus));
}

} // namespace rwkvmobile
