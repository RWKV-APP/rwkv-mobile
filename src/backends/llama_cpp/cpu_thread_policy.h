#pragma once

#include <algorithm>
#include <utility>
#include <vector>

namespace rwkvmobile {

constexpr bool desktop_llama_cpu_only(int requested_gpu_layers, bool has_gpu_devices) {
    return requested_gpu_layers == 0 || !has_gpu_devices;
}

constexpr int desktop_llama_cpu_threads(unsigned int logical_cpus) {
    return static_cast<int>(std::max(4u, logical_cpus / 2));
}

constexpr int macos_llama_cpu_threads(unsigned int logical_cpus) {
    return static_cast<int>(logical_cpus == 0 ? 8u : std::min(8u, logical_cpus));
}

constexpr int android_llama_cpu_threads(unsigned int performance_cpus, unsigned int uid) {
    // Leave room for App UI work; shell and system diagnostics keep their full budget.
    const bool app_process = uid % 100000u >= 10000u;
    return static_cast<int>(app_process ? std::min(4u, performance_cpus) : performance_cpus);
}

inline std::vector<int> android_llama_performance_cpus(
    const std::vector<std::pair<int, unsigned int>> & capacities) {
    unsigned int maximum = 0;
    for (const auto & cpu : capacities) {
        if (cpu.first < 0 || cpu.second == 0) return {};
        maximum = std::max(maximum, cpu.second);
    }
    std::vector<int> selected;
    for (const auto & cpu : capacities) {
        if (cpu.second >= maximum / 2 + maximum % 2) selected.push_back(cpu.first);
    }
    return selected;
}

} // namespace rwkvmobile
