#include "c_api.h"
#include "tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <sched.h>
#include <thread>
#include <exception>
#include <vector>

static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static uint64_t readFpcr() { uint64_t value; asm volatile("mrs %0, fpcr" : "=r"(value)); return value; }
static void writeFpcr(uint64_t value) { asm volatile("msr fpcr, %0" :: "r"(value) : "memory"); }
struct RestoreFpcr { uint64_t value = readFpcr(); ~RestoreFpcr() { writeFpcr(value); } };
static cpu_set_t readAffinity() {
    cpu_set_t value = {};
    check(sched_getaffinity(0, sizeof(value), &value) == 0, "read caller affinity");
    return value;
}
static void checkAffinity(const cpu_set_t & expected) {
    const auto actual = readAffinity();
    check(std::memcmp(&actual, &expected, sizeof(actual)) == 0, "caller affinity changed");
}

int main(int argc, char** argv) {
    RestoreFpcr restore;
    rwkvmobile_runtime_t runtime = nullptr;
    try {
        check(argc == 4, "usage: test_android_cpu_fpcr model.gguf tokenizer.txt prompt.txt");
        const uint64_t base = readFpcr() & ~(uint64_t(1) << 24);
        const auto incoming_affinity = readAffinity();
        writeFpcr(base);
        runtime = rwkvmobile_runtime_init();
        check(runtime, "runtime init");
        // The App requests offload even in Android builds with no GPU device.
        struct Args { int gpuLayers = 99; } args;
        const int model = rwkvmobile_runtime_load_model_with_extra(runtime, argv[1], "llama.cpp", argv[2], &args);
        check(model >= 0, "model load");
        checkAffinity(incoming_affinity);
        auto models = rwkvmobile_runtime_get_loaded_models_info(runtime);
        const int vocab = models.count == 1 && models.models ? models.models[0].vocab_size : 0;
        rwkvmobile_runtime_free_loaded_models_list(models);
        check(vocab > 1 && vocab <= 1048576, "vocabulary bounds");
        std::ifstream file(argv[3], std::ios::binary);
        check(file.good(), "prompt file");
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        rwkvmobile::trie_tokenizer tokenizer;
        check(tokenizer.load(argv[2]) == 0, "tokenizer load");
        const auto prompt = tokenizer.encode(text);
        check(!prompt.empty() && prompt.size() <= 2048 && tokenizer.decode(prompt) == text, "prompt roundtrip and bounds");
        std::vector<float> logits(vocab), reference;
        reference.reserve(size_t(vocab) * 64);
        auto run_mode = [&](int mode) {
            const auto caller_affinity = readAffinity();
            const uint64_t selected = base | (uint64_t(mode) << 24);
            writeFpcr(selected);
            check(rwkvmobile_runtime_clear_state(runtime, model) == 0, "clear state");
            check(readFpcr() == selected, "clear state changes caller FPCR");
            checkAffinity(caller_affinity);
            int token = 0;
            for (int step = 0; step < 64; ++step) {
                const int* ids = step ? &token : prompt.data();
                const int count = step ? 1 : int(prompt.size());
                check(rwkvmobile_runtime_eval_logits(runtime, model, ids, count, logits.data(), vocab) == 0, "evaluation");
                check(readFpcr() == selected, "evaluation changes caller FPCR");
                checkAffinity(caller_affinity);
                for (float value : logits) check(std::isfinite(value), "non-finite logits");
                if (!mode) reference.insert(reference.end(), logits.begin(), logits.end());
                else check(std::memcmp(reference.data() + size_t(step) * vocab, logits.data(), size_t(vocab) * sizeof(float)) == 0, "caller FPCR modes change full logits");
                token = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
        };
        run_mode(0);
        std::exception_ptr worker_error;
        std::thread other_caller([&] { try { run_mode(1); } catch (...) { worker_error = std::current_exception(); } });
        other_caller.join();
        if (worker_error) std::rethrow_exception(worker_error);
        checkAffinity(incoming_affinity);
        check(rwkvmobile_runtime_release(runtime) == 0, "runtime release");
        runtime = nullptr;
        checkAffinity(incoming_affinity);
        std::puts("ANDROID_CPU_FPCR_RESTORATION_AND_FULL_LOGITS_PASSED");
        std::puts("ANDROID_CPU_AFFINITY_CROSS_THREAD_RESTORATION_PASSED");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ANDROID_CPU_FPCR_FAILED %s\n", error.what());
        if (runtime) rwkvmobile_runtime_release(runtime);
        return 1;
    }
}
