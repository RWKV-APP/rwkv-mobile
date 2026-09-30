#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#if defined(ENABLE_PALM)
extern "C" void mollm_print_matmul_shape_profile(const char *, int);
#endif

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> [prompt_tokens=512] [decode_tokens=128] [repeats=1]" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    void *extra = std::string(argv[2]) == "qnn" ? std::getenv("RWKV_QNN_BACKEND_PATH") : nullptr;
    int model_id = runtime.load_model(argv[1], argv[2], "", extra);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    int vocab_size = runtime.get_vocab_size(model_id);
    int prompt_tokens = 512;
    if (argc >= 4) {
        prompt_tokens = std::atoi(argv[3]);
        if (prompt_tokens <= 0) {
            std::cerr << "prompt_tokens must be positive" << std::endl;
            return 1;
        }
    }

    const int decode_tokens = argc >= 5 ? std::atoi(argv[4]) : 128;
    const int repeats = argc >= 6 ? std::atoi(argv[5]) : 1;
    if (vocab_size <= 0 || decode_tokens <= 0 || repeats <= 0 ||
        prompt_tokens > 16384 || decode_tokens > 16384 || repeats > 100) {
        std::cerr << "Invalid benchmark dimensions" << std::endl;
        return 1;
    }
    std::vector<int> prompt_ids((size_t)prompt_tokens);
    std::vector<int> decode_ids((size_t)decode_tokens);
    rwkvmobile::Tensor1D logits;
    // Raw mt19937 output avoids platform-dependent uniform_int_distribution.
    std::mt19937 rng(20260930);
    uint32_t input_hash = 2166136261u;
    for (auto *ids : {&prompt_ids, &decode_ids}) {
        for (int &id : *ids) {
            id = static_cast<int>(rng() % static_cast<uint32_t>(vocab_size));
            input_hash = (input_hash ^ static_cast<uint32_t>(id)) * 16777619u;
        }
    }
    std::cout << std::setprecision(10) << "Protocol: pp=" << prompt_tokens
              << " tg=" << decode_tokens << " repeats=" << repeats
              << " seed=20260930 input_hash=" << input_hash << std::endl;
    size_t dump_index = 0;
    auto checksum = [&]() {
        if (!logits.valid() || logits.count != static_cast<size_t>(vocab_size)) return false;
        uint32_t hash = 2166136261u;
        double sum = 0;
        for (size_t i = 0; i < logits.count; ++i) {
            float value = rwkvmobile::tensor1d_get_f32(logits, i);
            if (!std::isfinite(value)) return false;
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            hash = (hash ^ bits) * 16777619u;
            sum += value;
        }
        std::cout << "Logits: count=" << logits.count << " fnv32=" << hash << " sum=" << sum << std::endl;
        if (const char *prefix = std::getenv("RWKV_BENCH_DUMP_LOGITS")) {
            std::ofstream dump(std::string(prefix) + "-" + std::to_string(dump_index++) + ".f32", std::ios::binary);
            if (!dump) return false;
            for (size_t i = 0; i < logits.count; ++i) {
                const float value = rwkvmobile::tensor1d_get_f32(logits, i);
                dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
            if (!dump) return false;
        }
        return true;
    };
    for (int run = 0; run <= repeats; ++run) {
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.clear_state(model_id), "State reset failed");
        runtime.reset_inference_speed_stats(model_id);
        const auto begin = std::chrono::steady_clock::now();
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, prompt_ids, logits), "Prefill failed");
        const auto prefill_end = std::chrono::steady_clock::now();
        if (!checksum()) return 1;
        const auto decode_begin = std::chrono::steady_clock::now();
        for (int id : decode_ids) {
            ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, id, logits), "Decode failed");
        }
        const auto end = std::chrono::steady_clock::now();
        if (!checksum()) return 1;
        const double pp_seconds = std::chrono::duration<double>(prefill_end - begin).count();
        const double tg_seconds = std::chrono::duration<double>(end - decode_begin).count();
        std::cout << "RESULT run=" << run << " warmup=" << (run == 0)
                  << " pp_tokens=" << prompt_tokens << " tg_tokens=" << decode_tokens
                  << " pp_seconds=" << pp_seconds << " tg_seconds=" << tg_seconds
                  << " prefill=" << prompt_tokens / pp_seconds
                  << " decode=" << decode_tokens / tg_seconds << std::endl;
    }

    runtime.release();
#if defined(ENABLE_PALM)
    if (std::getenv("MOLLM_MATMUL_SHAPE_PROFILE")) mollm_print_matmul_shape_profile("headless",20);
#endif

    return 0;
}
