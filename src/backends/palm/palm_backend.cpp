#include "palm_backend.h"

#include "engine/engine.h"
#include "logger.h"

#include <cerrno>
#include <climits>
#include <cstdlib>

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace rwkvmobile {
namespace {

bool metadata_int(const std::unordered_map<std::string, std::string>& metadata,
                  const char* key, int& value) {
    const auto it = metadata.find(key);
    if (it == metadata.end()) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(it->second.c_str(), &end, 10);
    if (errno || !end || *end != '\0' || parsed <= 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

} // namespace

palm_backend::palm_backend() = default;
palm_backend::~palm_backend() = default;

bool palm_backend::is_supported() {
#if defined(__aarch64__) && defined(__linux__)
    // Match Palm's armv8.2-a+dotprod+fp16+fp16fml baseline; i8mm is optional.
    const unsigned long required = HWCAP_ASIMDDP | HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_ASIMDFHM;
    return (getauxval(AT_HWCAP) & required) == required;
#elif defined(__aarch64__) && defined(__APPLE__)
    for (const char* feature : {"hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_FP16", "hw.optional.arm.FEAT_FHM"}) {
        int available = 0;
        size_t size = sizeof(available);
        if (sysctlbyname(feature, &available, &size, nullptr, 0) != 0 || !available) return false;
    }
    return true;
#else
    return true;
#endif
}

int palm_backend::load_model(std::string model_path, void*) {
    if (!is_supported()) {
        LOGE("Palm requires ARM dot-product and FP16 arithmetic support");
        return RWKV_ERROR_UNSUPPORTED;
    }
    const size_t separator = model_path.find_last_of("/\\");
#if !defined(_WIN32)
    if (separator != std::string::npos) {
        const std::string temp_directory = model_path.substr(0, separator);
        setenv("TMPDIR", temp_directory.c_str(), 1);
    }
#endif

    auto next = std::make_unique<LLMEngine>();
    EngineConfig config;
    config.package_path = std::move(model_path);
    config.weight_loading = WeightLoadingMode::MMAP;
    if (!next->load(config)) {
        LOGE("Palm failed to load .mollm package");
        return RWKV_ERROR_MODEL;
    }

    const auto& metadata = next->package_metadata();
    if (!metadata_int(metadata, "num_layers", n_layers) ||
        !metadata_int(metadata, "num_heads", num_heads) ||
        !metadata_int(metadata, "hidden_size", hidden_size) ||
        !metadata_int(metadata, "vocab_size", vocab_size)) {
        LOGE("Palm package is missing required model metadata");
        return RWKV_ERROR_MODEL;
    }

    version = 7;
    engine = std::move(next);
    token_history.clear();
    logits_storage.clear();
    return RWKV_SUCCESS;
}

int palm_backend::update_logits(const Tensor& hidden, int token_count, Tensor1D& logits) {
    if (!hidden.data || !engine) return RWKV_ERROR_EVAL;
    logits_storage = engine->run_lmhead_raw(hidden, token_count);
    if (logits_storage.size() != static_cast<size_t>(vocab_size)) {
        logits = {};
        return RWKV_ERROR_EVAL;
    }
    logits = Tensor1D::make(logits_storage.data(), TensorDType::F32, logits_storage.size());
    return RWKV_SUCCESS;
}

int palm_backend::eval(int id, Tensor1D& logits) {
    if (!engine) return RWKV_ERROR_BACKEND;
    const Tensor hidden = engine->decode_hidden(id);
    const int ret = update_logits(hidden, 1, logits);
    if (ret == RWKV_SUCCESS) token_history.push_back(id);
    return ret;
}

int palm_backend::eval(std::vector<int> ids, Tensor1D& logits) {
    if (!engine || ids.empty()) return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    const Tensor hidden = engine->prefill_hidden(ids);
    const int ret = update_logits(hidden, static_cast<int>(ids.size()), logits);
    if (ret == RWKV_SUCCESS) token_history.insert(token_history.end(), ids.begin(), ids.end());
    return ret;
}

int palm_backend::get_state(std::any& state) {
    state = token_history;
    return RWKV_SUCCESS;
}

int palm_backend::set_state(std::any state) {
    if (!engine) return RWKV_ERROR_BACKEND;
    if (!state.has_value()) return zero_state();
    std::vector<int> restored;
    try {
        restored = std::any_cast<std::vector<int>>(state);
    } catch (const std::bad_any_cast&) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    engine->reset();
    token_history.clear();
    if (!restored.empty() && !engine->prefill_hidden(restored).data) return RWKV_ERROR_EVAL;
    // ponytail: replay is O(history); replace with native Palm state snapshots if cache restores become measurable.
    token_history = std::move(restored);
    return RWKV_SUCCESS;
}

int palm_backend::free_state(std::any state) {
    state.reset();
    return RWKV_SUCCESS;
}

int palm_backend::zero_state() {
    if (!engine) return RWKV_ERROR_BACKEND;
    engine->reset();
    token_history.clear();
    return RWKV_SUCCESS;
}

int palm_backend::release_model() {
    engine.reset();
    token_history.clear();
    logits_storage.clear();
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile
