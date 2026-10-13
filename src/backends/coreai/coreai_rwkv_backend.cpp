#include "coreai_rwkv_backend.h"
#include "json.hpp"
#include "logger.h"
#include <fstream>
#include <limits>

extern "C" {
int32_t rwkv_coreai_available();
void *rwkv_coreai_load(const char *path);
void rwkv_coreai_free(void *context);
int32_t rwkv_coreai_eval(void *context, const int32_t *tokens, int32_t count, float *logits, int32_t capacity);
int64_t rwkv_coreai_state_bytes(void *context);
int32_t rwkv_coreai_read_state(void *context, void *output, int64_t bytes);
int32_t rwkv_coreai_write_state(void *context, const void *input, int64_t bytes);
}

namespace rwkvmobile {

bool coreai_rwkv_backend::is_supported() { return rwkv_coreai_available() != 0; }
bool coreai_rwkv_backend::is_available() { return is_supported(); }

int coreai_rwkv_backend::init(void *) {
    return is_supported() ? RWKV_SUCCESS : RWKV_ERROR_UNSUPPORTED;
}

int coreai_rwkv_backend::load_model(std::string path, void *) {
    release_model();
    progress.store(0);
    try {
        std::ifstream file(path + "/config.json");
        auto config = nlohmann::json::parse(file);
        if (config.at("format_version") != 1 || config.at("backend") != "coreai" ||
            (config.at("precision") != "float16" && config.at("precision") != "float32")) return RWKV_ERROR_MODEL;
        n_layers = config.at("n_layers").get<int>();
        hidden_size = config.at("hidden_size").get<int>();
        num_heads = config.at("num_heads").get<int>();
        vocab_size = config.at("vocab_size").get<int>();
        version = config.at("model_version").get<int>();
        if (n_layers <= 0 || hidden_size <= 0 || num_heads <= 0 || vocab_size <= 0 ||
            vocab_size > 1000000 || version != 7) return RWKV_ERROR_MODEL;
    } catch (const std::exception &error) {
        LOGE("Invalid Core AI model config: %s\n", error.what());
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    context = rwkv_coreai_load(path.c_str());
    if (!context) return RWKV_ERROR_MODEL;
    state_bytes = rwkv_coreai_state_bytes(context);
    if (state_bytes <= 0) {
        release_model();
        return RWKV_ERROR_MODEL;
    }
    output.resize(vocab_size);
    progress.store(1);
    LOGI("Core AI model loaded; GPU preferred, actual compute placement requires profiling\n");
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::eval(int id, Tensor1D &logits) {
    if (!context) return RWKV_ERROR_EVAL;
    int32_t token = id;
    if (rwkv_coreai_eval(context, &token, 1, output.data(), vocab_size) != 0) return RWKV_ERROR_EVAL;
    logits = Tensor1D::make(output.data(), TensorDType::F32, output.size());
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::eval(std::vector<int> ids, Tensor1D &logits) {
    if (!context || ids.empty() || ids.size() > std::numeric_limits<int32_t>::max()) return RWKV_ERROR_EVAL;
    std::vector<int32_t> tokens(ids.begin(), ids.end());
    if (rwkv_coreai_eval(context, tokens.data(), static_cast<int32_t>(tokens.size()), output.data(), vocab_size) != 0)
        return RWKV_ERROR_EVAL;
    logits = Tensor1D::make(output.data(), TensorDType::F32, output.size());
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::get_state(std::any &state) {
    if (!context) return RWKV_ERROR_EVAL;
    std::vector<uint8_t> bytes(static_cast<size_t>(state_bytes));
    if (rwkv_coreai_read_state(context, bytes.data(), state_bytes) != 0) return RWKV_ERROR_EVAL;
    state = std::move(bytes);
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::set_state(std::any state) {
    if (!context) return RWKV_ERROR_EVAL;
    if (!state.has_value()) return zero_state();
    const auto *bytes = std::any_cast<std::vector<uint8_t>>(&state);
    if (!bytes || bytes->size() != static_cast<size_t>(state_bytes)) return RWKV_ERROR_INVALID_PARAMETERS;
    return rwkv_coreai_write_state(context, bytes->data(), state_bytes) == 0 ? RWKV_SUCCESS : RWKV_ERROR_EVAL;
}

int coreai_rwkv_backend::zero_state() {
    if (!context) return RWKV_ERROR_EVAL;
    return rwkv_coreai_write_state(context, nullptr, state_bytes) == 0 ? RWKV_SUCCESS : RWKV_ERROR_EVAL;
}

int coreai_rwkv_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    const auto *bytes = std::any_cast<std::vector<uint8_t>>(&state);
    if (!context || !bytes || bytes->size() != static_cast<size_t>(state_bytes)) return RWKV_ERROR_INVALID_PARAMETERS;
    data = *bytes;
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    if (!context || data.size() != static_cast<size_t>(state_bytes)) return RWKV_ERROR_INVALID_PARAMETERS;
    state = data;
    return RWKV_SUCCESS;
}

int coreai_rwkv_backend::release_model() {
    if (context) rwkv_coreai_free(context);
    context = nullptr;
    state_bytes = 0;
    output.clear();
    progress.store(0);
    return RWKV_SUCCESS;
}

}
