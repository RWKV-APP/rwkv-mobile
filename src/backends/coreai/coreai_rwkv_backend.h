#pragma once

#include "backend.h"
#include <atomic>

namespace rwkvmobile {

class coreai_rwkv_backend final : public execution_provider {
public:
    ~coreai_rwkv_backend() { release_model(); }
    int init(void *extra) override;
    int load_model(std::string path, void *extra) override;
    float get_load_progress() const override { return progress.load(); }
    int eval(int id, Tensor1D &logits) override;
    int eval(std::vector<int> ids, Tensor1D &logits) override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override { state.reset(); return 0; }
    int zero_state() override;
    int get_state_on_batch_slot(int slot, std::any &state) override {
        return slot == 0 ? get_state(state) : RWKV_ERROR_UNSUPPORTED;
    }
    int set_state_on_batch_slot(int slot, std::any state) override {
        return slot == 0 ? set_state(std::move(state)) : RWKV_ERROR_UNSUPPORTED;
    }
    int zero_state_on_batch_slot(int slot) override {
        return slot == 0 ? zero_state() : RWKV_ERROR_UNSUPPORTED;
    }
    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;
    int release_model() override;
    int release() override { return release_model(); }
    bool is_available() override;
    static bool is_supported();

private:
    void *context = nullptr;
    int64_t state_bytes = 0;
    std::vector<float> output;
    std::atomic<float> progress{0};
};

}
