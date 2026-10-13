#ifndef RWKV_MOBILE_NNRT_BACKEND_H
#define RWKV_MOBILE_NNRT_BACKEND_H

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rwkvmobile {

struct nnrt_backend_args {
    static constexpr uint32_t kVersion = 2;

    uint32_t struct_version = kVersion;
    size_t device_id = static_cast<size_t>(-1);
    const char *required_device_name = "kirin9020";
    const char *cache_dir = nullptr;
    uint32_t cache_version = 1;
    bool strict_accelerator = true;
    bool enable_profiling = true;
    const char *profiling_output_dir = nullptr;
    bool enable_stage_diagnostics = false;
};

class nnrt_backend final : public execution_provider {
public:
    nnrt_backend();
    ~nnrt_backend();

    int init(void *extra) override;
    int load_model(std::string model_path, void *extra = nullptr) override;
    float get_load_progress() const override;

    int eval(int id, Tensor1D &logits) override;
    int eval(std::vector<int> ids, Tensor1D &logits) override;

    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;

    int release_model() override;
    int release() override;
    bool is_available() override;

    double get_prefill_speed() override;
    double get_decode_speed() override;
    void reset_speed_stats() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rwkvmobile

#endif
