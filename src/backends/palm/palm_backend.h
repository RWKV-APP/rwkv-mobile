#ifndef PALM_BACKEND_H
#define PALM_BACKEND_H

#include "backend.h"

struct Tensor;
class LLMEngine;

namespace rwkvmobile {

class palm_backend : public execution_provider {
public:
    palm_backend();
    ~palm_backend();

    int load_model(std::string model_path, void* extra = nullptr) override;
    int eval(int id, Tensor1D& logits) override;
    int eval(std::vector<int> ids, Tensor1D& logits) override;
    int get_state(std::any& state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    static bool is_supported();
    bool is_available() override { return is_supported(); }

private:
    int update_logits(const Tensor& hidden, int token_count, Tensor1D& logits);

    std::unique_ptr<LLMEngine> engine;
    std::vector<int> token_history;
    std::vector<float> logits_storage;
};

} // namespace rwkvmobile

#endif
