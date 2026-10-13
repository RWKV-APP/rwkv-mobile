#include "nnrt_backend.h"

#include "logger.h"
#include "safetensors_file.h"

#include <CANNKit/hiai_options.h>
#include <neural_network_runtime/neural_network_core.h>
#include <neural_network_runtime/neural_network_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rwkvmobile {
namespace {

using Shape = std::vector<int32_t>;
using StateData = std::vector<std::vector<uint8_t>>;

constexpr int kRwkv7HeadSize = 64;
constexpr int kExpectedLayers = 12;
constexpr int kExpectedHiddenSize = 768;
constexpr int kExpectedHeadCount = 12;
constexpr int kExpectedFfnSize = 3072;
constexpr int kExpectedVocabSize = 65536;
constexpr int kPrefillSequenceLength = 64;
constexpr const char *kPrecomputedEmbeddingMarker = "rwkv.nnrt.emb_ln0_precomputed";
// Bump whenever a graph operation, constant, tensor shape, or numerical formula changes.
constexpr uint32_t kGraphCacheSchemaVersion = 8;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr float kGroupNormEpsilon = 64.0e-5F;
constexpr float kTimeDecayScale = -0.606531F;

struct ModelDeleter {
    void operator()(OH_NNModel *model) const {
        OH_NNModel_Destroy(&model);
    }
};

struct CompilationDeleter {
    void operator()(OH_NNCompilation *compilation) const {
        OH_NNCompilation_Destroy(&compilation);
    }
};

struct ExecutorDeleter {
    void operator()(OH_NNExecutor *executor) const {
        OH_NNExecutor_Destroy(&executor);
    }
};

struct TensorDescDeleter {
    void operator()(NN_TensorDesc *description) const {
        OH_NNTensorDesc_Destroy(&description);
    }
};

struct TensorDeleter {
    void operator()(NN_Tensor *tensor) const {
        OH_NNTensor_Destroy(&tensor);
    }
};

using ModelPtr = std::unique_ptr<OH_NNModel, ModelDeleter>;
using CompilationPtr = std::unique_ptr<OH_NNCompilation, CompilationDeleter>;
using ExecutorPtr = std::unique_ptr<OH_NNExecutor, ExecutorDeleter>;
using TensorDescPtr = std::unique_ptr<NN_TensorDesc, TensorDescDeleter>;
using TensorPtr = std::unique_ptr<NN_Tensor, TensorDeleter>;

void require_success(OH_NN_ReturnCode code, const std::string &stage) {
    if (code != OH_NN_SUCCESS) {
        throw std::runtime_error(stage + " failed with NNRT error " + std::to_string(static_cast<int>(code)));
    }
}

uint64_t shape_element_count(const Shape &shape) {
    uint64_t count = 1;
    for (const int32_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("negative fixed tensor dimension");
        }
        const uint64_t value = static_cast<uint64_t>(dimension);
        if (value != 0 && count > std::numeric_limits<uint64_t>::max() / value) {
            throw std::runtime_error("tensor element count overflow");
        }
        count *= value;
    }
    return count;
}

std::string normalize_device_name(const char *name) {
    std::string result;
    if (name == nullptr) {
        return result;
    }
    for (const char value : std::string(name)) {
        const unsigned char unsigned_value = static_cast<unsigned char>(value);
        if (std::isalnum(unsigned_value) == 0) {
            continue;
        }
        result.push_back(static_cast<char>(std::tolower(unsigned_value)));
    }
    return result;
}

bool device_name_matches(const char *name, const std::string &required_name) {
    if (required_name.empty()) {
        return true;
    }
    const std::string normalized_name = normalize_device_name(name);
    const std::string normalized_required_name = normalize_device_name(required_name.c_str());
    return !normalized_required_name.empty() &&
           normalized_name.find(normalized_required_name) != std::string::npos;
}

Shape shape_from_info(const SafetensorsTensorInfo &info) {
    Shape shape;
    shape.reserve(info.shape.size());
    for (const int64_t dimension : info.shape) {
        if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("unsupported tensor dimension in " + info.name);
        }
        shape.push_back(static_cast<int32_t>(dimension));
    }
    return shape;
}

std::vector<uint8_t> bytes_from_value(const void *data, size_t size) {
    std::vector<uint8_t> bytes(size);
    if (size != 0) {
        std::memcpy(bytes.data(), data, size);
    }
    return bytes;
}

template <typename T>
std::vector<uint8_t> bytes_from_scalar(T value) {
    return bytes_from_value(&value, sizeof(value));
}

struct GraphDefinition {
    std::vector<std::string> operation_names;
    std::vector<size_t> state_bytes;
    size_t input_count = 0;
    size_t output_count = 0;
};

class GraphBuilder {
public:
    GraphBuilder(OH_NNModel *model, const SafetensorsFile &file)
        : model_(model), file_(file) {}

    GraphDefinition build_embedding() {
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const uint32_t token_input = add_tensor({1}, OH_NN_INT32, OH_NN_TENSOR, nullptr, 0);
        inputs_.push_back(token_input);
        const uint32_t embedding_weight = add_file_tensor("emb.weight");
        const uint32_t gather_axis = add_i32_tensor({1}, 0, OH_NN_TENSOR);
        const uint32_t gathered = add_gather(
            embedding_weight,
            token_input,
            gather_axis,
            vector_shape,
            "embedding.gather");
        uint32_t normalized = gathered;
        if (!file_.has_tensor(kPrecomputedEmbeddingMarker)) {
            normalized = add_layer_norm(
                gathered,
                vector_shape,
                add_file_tensor("blocks.0.ln0.weight"),
                add_file_tensor("blocks.0.ln0.bias"),
                kLayerNormEpsilon,
                "embedding.layer_norm");
        }
        outputs_.push_back(normalized);
        if (normalized != gathered) {
            outputs_.push_back(gathered);
        }
        return finish("embedding");
    }

    GraphDefinition build_embedding_sequence() {
        const Shape sequence_shape = {kPrefillSequenceLength, kExpectedHiddenSize};
        const uint32_t token_input = add_tensor(
            {kPrefillSequenceLength}, OH_NN_INT32, OH_NN_TENSOR, nullptr, 0);
        inputs_.push_back(token_input);
        const uint32_t embedding_weight = add_file_tensor("emb.weight");
        const uint32_t gather_axis = add_i32_tensor({1}, 0, OH_NN_TENSOR);
        const uint32_t gathered = add_gather(
            embedding_weight,
            token_input,
            gather_axis,
            sequence_shape,
            "prefill.embedding.gather");
        uint32_t normalized = gathered;
        if (!file_.has_tensor(kPrecomputedEmbeddingMarker)) {
            normalized = add_layer_norm(
                gathered,
                sequence_shape,
                add_file_tensor("blocks.0.ln0.weight"),
                add_file_tensor("blocks.0.ln0.bias"),
                kLayerNormEpsilon,
                "prefill.embedding.layer_norm");
        }
        outputs_.push_back(normalized);
        return finish("prefill embedding");
    }

    GraphDefinition build_layer(int layer) {
        if (layer < 0 || layer >= kExpectedLayers) {
            throw std::runtime_error("invalid RWKV7 layer graph index");
        }
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const Shape wkv_shape = {kExpectedHeadCount, kRwkv7HeadSize, kRwkv7HeadSize};
        const uint32_t x_input = add_tensor(vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t attention_state = add_tensor(vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t wkv_state = add_tensor(wkv_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t ffn_state = add_tensor(vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        inputs_ = {x_input, attention_state, wkv_state, ffn_state};
        uint32_t value_first = 0;
        if (layer > 0) {
            value_first = add_tensor(vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
            inputs_.push_back(value_first);
        }
        std::vector<uint32_t> state_outputs;
        state_outputs.reserve(3);
        const uint32_t x_output = add_layer(
            layer,
            x_input,
            attention_state,
            wkv_state,
            ffn_state,
            value_first,
            state_outputs);
        outputs_.push_back(x_output);
        outputs_.insert(outputs_.end(), state_outputs.begin(), state_outputs.end());
        if (layer == 0) {
            outputs_.push_back(value_first);
        }
        state_bytes_.push_back(shape_element_count(vector_shape) * sizeof(uint16_t));
        state_bytes_.push_back(shape_element_count(wkv_shape) * sizeof(uint16_t));
        state_bytes_.push_back(shape_element_count(vector_shape) * sizeof(uint16_t));
        return finish("layer " + std::to_string(layer));
    }

    GraphDefinition build_layer_sequence(int layer) {
        if (layer < 0 || layer >= kExpectedLayers) {
            throw std::runtime_error("invalid RWKV7 prefill layer graph index");
        }
        const Shape sequence_shape = {kPrefillSequenceLength, kExpectedHiddenSize};
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const Shape wkv_shape = {kExpectedHeadCount, kRwkv7HeadSize, kRwkv7HeadSize};
        const uint32_t x_input = add_tensor(
            sequence_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t attention_state = add_tensor(
            vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t wkv_state = add_tensor(
            wkv_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        const uint32_t ffn_state = add_tensor(
            vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        inputs_ = {x_input, attention_state, wkv_state, ffn_state};
        uint32_t value_first = 0;
        if (layer > 0) {
            value_first = add_tensor(
                sequence_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
            inputs_.push_back(value_first);
        }
        std::vector<uint32_t> state_outputs;
        state_outputs.reserve(3);
        const uint32_t x_output = add_layer_sequence(
            layer,
            x_input,
            attention_state,
            wkv_state,
            ffn_state,
            value_first,
            state_outputs);
        outputs_.push_back(x_output);
        outputs_.insert(outputs_.end(), state_outputs.begin(), state_outputs.end());
        if (layer == 0) {
            outputs_.push_back(value_first);
        }
        state_bytes_.push_back(shape_element_count(vector_shape) * sizeof(uint16_t));
        state_bytes_.push_back(shape_element_count(wkv_shape) * sizeof(uint16_t));
        state_bytes_.push_back(shape_element_count(vector_shape) * sizeof(uint16_t));
        return finish("prefill layer " + std::to_string(layer));
    }

    GraphDefinition build_head() {
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const uint32_t x_input = add_tensor(vector_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        inputs_.push_back(x_input);
        const uint32_t normalized = add_layer_norm(
            x_input,
            vector_shape,
            add_file_tensor("ln_out.weight"),
            add_file_tensor("ln_out.bias"),
            kLayerNormEpsilon,
            "output.layer_norm");
        const uint32_t logits = add_linear(
            normalized,
            "head.weight",
            {1, kExpectedVocabSize},
            "output.head");
        outputs_.push_back(logits);
        return finish("head");
    }

private:
    GraphDefinition finish(const std::string &name) {
        OH_NN_UInt32Array input_array = {inputs_.data(), static_cast<uint32_t>(inputs_.size())};
        OH_NN_UInt32Array output_array = {outputs_.data(), static_cast<uint32_t>(outputs_.size())};
        require_success(
            OH_NNModel_SpecifyInputsAndOutputs(model_, &input_array, &output_array),
            "specify NNRT " + name + " inputs and outputs");
        require_success(OH_NNModel_Finish(model_), "finish NNRT " + name + " model");

        GraphDefinition definition;
        definition.operation_names = std::move(operation_names_);
        definition.state_bytes = std::move(state_bytes_);
        definition.input_count = inputs_.size();
        definition.output_count = outputs_.size();
        return definition;
    }

    uint32_t add_tensor(
        const Shape &shape,
        OH_NN_DataType data_type,
        OH_NN_TensorType tensor_type,
        const void *data,
        size_t data_size) {
        TensorDescPtr description(OH_NNTensorDesc_Create());
        if (!description) {
            throw std::runtime_error("failed to create NNRT tensor description");
        }
        require_success(
            OH_NNTensorDesc_SetShape(
                description.get(),
                shape.empty() ? nullptr : shape.data(),
                shape.size()),
            "set NNRT tensor shape");
        require_success(
            OH_NNTensorDesc_SetDataType(description.get(), data_type),
            "set NNRT tensor data type");
        require_success(
            OH_NNTensorDesc_SetFormat(description.get(), OH_NN_FORMAT_NONE),
            "set NNRT tensor format");
        require_success(
            OH_NNModel_AddTensorToModel(model_, description.get()),
            "add NNRT tensor to model");
        const uint32_t index = tensor_count_++;
        require_success(
            OH_NNModel_SetTensorType(model_, index, tensor_type),
            "set NNRT tensor type");
        if (data != nullptr) {
            require_success(
                OH_NNModel_SetTensorData(model_, index, data, data_size),
                "set NNRT constant tensor data");
        }
        return index;
    }

    uint32_t add_owned_tensor(
        const Shape &shape,
        OH_NN_DataType data_type,
        OH_NN_TensorType tensor_type,
        std::vector<uint8_t> data) {
        owned_constants_.push_back(std::move(data));
        const std::vector<uint8_t> &stored = owned_constants_.back();
        if (shape.empty()) {
            const uint32_t index = tensor_count_++;
            const OH_NN_Tensor tensor = {
                data_type,
                0,
                nullptr,
                nullptr,
                tensor_type,
            };
            require_success(
                OH_NNModel_AddTensor(model_, &tensor),
                "add NNRT scalar tensor to model");
            require_success(
                OH_NNModel_SetTensorData(model_, index, stored.data(), stored.size()),
                "set NNRT scalar tensor data");
            return index;
        }
        return add_tensor(shape, data_type, tensor_type, stored.data(), stored.size());
    }

    uint32_t add_i32_tensor(const Shape &shape, int32_t value, OH_NN_TensorType tensor_type) {
        return add_owned_tensor(shape, OH_NN_INT32, tensor_type, bytes_from_scalar(value));
    }

    uint32_t add_i32_tensor(
        const Shape &shape,
        const std::vector<int32_t> &values,
        OH_NN_TensorType tensor_type) {
        return add_owned_tensor(
            shape,
            OH_NN_INT32,
            tensor_type,
            bytes_from_value(values.data(), values.size() * sizeof(values[0])));
    }

    uint32_t add_i32_parameter(int32_t value, OH_NN_TensorType tensor_type) {
        return add_owned_tensor({}, OH_NN_INT32, tensor_type, bytes_from_scalar(value));
    }

    uint32_t add_i64_parameter(int64_t value, OH_NN_TensorType tensor_type) {
        return add_owned_tensor({}, OH_NN_INT64, tensor_type, bytes_from_scalar(value));
    }

    uint32_t add_f32_parameter(float value, OH_NN_TensorType tensor_type) {
        return add_owned_tensor({}, OH_NN_FLOAT32, tensor_type, bytes_from_scalar(value));
    }

    uint32_t add_bool_parameter(bool value, OH_NN_TensorType tensor_type) {
        return add_owned_tensor({}, OH_NN_BOOL, tensor_type, bytes_from_scalar(value));
    }

    uint32_t add_activation_parameter(OH_NN_TensorType tensor_type) {
        const int8_t activation = OH_NN_FUSED_NONE;
        return add_owned_tensor({}, OH_NN_INT8, tensor_type, bytes_from_scalar(activation));
    }

    uint32_t add_f16_channel_constant(const Shape &shape, float value) {
        const size_t count = static_cast<size_t>(shape_element_count(shape));
        const half_float::half half_value(value);
        std::vector<half_float::half> values(count, half_value);
        return add_owned_tensor(
            shape,
            OH_NN_FLOAT16,
            OH_NN_TENSOR,
            bytes_from_value(values.data(), values.size() * sizeof(values[0])));
    }

    uint32_t add_file_tensor(const std::string &name, const Shape &override_shape = {}) {
        const std::string cache_key = override_shape.empty()
            ? name
            : name + "#" + shape_key(override_shape);
        const auto cached = file_tensor_indices_.find(cache_key);
        if (cached != file_tensor_indices_.end()) {
            return cached->second;
        }

        const SafetensorsTensorInfo *info = file_.get_tensor_info(name);
        const void *data = file_.get_tensor_data(name);
        if (info == nullptr || data == nullptr) {
            throw std::runtime_error("missing safetensors tensor: " + name);
        }
        if (info->dtype != SafetensorsDType::F16) {
            throw std::runtime_error("NNRT v1 requires F16 tensor: " + name);
        }
        const Shape shape = override_shape.empty() ? shape_from_info(*info) : override_shape;
        if (shape_element_count(shape) * sizeof(uint16_t) != info->data_size) {
            throw std::runtime_error("NNRT tensor reshape changes element count: " + name);
        }
        const uint32_t index = add_tensor(
            shape,
            OH_NN_FLOAT16,
            OH_NN_TENSOR,
            data,
            static_cast<size_t>(info->data_size));
        file_tensor_indices_.emplace(cache_key, index);
        return index;
    }

    uint32_t file_vector(const std::string &name) {
        return add_file_tensor(name, {1, kExpectedHiddenSize});
    }

    std::string shape_key(const Shape &shape) const {
        std::string result;
        for (const int32_t dimension : shape) {
            result += std::to_string(dimension) + ",";
        }
        return result;
    }

    uint32_t add_operation(
        OH_NN_OperationType type,
        std::vector<uint32_t> parameters,
        std::vector<uint32_t> inputs,
        const Shape &output_shape,
        const std::string &name) {
        const uint32_t output = add_tensor(output_shape, OH_NN_FLOAT16, OH_NN_TENSOR, nullptr, 0);
        OH_NN_UInt32Array parameter_array = {
            parameters.empty() ? nullptr : parameters.data(),
            static_cast<uint32_t>(parameters.size())};
        OH_NN_UInt32Array input_array = {
            inputs.empty() ? nullptr : inputs.data(),
            static_cast<uint32_t>(inputs.size())};
        uint32_t output_index = output;
        OH_NN_UInt32Array output_array = {&output_index, 1};
        require_success(
            OH_NNModel_AddOperation(model_, type, &parameter_array, &input_array, &output_array),
            "add NNRT operation " + name);
        operation_names_.push_back(name);
        return output;
    }

    uint32_t add_binary(
        OH_NN_OperationType type,
        OH_NN_TensorType activation_type,
        uint32_t left,
        uint32_t right,
        const Shape &output_shape,
        const std::string &name) {
        return add_operation(
            type,
            {add_activation_parameter(activation_type)},
            {left, right},
            output_shape,
            name);
    }

    uint32_t add_add(uint32_t left, uint32_t right, const Shape &shape, const std::string &name) {
        return add_binary(OH_NN_OPS_ADD, OH_NN_ADD_ACTIVATIONTYPE, left, right, shape, name);
    }

    uint32_t add_sub(uint32_t left, uint32_t right, const Shape &shape, const std::string &name) {
        return add_binary(OH_NN_OPS_SUB, OH_NN_SUB_ACTIVATIONTYPE, left, right, shape, name);
    }

    uint32_t add_mul(uint32_t left, uint32_t right, const Shape &shape, const std::string &name) {
        return add_binary(OH_NN_OPS_MUL, OH_NN_MUL_ACTIVATION_TYPE, left, right, shape, name);
    }

    uint32_t add_div(uint32_t left, uint32_t right, const Shape &shape, const std::string &name) {
        return add_binary(OH_NN_OPS_DIV, OH_NN_DIV_ACTIVATIONTYPE, left, right, shape, name);
    }

    uint32_t add_unary(
        OH_NN_OperationType type,
        uint32_t input,
        const Shape &shape,
        const std::string &name) {
        return add_operation(type, {}, {input}, shape, name);
    }

    uint32_t add_matmul(
        uint32_t left,
        uint32_t right,
        const Shape &shape,
        bool transpose_left,
        bool transpose_right,
        const std::string &name) {
        return add_operation(
            OH_NN_OPS_MATMUL,
            {
                add_bool_parameter(transpose_left, OH_NN_MATMUL_TRANSPOSE_A),
                add_bool_parameter(transpose_right, OH_NN_MATMUL_TRANSPOSE_B),
                add_activation_parameter(OH_NN_MATMUL_ACTIVATION_TYPE),
            },
            {left, right},
            shape,
            name);
    }

    uint32_t add_linear(
        uint32_t input,
        const std::string &weight_name,
        const Shape &output_shape,
        const std::string &name) {
        return add_matmul(input, add_file_tensor(weight_name), output_shape, false, true, name);
    }

    uint32_t add_reshape(uint32_t input, const Shape &shape, const std::string &name) {
        std::vector<int32_t> shape_values(shape.begin(), shape.end());
        const uint32_t shape_tensor = add_owned_tensor(
            {static_cast<int32_t>(shape_values.size())},
            OH_NN_INT32,
            OH_NN_TENSOR,
            bytes_from_value(shape_values.data(), shape_values.size() * sizeof(shape_values[0])));
        return add_operation(OH_NN_OPS_RESHAPE, {}, {input, shape_tensor}, shape, name);
    }

    uint32_t add_gather(
        uint32_t input,
        uint32_t indices,
        uint32_t axis,
        const Shape &shape,
        const std::string &name) {
        return add_operation(OH_NN_OPS_GATHER, {}, {input, indices, axis}, shape, name);
    }

    uint32_t add_concat(
        const std::vector<uint32_t> &inputs,
        int64_t axis,
        const Shape &shape,
        const std::string &name) {
        return add_operation(
            OH_NN_OPS_CONCAT,
            {add_i64_parameter(axis, OH_NN_CONCAT_AXIS)},
            inputs,
            shape,
            name);
    }

    uint32_t add_gather_rows(
        uint32_t input,
        const std::vector<int32_t> &rows,
        const Shape &shape,
        const std::string &name) {
        const uint32_t indices = add_i32_tensor(
            {static_cast<int32_t>(rows.size())}, rows, OH_NN_TENSOR);
        const uint32_t axis = add_i32_tensor({1}, 0, OH_NN_TENSOR);
        return add_gather(input, indices, axis, shape, name);
    }

    uint32_t add_gather_row(
        uint32_t input,
        int32_t row,
        const Shape &shape,
        const std::string &name) {
        return add_gather_rows(input, {row}, shape, name);
    }

    uint32_t add_shifted_sequence(
        uint32_t initial_state,
        uint32_t sequence,
        const Shape &sequence_shape,
        const std::string &name) {
        if (sequence_shape.size() != 2 || sequence_shape[0] != kPrefillSequenceLength) {
            throw std::runtime_error("invalid fixed prefill sequence shape");
        }
        std::vector<int32_t> prefix_rows(kPrefillSequenceLength - 1);
        for (int32_t row = 0; row < kPrefillSequenceLength - 1; ++row) {
            prefix_rows[static_cast<size_t>(row)] = row;
        }
        const uint32_t prefix = add_gather_rows(
            sequence,
            prefix_rows,
            {kPrefillSequenceLength - 1, sequence_shape[1]},
            name + ".prefix");
        return add_concat(
            {initial_state, prefix},
            0,
            sequence_shape,
            name + ".concat");
    }

    uint32_t add_scatter_nd(
        uint32_t indices,
        uint32_t updates,
        uint32_t output_shape,
        const Shape &shape,
        const std::string &name) {
        return add_operation(
            OH_NN_OPS_SCATTER_ND,
            {},
            {indices, updates, output_shape},
            shape,
            name);
    }

    uint32_t wkv_diagonal_indices() {
        if (wkv_diagonal_indices_ != std::numeric_limits<uint32_t>::max()) {
            return wkv_diagonal_indices_;
        }
        std::vector<int32_t> indices(
            static_cast<size_t>(kExpectedHeadCount) * kRwkv7HeadSize * 3);
        for (int head = 0; head < kExpectedHeadCount; ++head) {
            for (int channel = 0; channel < kRwkv7HeadSize; ++channel) {
                const size_t offset =
                    (static_cast<size_t>(head) * kRwkv7HeadSize + channel) * 3;
                indices[offset] = head;
                indices[offset + 1] = channel;
                indices[offset + 2] = channel;
            }
        }
        wkv_diagonal_indices_ = add_owned_tensor(
            {kExpectedHiddenSize, 3},
            OH_NN_INT32,
            OH_NN_TENSOR,
            bytes_from_value(indices.data(), indices.size() * sizeof(indices[0])));
        return wkv_diagonal_indices_;
    }

    uint32_t wkv_shape_tensor() {
        if (wkv_shape_tensor_ != std::numeric_limits<uint32_t>::max()) {
            return wkv_shape_tensor_;
        }
        const std::vector<int32_t> shape = {
            kExpectedHeadCount,
            kRwkv7HeadSize,
            kRwkv7HeadSize,
        };
        wkv_shape_tensor_ = add_owned_tensor(
            {static_cast<int32_t>(shape.size())},
            OH_NN_INT32,
            OH_NN_TENSOR,
            bytes_from_value(shape.data(), shape.size() * sizeof(shape[0])));
        return wkv_shape_tensor_;
    }

    uint32_t add_reduce(
        OH_NN_OperationType type,
        OH_NN_TensorType keep_dims_type,
        OH_NN_TensorType reduce_to_end_type,
        OH_NN_TensorType coefficient_type,
        uint32_t input,
        int32_t axis,
        const Shape &shape,
        const std::string &name) {
        const uint32_t axis_tensor = add_i32_tensor({1}, axis, OH_NN_TENSOR);
        return add_operation(
            type,
            {
                add_bool_parameter(true, keep_dims_type),
                add_bool_parameter(false, reduce_to_end_type),
                add_f32_parameter(1.0F, coefficient_type),
            },
            {input, axis_tensor},
            shape,
            name);
    }

    uint32_t add_reduce_mean(
        uint32_t input,
        int32_t axis,
        const Shape &shape,
        const std::string &name) {
        return add_reduce(
            OH_NN_OPS_REDUCE_MEAN,
            OH_NN_REDUCE_MEAN_KEEP_DIMS,
            OH_NN_REDUCE_MEAN_REDUCE_TO_END,
            OH_NN_REDUCE_MEAN_COEFF,
            input,
            axis,
            shape,
            name);
    }

    uint32_t add_reduce_sum(
        uint32_t input,
        int32_t axis,
        const Shape &shape,
        const std::string &name) {
        return add_reduce(
            OH_NN_OPS_REDUCE_SUM,
            OH_NN_REDUCE_SUM_KEEP_DIMS,
            OH_NN_REDUCE_SUM_REDUCE_TO_END,
            OH_NN_REDUCE_SUM_COEFF,
            input,
            axis,
            shape,
            name);
    }

    uint32_t add_exp(uint32_t input, const Shape &shape, const std::string &name) {
        return add_operation(
            OH_NN_OPS_EXP,
            {
                add_f32_parameter(-1.0F, OH_NN_EXP_BASE),
                add_f32_parameter(1.0F, OH_NN_EXP_SCALE),
                add_f32_parameter(0.0F, OH_NN_EXP_SHIFT),
            },
            {input},
            shape,
            name);
    }

    uint32_t add_layer_norm(
        uint32_t input,
        const Shape &shape,
        uint32_t scale,
        uint32_t bias,
        float epsilon,
        const std::string &name) {
        if (shape.size() != 2 || shape[0] <= 0) {
            throw std::runtime_error("NNRT layer norm expects a [sequence, hidden] tensor");
        }
        const Shape rank_three_shape = {1, shape[0], shape[1]};
        const uint32_t rank_three_input = add_reshape(input, rank_three_shape, name + ".input_reshape");
        const uint32_t normalized = add_operation(
            OH_NN_OPS_LAYER_NORM,
            {
                add_i64_parameter(2, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS),
                add_f32_parameter(epsilon, OH_NN_LAYER_NORM_EPSILON),
                add_i64_parameter(2, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS),
            },
            {rank_three_input, scale, bias},
            rank_three_shape,
            name);
        return add_reshape(normalized, shape, name + ".output_reshape");
    }

    uint32_t add_normalize(
        uint32_t input,
        const Shape &shape,
        int32_t axis,
        uint32_t scale,
        uint32_t bias,
        float epsilon,
        const std::string &name) {
        Shape reduced_shape = shape;
        const int32_t normalized_axis = axis < 0 ? static_cast<int32_t>(shape.size()) + axis : axis;
        if (normalized_axis < 0 || static_cast<size_t>(normalized_axis) >= shape.size()) {
            throw std::runtime_error("invalid normalization axis");
        }
        reduced_shape[static_cast<size_t>(normalized_axis)] = 1;
        const uint32_t mean = add_reduce_mean(input, axis, reduced_shape, name + ".mean");
        const uint32_t centered = add_sub(input, mean, shape, name + ".center");
        const uint32_t square = add_mul(centered, centered, shape, name + ".square");
        const uint32_t variance = add_reduce_mean(square, axis, reduced_shape, name + ".variance");
        const uint32_t stabilized = add_add(
            variance,
            add_f16_channel_constant(reduced_shape, epsilon),
            reduced_shape,
            name + ".epsilon");
        const uint32_t inverse_deviation = add_unary(
            OH_NN_OPS_RSQRT,
            stabilized,
            reduced_shape,
            name + ".rsqrt");
        const uint32_t normalized = add_mul(centered, inverse_deviation, shape, name + ".normalize");
        const uint32_t scaled = add_mul(normalized, scale, shape, name + ".scale");
        return add_add(scaled, bias, shape, name + ".bias");
    }

    uint32_t add_l2_normalize(
        uint32_t input,
        const Shape &shape,
        int32_t axis,
        const std::string &name) {
        Shape reduced_shape = shape;
        const int32_t normalized_axis = axis < 0 ? static_cast<int32_t>(shape.size()) + axis : axis;
        reduced_shape[static_cast<size_t>(normalized_axis)] = 1;
        const uint32_t square = add_mul(input, input, shape, name + ".square");
        const uint32_t sum = add_reduce_sum(square, axis, reduced_shape, name + ".sum");
        const uint32_t inverse_norm = add_unary(
            OH_NN_OPS_RSQRT,
            sum,
            reduced_shape,
            name + ".rsqrt");
        return add_mul(input, inverse_norm, shape, name + ".normalize");
    }

    uint32_t add_layer(
        int layer,
        uint32_t input,
        uint32_t attention_state,
        uint32_t wkv_state,
        uint32_t ffn_state,
        uint32_t &value_first,
        std::vector<uint32_t> &state_outputs) {
        const std::string block = "blocks." + std::to_string(layer);
        const std::string attention = block + ".att";
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const Shape head_vector_shape = {kExpectedHeadCount, kRwkv7HeadSize};
        const Shape head_column_shape = {kExpectedHeadCount, kRwkv7HeadSize, 1};
        const Shape head_row_shape = {kExpectedHeadCount, 1, kRwkv7HeadSize};
        const Shape wkv_shape = {kExpectedHeadCount, kRwkv7HeadSize, kRwkv7HeadSize};

        const uint32_t attention_normalized = add_layer_norm(
            input,
            vector_shape,
            add_file_tensor(block + ".ln1.weight"),
            add_file_tensor(block + ".ln1.bias"),
            kLayerNormEpsilon,
            block + ".att.layer_norm");
        const uint32_t shift_delta = add_sub(
            attention_state,
            attention_normalized,
            vector_shape,
            block + ".att.shift_delta");

        const auto token_mix = [&](const std::string &suffix) {
            const uint32_t mixed_delta = add_mul(
                shift_delta,
                file_vector(attention + ".x_" + suffix),
                vector_shape,
                block + ".att.mix_" + suffix + ".mul");
            return add_add(
                attention_normalized,
                mixed_delta,
                vector_shape,
                block + ".att.mix_" + suffix + ".add");
        };

        const uint32_t xr = token_mix("r");
        const uint32_t xw = token_mix("w");
        const uint32_t xk = token_mix("k");
        const uint32_t xv = token_mix("v");
        const uint32_t xa = token_mix("a");
        const uint32_t xg = token_mix("g");

        const uint32_t receptance = add_linear(
            xr,
            attention + ".receptance.weight",
            vector_shape,
            block + ".att.receptance");
        uint32_t key = add_linear(
            xk,
            attention + ".key.weight",
            vector_shape,
            block + ".att.key");
        uint32_t value = add_linear(
            xv,
            attention + ".value.weight",
            vector_shape,
            block + ".att.value");

        const uint32_t w_hidden = add_linear(
            xw,
            attention + ".w1",
            {1, 64},
            block + ".att.w1");
        const uint32_t w_tanh = add_unary(
            OH_NN_OPS_TANH,
            w_hidden,
            {1, 64},
            block + ".att.w_tanh");
        const uint32_t w_projected = add_linear(
            w_tanh,
            attention + ".w2",
            vector_shape,
            block + ".att.w2");
        const uint32_t w_with_bias = add_add(
            w_projected,
            file_vector(attention + ".w0"),
            vector_shape,
            block + ".att.w0");
        const uint32_t w_sigmoid = add_unary(
            OH_NN_OPS_SIGMOID,
            w_with_bias,
            vector_shape,
            block + ".att.w_sigmoid");
        const uint32_t w_scaled = add_mul(
            w_sigmoid,
            add_f16_channel_constant(vector_shape, kTimeDecayScale),
            vector_shape,
            block + ".att.w_scale");
        const uint32_t time_decay_vector = add_exp(
            w_scaled,
            vector_shape,
            block + ".att.w_exp");

        const uint32_t a_hidden = add_linear(
            xa,
            attention + ".a1",
            {1, 64},
            block + ".att.a1");
        const uint32_t a_projected = add_linear(
            a_hidden,
            attention + ".a2",
            vector_shape,
            block + ".att.a2");
        const uint32_t a_with_bias = add_add(
            a_projected,
            file_vector(attention + ".a0"),
            vector_shape,
            block + ".att.a0");
        const uint32_t a = add_unary(
            OH_NN_OPS_SIGMOID,
            a_with_bias,
            vector_shape,
            block + ".att.a_sigmoid");

        const uint32_t g_hidden = add_linear(
            xg,
            attention + ".g1",
            {1, 128},
            block + ".att.g1");
        const uint32_t g_sigmoid = add_unary(
            OH_NN_OPS_SIGMOID,
            g_hidden,
            {1, 128},
            block + ".att.g_sigmoid");
        const uint32_t gate = add_linear(
            g_sigmoid,
            attention + ".g2",
            vector_shape,
            block + ".att.g2");

        uint32_t kk = add_mul(
            key,
            file_vector(attention + ".k_k"),
            vector_shape,
            block + ".att.kk_scale");
        kk = add_reshape(kk, {1, kExpectedHeadCount, kRwkv7HeadSize}, block + ".att.kk_reshape");
        kk = add_l2_normalize(
            kk,
            {1, kExpectedHeadCount, kRwkv7HeadSize},
            2,
            block + ".att.kk_norm");
        kk = add_reshape(kk, head_vector_shape, block + ".att.kk_output_reshape");

        const uint32_t a_minus_one = add_sub(
            a,
            add_f16_channel_constant(vector_shape, 1.0F),
            vector_shape,
            block + ".att.a_minus_one");
        const uint32_t key_adjustment = add_mul(
            a_minus_one,
            file_vector(attention + ".k_a"),
            vector_shape,
            block + ".att.key_adjustment");
        const uint32_t key_factor = add_add(
            key_adjustment,
            add_f16_channel_constant(vector_shape, 1.0F),
            vector_shape,
            block + ".att.key_factor");
        key = add_mul(key, key_factor, vector_shape, block + ".att.key_control");

        if (layer == 0) {
            value_first = value;
        } else {
            const SafetensorsTensorInfo *v1_info = file_.get_tensor_info(attention + ".v1");
            if (v1_info == nullptr || v1_info->shape.size() != 2) {
                throw std::runtime_error("invalid value residual rank in " + attention);
            }
            const int32_t value_rank = static_cast<int32_t>(v1_info->shape[0]);
            const uint32_t v_hidden = add_linear(
                xv,
                attention + ".v1",
                {1, value_rank},
                block + ".att.v1");
            const uint32_t v_projected = add_linear(
                v_hidden,
                attention + ".v2",
                vector_shape,
                block + ".att.v2");
            const uint32_t v_with_bias = add_add(
                v_projected,
                file_vector(attention + ".v0"),
                vector_shape,
                block + ".att.v0");
            const uint32_t value_mix = add_unary(
                OH_NN_OPS_SIGMOID,
                v_with_bias,
                vector_shape,
                block + ".att.v_sigmoid");
            const uint32_t value_delta = add_sub(
                value_first,
                value,
                vector_shape,
                block + ".att.value_delta");
            value = add_add(
                value,
                add_mul(value_delta, value_mix, vector_shape, block + ".att.value_mix_mul"),
                vector_shape,
                block + ".att.value_mix_add");
        }

        const uint32_t flattened_time_decay = add_reshape(
            time_decay_vector,
            {kExpectedHiddenSize},
            block + ".att.decay_reshape");
        const uint32_t diagonal_time_decay = add_scatter_nd(
            wkv_diagonal_indices(),
            flattened_time_decay,
            wkv_shape_tensor(),
            wkv_shape,
            block + ".att.decay_diagonal");
        const uint32_t key_row = add_reshape(key, head_row_shape, block + ".att.key_reshape");
        const uint32_t value_column = add_reshape(value, head_column_shape, block + ".att.value_reshape");
        const uint32_t receptance_column = add_reshape(
            receptance,
            head_column_shape,
            block + ".att.receptance_reshape");
        const uint32_t a_column = add_unary(
            OH_NN_OPS_NEG,
            add_reshape(kk, head_column_shape, block + ".att.a_reshape"),
            head_column_shape,
            block + ".att.a_neg");
        const uint32_t b_row = add_reshape(
            add_mul(kk, add_reshape(a, head_vector_shape, block + ".att.a_vector"), head_vector_shape, block + ".att.b_mul"),
            head_row_shape,
            block + ".att.b_reshape");

        const uint32_t decayed_state = add_matmul(
            wkv_state,
            diagonal_time_decay,
            wkv_shape,
            false,
            false,
            block + ".att.state_decay");
        const uint32_t state_a = add_matmul(
            wkv_state,
            a_column,
            head_column_shape,
            false,
            false,
            block + ".att.state_a");
        const uint32_t state_ab = add_matmul(
            state_a,
            b_row,
            wkv_shape,
            false,
            false,
            block + ".att.state_ab");
        const uint32_t value_key = add_matmul(
            value_column,
            key_row,
            wkv_shape,
            false,
            false,
            block + ".att.value_key");
        const uint32_t state_with_ab = add_add(
            decayed_state,
            state_ab,
            wkv_shape,
            block + ".att.state_add_ab");
        const uint32_t wkv_state_output = add_add(
            state_with_ab,
            value_key,
            wkv_shape,
            block + ".att.state_add_vk");
        uint32_t attention_output = add_matmul(
            wkv_state_output,
            receptance_column,
            head_column_shape,
            false,
            false,
            block + ".att.state_receptance");
        attention_output = add_reshape(
            attention_output,
            {1, kExpectedHeadCount, kRwkv7HeadSize},
            block + ".att.group_reshape");
        attention_output = add_normalize(
            attention_output,
            {1, kExpectedHeadCount, kRwkv7HeadSize},
            2,
            add_file_tensor(attention + ".ln_x.weight", {1, kExpectedHeadCount, kRwkv7HeadSize}),
            add_file_tensor(attention + ".ln_x.bias", {1, kExpectedHeadCount, kRwkv7HeadSize}),
            kGroupNormEpsilon,
            block + ".att.group_norm");
        attention_output = add_reshape(
            attention_output,
            head_vector_shape,
            block + ".att.group_output_reshape");

        const uint32_t receptance_heads = add_reshape(
            receptance,
            head_vector_shape,
            block + ".att.receptance_heads");
        const uint32_t key_heads = add_reshape(key, head_vector_shape, block + ".att.key_heads");
        const uint32_t value_heads = add_reshape(value, head_vector_shape, block + ".att.value_heads");
        const uint32_t receptance_key = add_mul(
            receptance_heads,
            key_heads,
            head_vector_shape,
            block + ".att.receptance_key");
        const uint32_t weighted_receptance_key = add_mul(
            receptance_key,
            add_file_tensor(attention + ".r_k"),
            head_vector_shape,
            block + ".att.time_first_weight");
        const uint32_t time_first_scalar = add_reduce_sum(
            weighted_receptance_key,
            1,
            {kExpectedHeadCount, 1},
            block + ".att.time_first_sum");
        const uint32_t time_first = add_mul(
            time_first_scalar,
            value_heads,
            head_vector_shape,
            block + ".att.time_first_value");
        attention_output = add_add(
            attention_output,
            time_first,
            head_vector_shape,
            block + ".att.time_first_add");
        attention_output = add_reshape(
            attention_output,
            vector_shape,
            block + ".att.output_reshape");
        attention_output = add_mul(
            attention_output,
            gate,
            vector_shape,
            block + ".att.gate");
        attention_output = add_linear(
            attention_output,
            attention + ".output.weight",
            vector_shape,
            block + ".att.output");
        uint32_t x = add_add(input, attention_output, vector_shape, block + ".att.residual");

        const uint32_t ffn_normalized = add_layer_norm(
            x,
            vector_shape,
            add_file_tensor(block + ".ln2.weight"),
            add_file_tensor(block + ".ln2.bias"),
            kLayerNormEpsilon,
            block + ".ffn.layer_norm");
        const uint32_t ffn_delta = add_sub(
            ffn_state,
            ffn_normalized,
            vector_shape,
            block + ".ffn.shift_delta");
        const uint32_t ffn_mixed = add_add(
            ffn_normalized,
            add_mul(
                ffn_delta,
                file_vector(block + ".ffn.x_k"),
                vector_shape,
                block + ".ffn.mix_mul"),
            vector_shape,
            block + ".ffn.mix_add");
        uint32_t ffn_key = add_linear(
            ffn_mixed,
            block + ".ffn.key.weight",
            {1, kExpectedFfnSize},
            block + ".ffn.key");
        ffn_key = add_unary(
            OH_NN_OPS_RELU,
            ffn_key,
            {1, kExpectedFfnSize},
            block + ".ffn.relu");
        ffn_key = add_mul(
            ffn_key,
            ffn_key,
            {1, kExpectedFfnSize},
            block + ".ffn.square");
        const uint32_t ffn_value = add_linear(
            ffn_key,
            block + ".ffn.value.weight",
            vector_shape,
            block + ".ffn.value");
        x = add_add(x, ffn_value, vector_shape, block + ".ffn.residual");

        state_outputs.push_back(attention_normalized);
        state_outputs.push_back(wkv_state_output);
        state_outputs.push_back(ffn_normalized);
        return x;
    }

    uint32_t add_layer_sequence(
        int layer,
        uint32_t input,
        uint32_t attention_state,
        uint32_t wkv_state,
        uint32_t ffn_state,
        uint32_t &value_first,
        std::vector<uint32_t> &state_outputs) {
        const std::string block = "blocks." + std::to_string(layer);
        const std::string attention = block + ".att";
        const std::string prefix = "prefill." + block;
        const Shape sequence_shape = {kPrefillSequenceLength, kExpectedHiddenSize};
        const Shape vector_shape = {1, kExpectedHiddenSize};
        const Shape sequence_head_shape = {
            kPrefillSequenceLength, kExpectedHeadCount, kRwkv7HeadSize};
        const Shape sequence_head_column_shape = {
            kPrefillSequenceLength, kExpectedHeadCount, 1};
        const Shape head_column_shape = {kExpectedHeadCount, kRwkv7HeadSize, 1};
        const Shape head_row_shape = {kExpectedHeadCount, 1, kRwkv7HeadSize};
        const Shape wkv_shape = {kExpectedHeadCount, kRwkv7HeadSize, kRwkv7HeadSize};

        const uint32_t attention_normalized = add_layer_norm(
            input,
            sequence_shape,
            add_file_tensor(block + ".ln1.weight"),
            add_file_tensor(block + ".ln1.bias"),
            kLayerNormEpsilon,
            prefix + ".att.layer_norm");
        const uint32_t attention_past = add_shifted_sequence(
            attention_state,
            attention_normalized,
            sequence_shape,
            prefix + ".att.shift");
        const uint32_t shift_delta = add_sub(
            attention_past,
            attention_normalized,
            sequence_shape,
            prefix + ".att.shift_delta");

        const auto token_mix = [&](const std::string &suffix) {
            const uint32_t mixed_delta = add_mul(
                shift_delta,
                file_vector(attention + ".x_" + suffix),
                sequence_shape,
                prefix + ".att.mix_" + suffix + ".mul");
            return add_add(
                attention_normalized,
                mixed_delta,
                sequence_shape,
                prefix + ".att.mix_" + suffix + ".add");
        };

        const uint32_t xr = token_mix("r");
        const uint32_t xw = token_mix("w");
        const uint32_t xk = token_mix("k");
        const uint32_t xv = token_mix("v");
        const uint32_t xa = token_mix("a");
        const uint32_t xg = token_mix("g");

        const uint32_t receptance = add_linear(
            xr,
            attention + ".receptance.weight",
            sequence_shape,
            prefix + ".att.receptance");
        uint32_t key = add_linear(
            xk,
            attention + ".key.weight",
            sequence_shape,
            prefix + ".att.key");
        uint32_t value = add_linear(
            xv,
            attention + ".value.weight",
            sequence_shape,
            prefix + ".att.value");

        const uint32_t w_hidden = add_linear(
            xw,
            attention + ".w1",
            {kPrefillSequenceLength, 64},
            prefix + ".att.w1");
        const uint32_t w_tanh = add_unary(
            OH_NN_OPS_TANH,
            w_hidden,
            {kPrefillSequenceLength, 64},
            prefix + ".att.w_tanh");
        const uint32_t w_projected = add_linear(
            w_tanh,
            attention + ".w2",
            sequence_shape,
            prefix + ".att.w2");
        const uint32_t w_with_bias = add_add(
            w_projected,
            file_vector(attention + ".w0"),
            sequence_shape,
            prefix + ".att.w0");
        const uint32_t w_sigmoid = add_unary(
            OH_NN_OPS_SIGMOID,
            w_with_bias,
            sequence_shape,
            prefix + ".att.w_sigmoid");
        const uint32_t w_scaled = add_mul(
            w_sigmoid,
            add_f16_channel_constant({1, kExpectedHiddenSize}, kTimeDecayScale),
            sequence_shape,
            prefix + ".att.w_scale");
        const uint32_t time_decay_vector = add_exp(
            w_scaled,
            sequence_shape,
            prefix + ".att.w_exp");

        const uint32_t a_hidden = add_linear(
            xa,
            attention + ".a1",
            {kPrefillSequenceLength, 64},
            prefix + ".att.a1");
        const uint32_t a_projected = add_linear(
            a_hidden,
            attention + ".a2",
            sequence_shape,
            prefix + ".att.a2");
        const uint32_t a_with_bias = add_add(
            a_projected,
            file_vector(attention + ".a0"),
            sequence_shape,
            prefix + ".att.a0");
        const uint32_t a = add_unary(
            OH_NN_OPS_SIGMOID,
            a_with_bias,
            sequence_shape,
            prefix + ".att.a_sigmoid");

        const uint32_t g_hidden = add_linear(
            xg,
            attention + ".g1",
            {kPrefillSequenceLength, 128},
            prefix + ".att.g1");
        const uint32_t g_sigmoid = add_unary(
            OH_NN_OPS_SIGMOID,
            g_hidden,
            {kPrefillSequenceLength, 128},
            prefix + ".att.g_sigmoid");
        const uint32_t gate = add_linear(
            g_sigmoid,
            attention + ".g2",
            sequence_shape,
            prefix + ".att.g2");

        uint32_t kk = add_mul(
            key,
            file_vector(attention + ".k_k"),
            sequence_shape,
            prefix + ".att.kk_scale");
        kk = add_reshape(kk, sequence_head_shape, prefix + ".att.kk_reshape");
        kk = add_l2_normalize(
            kk,
            sequence_head_shape,
            2,
            prefix + ".att.kk_norm");

        const uint32_t a_minus_one = add_sub(
            a,
            add_f16_channel_constant({1, kExpectedHiddenSize}, 1.0F),
            sequence_shape,
            prefix + ".att.a_minus_one");
        const uint32_t key_adjustment = add_mul(
            a_minus_one,
            file_vector(attention + ".k_a"),
            sequence_shape,
            prefix + ".att.key_adjustment");
        const uint32_t key_factor = add_add(
            key_adjustment,
            add_f16_channel_constant({1, kExpectedHiddenSize}, 1.0F),
            sequence_shape,
            prefix + ".att.key_factor");
        key = add_mul(key, key_factor, sequence_shape, prefix + ".att.key_control");

        if (layer == 0) {
            value_first = value;
        } else {
            const SafetensorsTensorInfo *v1_info = file_.get_tensor_info(attention + ".v1");
            if (v1_info == nullptr || v1_info->shape.size() != 2) {
                throw std::runtime_error("invalid value residual rank in " + attention);
            }
            const int32_t value_rank = static_cast<int32_t>(v1_info->shape[0]);
            const uint32_t v_hidden = add_linear(
                xv,
                attention + ".v1",
                {kPrefillSequenceLength, value_rank},
                prefix + ".att.v1");
            const uint32_t v_projected = add_linear(
                v_hidden,
                attention + ".v2",
                sequence_shape,
                prefix + ".att.v2");
            const uint32_t v_with_bias = add_add(
                v_projected,
                file_vector(attention + ".v0"),
                sequence_shape,
                prefix + ".att.v0");
            const uint32_t value_mix = add_unary(
                OH_NN_OPS_SIGMOID,
                v_with_bias,
                sequence_shape,
                prefix + ".att.v_sigmoid");
            const uint32_t value_delta = add_sub(
                value_first,
                value,
                sequence_shape,
                prefix + ".att.value_delta");
            value = add_add(
                value,
                add_mul(
                    value_delta,
                    value_mix,
                    sequence_shape,
                    prefix + ".att.value_mix_mul"),
                sequence_shape,
                prefix + ".att.value_mix_add");
        }

        const uint32_t key_heads = add_reshape(
            key, sequence_head_shape, prefix + ".att.key_heads");
        const uint32_t value_heads = add_reshape(
            value, sequence_head_shape, prefix + ".att.value_heads");
        const uint32_t receptance_heads = add_reshape(
            receptance, sequence_head_shape, prefix + ".att.receptance_heads");
        const uint32_t a_heads = add_reshape(
            a, sequence_head_shape, prefix + ".att.a_heads");
        const uint32_t negative_kk = add_unary(
            OH_NN_OPS_NEG,
            kk,
            sequence_head_shape,
            prefix + ".att.negative_kk");
        const uint32_t b_heads = add_mul(
            kk,
            a_heads,
            sequence_head_shape,
            prefix + ".att.b_heads");

        uint32_t current_state = wkv_state;
        std::vector<uint32_t> attention_rows;
        attention_rows.reserve(kPrefillSequenceLength);
        for (int32_t token_index = 0; token_index < kPrefillSequenceLength; ++token_index) {
            const std::string token_prefix =
                prefix + ".att.wkv." + std::to_string(token_index);
            uint32_t decay_row = add_gather_row(
                time_decay_vector,
                token_index,
                vector_shape,
                token_prefix + ".decay_row");
            decay_row = add_reshape(
                decay_row,
                {kExpectedHeadCount, 1, kRwkv7HeadSize},
                token_prefix + ".decay_broadcast");
            decay_row = add_concat(
                std::vector<uint32_t>(kRwkv7HeadSize, decay_row),
                1,
                wkv_shape,
                token_prefix + ".decay_expand");

            const uint32_t key_row = add_reshape(
                add_gather_row(
                    key_heads,
                    token_index,
                    {1, kExpectedHeadCount, kRwkv7HeadSize},
                    token_prefix + ".key_row"),
                head_row_shape,
                token_prefix + ".key_row_reshape");
            const uint32_t value_column = add_reshape(
                add_gather_row(
                    value_heads,
                    token_index,
                    {1, kExpectedHeadCount, kRwkv7HeadSize},
                    token_prefix + ".value_row"),
                head_column_shape,
                token_prefix + ".value_column");
            const uint32_t receptance_column = add_reshape(
                add_gather_row(
                    receptance_heads,
                    token_index,
                    {1, kExpectedHeadCount, kRwkv7HeadSize},
                    token_prefix + ".receptance_row"),
                head_column_shape,
                token_prefix + ".receptance_column");
            const uint32_t a_column = add_reshape(
                add_gather_row(
                    negative_kk,
                    token_index,
                    {1, kExpectedHeadCount, kRwkv7HeadSize},
                    token_prefix + ".a_row"),
                head_column_shape,
                token_prefix + ".a_column");
            const uint32_t b_row = add_reshape(
                add_gather_row(
                    b_heads,
                    token_index,
                    {1, kExpectedHeadCount, kRwkv7HeadSize},
                    token_prefix + ".b_row"),
                head_row_shape,
                token_prefix + ".b_row_reshape");

            const uint32_t decayed_state = add_mul(
                current_state,
                decay_row,
                wkv_shape,
                token_prefix + ".state_decay");
            const uint32_t state_a = add_matmul(
                current_state,
                a_column,
                head_column_shape,
                false,
                false,
                token_prefix + ".state_a");
            const uint32_t state_ab = add_matmul(
                state_a,
                b_row,
                wkv_shape,
                false,
                false,
                token_prefix + ".state_ab");
            const uint32_t value_key = add_matmul(
                value_column,
                key_row,
                wkv_shape,
                false,
                false,
                token_prefix + ".value_key");
            current_state = add_add(
                add_add(
                    decayed_state,
                    state_ab,
                    wkv_shape,
                    token_prefix + ".state_add_ab"),
                value_key,
                wkv_shape,
                token_prefix + ".state_add_vk");
            uint32_t attention_row = add_matmul(
                current_state,
                receptance_column,
                head_column_shape,
                false,
                false,
                token_prefix + ".state_receptance");
            attention_row = add_reshape(
                attention_row,
                {1, kExpectedHeadCount, kRwkv7HeadSize},
                token_prefix + ".output_row");
            attention_rows.push_back(attention_row);
        }

        uint32_t attention_output = add_concat(
            attention_rows,
            0,
            sequence_head_shape,
            prefix + ".att.wkv.output_concat");
        attention_output = add_normalize(
            attention_output,
            sequence_head_shape,
            2,
            add_file_tensor(
                attention + ".ln_x.weight", {1, kExpectedHeadCount, kRwkv7HeadSize}),
            add_file_tensor(
                attention + ".ln_x.bias", {1, kExpectedHeadCount, kRwkv7HeadSize}),
            kGroupNormEpsilon,
            prefix + ".att.group_norm");

        const uint32_t receptance_key = add_mul(
            receptance_heads,
            key_heads,
            sequence_head_shape,
            prefix + ".att.receptance_key");
        const uint32_t weighted_receptance_key = add_mul(
            receptance_key,
            add_file_tensor(attention + ".r_k"),
            sequence_head_shape,
            prefix + ".att.time_first_weight");
        const uint32_t time_first_scalar = add_reduce_sum(
            weighted_receptance_key,
            2,
            sequence_head_column_shape,
            prefix + ".att.time_first_sum");
        const uint32_t time_first = add_mul(
            time_first_scalar,
            value_heads,
            sequence_head_shape,
            prefix + ".att.time_first_value");
        attention_output = add_add(
            attention_output,
            time_first,
            sequence_head_shape,
            prefix + ".att.time_first_add");
        attention_output = add_reshape(
            attention_output,
            sequence_shape,
            prefix + ".att.output_reshape");
        attention_output = add_mul(
            attention_output,
            gate,
            sequence_shape,
            prefix + ".att.gate");
        attention_output = add_linear(
            attention_output,
            attention + ".output.weight",
            sequence_shape,
            prefix + ".att.output");
        uint32_t x = add_add(
            input, attention_output, sequence_shape, prefix + ".att.residual");

        const uint32_t ffn_normalized = add_layer_norm(
            x,
            sequence_shape,
            add_file_tensor(block + ".ln2.weight"),
            add_file_tensor(block + ".ln2.bias"),
            kLayerNormEpsilon,
            prefix + ".ffn.layer_norm");
        const uint32_t ffn_past = add_shifted_sequence(
            ffn_state,
            ffn_normalized,
            sequence_shape,
            prefix + ".ffn.shift");
        const uint32_t ffn_delta = add_sub(
            ffn_past,
            ffn_normalized,
            sequence_shape,
            prefix + ".ffn.shift_delta");
        const uint32_t ffn_mixed = add_add(
            ffn_normalized,
            add_mul(
                ffn_delta,
                file_vector(block + ".ffn.x_k"),
                sequence_shape,
                prefix + ".ffn.mix_mul"),
            sequence_shape,
            prefix + ".ffn.mix_add");
        uint32_t ffn_key = add_linear(
            ffn_mixed,
            block + ".ffn.key.weight",
            {kPrefillSequenceLength, kExpectedFfnSize},
            prefix + ".ffn.key");
        ffn_key = add_unary(
            OH_NN_OPS_RELU,
            ffn_key,
            {kPrefillSequenceLength, kExpectedFfnSize},
            prefix + ".ffn.relu");
        ffn_key = add_mul(
            ffn_key,
            ffn_key,
            {kPrefillSequenceLength, kExpectedFfnSize},
            prefix + ".ffn.square");
        const uint32_t ffn_value = add_linear(
            ffn_key,
            block + ".ffn.value.weight",
            sequence_shape,
            prefix + ".ffn.value");
        x = add_add(x, ffn_value, sequence_shape, prefix + ".ffn.residual");

        state_outputs.push_back(add_gather_row(
            attention_normalized,
            kPrefillSequenceLength - 1,
            vector_shape,
            prefix + ".att.state_output"));
        state_outputs.push_back(current_state);
        state_outputs.push_back(add_gather_row(
            ffn_normalized,
            kPrefillSequenceLength - 1,
            vector_shape,
            prefix + ".ffn.state_output"));
        return x;
    }

    OH_NNModel *model_ = nullptr;
    const SafetensorsFile &file_;
    uint32_t tensor_count_ = 0;
    std::deque<std::vector<uint8_t>> owned_constants_;
    std::unordered_map<std::string, uint32_t> file_tensor_indices_;
    std::vector<uint32_t> inputs_;
    std::vector<uint32_t> outputs_;
    std::vector<std::string> operation_names_;
    std::vector<size_t> state_bytes_;
    uint32_t wkv_diagonal_indices_ = std::numeric_limits<uint32_t>::max();
    uint32_t wkv_shape_tensor_ = std::numeric_limits<uint32_t>::max();
};

struct RuntimeTensor {
    TensorDescPtr description;
    TensorPtr tensor;
    void *data = nullptr;
    size_t bytes = 0;
};

struct RuntimeGraph {
    std::string name;
    std::string cache_path;
    std::string profiling_path;
    ModelPtr model;
    CompilationPtr compilation;
    ExecutorPtr executor;
    std::vector<RuntimeTensor> inputs;
    std::vector<RuntimeTensor> outputs;
    std::vector<NN_Tensor *> input_handles;
    std::vector<NN_Tensor *> output_handles;
    std::vector<std::string> operation_names;
};

RuntimeTensor create_runtime_tensor(OH_NNExecutor *executor, size_t device_id, size_t index, bool input) {
    RuntimeTensor result;
    result.description.reset(input
        ? OH_NNExecutor_CreateInputTensorDesc(executor, index)
        : OH_NNExecutor_CreateOutputTensorDesc(executor, index));
    if (!result.description) {
        throw std::runtime_error("failed to create NNRT runtime tensor description");
    }
    result.tensor.reset(OH_NNTensor_Create(device_id, result.description.get()));
    if (!result.tensor) {
        throw std::runtime_error("failed to create NNRT runtime tensor");
    }
    result.data = OH_NNTensor_GetDataBuffer(result.tensor.get());
    if (result.data == nullptr) {
        throw std::runtime_error("failed to map NNRT runtime tensor");
    }
    require_success(OH_NNTensor_GetSize(result.tensor.get(), &result.bytes), "query NNRT tensor size");
    return result;
}

} // namespace

struct nnrt_backend::Impl {
    nnrt_backend_args args;
    std::string required_device_name = "kirin9020";
    std::string cache_dir;
    std::string profiling_output_dir;
    size_t device_id = static_cast<size_t>(-1);
    std::string device_name;

    std::unique_ptr<SafetensorsFile> file;
    std::unique_ptr<RuntimeGraph> embedding_graph;
    std::vector<std::unique_ptr<RuntimeGraph>> layer_graphs;
    std::unique_ptr<RuntimeGraph> head_graph;
    std::unique_ptr<RuntimeGraph> prefill_embedding_graph;
    std::vector<std::unique_ptr<RuntimeGraph>> prefill_layer_graphs;
    std::vector<size_t> state_bytes;
    std::atomic<float> load_progress {0.0F};
    std::atomic<double> prefill_speed {-1.0};
    std::atomic<double> decode_speed {-1.0};
    bool diagnostics_emitted = false;

    void copy_args(void *extra) {
        if (extra == nullptr) {
            args = nnrt_backend_args {};
        } else {
            const nnrt_backend_args *provided = static_cast<const nnrt_backend_args *>(extra);
            if (provided->struct_version != nnrt_backend_args::kVersion) {
                throw std::runtime_error("unsupported nnrt_backend_args version");
            }
            args = *provided;
        }
        required_device_name = args.required_device_name == nullptr ? "kirin9020" : args.required_device_name;
        cache_dir = args.cache_dir == nullptr ? "" : args.cache_dir;
        profiling_output_dir = args.profiling_output_dir == nullptr ? "" : args.profiling_output_dir;
        if (!args.strict_accelerator) {
            throw std::runtime_error("NNRT backend requires strict_accelerator=true");
        }
    }

    void select_device() {
        const size_t *device_ids = nullptr;
        uint32_t device_count = 0;
        require_success(OH_NNDevice_GetAllDevicesID(&device_ids, &device_count), "enumerate NNRT devices");
        if (device_ids == nullptr || device_count == 0) {
            throw std::runtime_error("NNRT returned no devices");
        }

        const bool explicit_id = args.device_id != static_cast<size_t>(-1);
        for (uint32_t index = 0; index < device_count; ++index) {
            const size_t candidate = device_ids[index];
            if (explicit_id && candidate != args.device_id) {
                continue;
            }
            OH_NN_DeviceType type = OH_NN_OTHERS;
            if (OH_NNDevice_GetType(candidate, &type) != OH_NN_SUCCESS || type != OH_NN_ACCELERATOR) {
                continue;
            }
            const char *name = nullptr;
            if (OH_NNDevice_GetName(candidate, &name) != OH_NN_SUCCESS || name == nullptr) {
                continue;
            }
            if (!device_name_matches(name, required_device_name)) {
                continue;
            }
            device_id = candidate;
            device_name = name;
            LOGI("[nnrt] selected accelerator id=%zu name=%s", device_id, device_name.c_str());
            return;
        }
        throw std::runtime_error(
            "required Kirin NNRT accelerator was not found: " + required_device_name);
    }

    void validate_model_info() {
        const SafetensorsTensorInfo *embedding = file->get_tensor_info("emb.weight");
        const SafetensorsTensorInfo *head = file->get_tensor_info("head.weight");
        const SafetensorsTensorInfo *ffn = file->get_tensor_info("blocks.0.ffn.key.weight");
        const SafetensorsTensorInfo *heads = file->get_tensor_info("blocks.0.att.r_k");
        if (embedding == nullptr || head == nullptr || ffn == nullptr || heads == nullptr) {
            throw std::runtime_error("safetensors file is missing RWKV7 model metadata tensors");
        }
        if (embedding->shape != std::vector<int64_t>({kExpectedVocabSize, kExpectedHiddenSize}) ||
            head->shape != std::vector<int64_t>({kExpectedVocabSize, kExpectedHiddenSize}) ||
            ffn->shape != std::vector<int64_t>({kExpectedFfnSize, kExpectedHiddenSize}) ||
            heads->shape != std::vector<int64_t>({kExpectedHeadCount, kRwkv7HeadSize}) ||
            file->has_tensor("blocks.12.ln1.weight")) {
            throw std::runtime_error("NNRT v1 only accepts RWKV7 0.1B L12-D768 F16 safetensors");
        }
        for (int layer = 0; layer < kExpectedLayers; ++layer) {
            if (!file->has_tensor("blocks." + std::to_string(layer) + ".ln1.weight")) {
                throw std::runtime_error("missing RWKV7 layer " + std::to_string(layer));
            }
        }
    }

    void validate_operation_support(const RuntimeGraph &graph) {
        const bool *supported = nullptr;
        uint32_t operation_count = 0;
        require_success(
            OH_NNModel_GetAvailableOperations(
                graph.model.get(),
                device_id,
                &supported,
                &operation_count),
            "query Kirin NNRT operation support for " + graph.name);
        if (operation_count == 0 && supported == nullptr) {
            LOGW(
                "[nnrt] accelerator %s returned no capability bitmap for %s; compilation and NPU profiling remain mandatory",
                device_name.c_str(),
                graph.name.c_str());
            return;
        }
        if (supported == nullptr || operation_count != graph.operation_names.size()) {
            throw std::runtime_error(
                "NNRT operation support result for " + graph.name +
                " has unexpected length: got " +
                std::to_string(operation_count) + ", expected " +
                std::to_string(graph.operation_names.size()));
        }
        bool all_supported = true;
        for (uint32_t index = 0; index < operation_count; ++index) {
            if (supported[index]) {
                continue;
            }
            all_supported = false;
            LOGE(
                "[nnrt] accelerator rejected %s op[%u]=%s",
                graph.name.c_str(),
                index,
                graph.operation_names[index].c_str());
        }
        if (!all_supported) {
            throw std::runtime_error(
                "Kirin accelerator does not support the complete " + graph.name + " graph");
        }
    }

    int configure_compilation(RuntimeGraph &graph, const std::string &model_path) {
        graph.compilation.reset(OH_NNCompilation_Construct(graph.model.get()));
        if (!graph.compilation) {
            LOGE("[nnrt] failed to construct %s compilation", graph.name.c_str());
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        OH_NN_ReturnCode code = OH_NNCompilation_SetDevice(graph.compilation.get(), device_id);
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] bind %s compilation to Kirin accelerator failed with NNRT error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }

        HiAI_ExecuteDevice devices[] = {HIAI_EXECUTE_DEVICE_NPU};
        code = HMS_HiAIOptions_SetModelDeviceOrder(graph.compilation.get(), devices, 1);
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] set %s CANN device order failed with NNRT error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        code = HMS_HiAIOptions_SetFallbackMode(
            graph.compilation.get(),
            HIAI_FALLBACK_DISABLED);
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] disable %s CANN fallback failed with NNRT error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }

        code = OH_NNCompilation_SetPerformanceMode(
            graph.compilation.get(),
            OH_NN_PERFORMANCE_EXTREME);
        if (code != OH_NN_SUCCESS) {
            LOGW(
                "[nnrt] set %s extreme performance mode failed with NNRT error %d; trying high mode",
                graph.name.c_str(),
                static_cast<int>(code));
            code = OH_NNCompilation_SetPerformanceMode(
                graph.compilation.get(),
                OH_NN_PERFORMANCE_HIGH);
        }
        if (code != OH_NN_SUCCESS) {
            LOGW(
                "[nnrt] set %s high performance mode failed with NNRT error %d; using device default",
                graph.name.c_str(),
                static_cast<int>(code));
        }

        code = OH_NNCompilation_SetPriority(
            graph.compilation.get(),
            OH_NN_PRIORITY_HIGH);
        if (code != OH_NN_SUCCESS) {
            LOGW(
                "[nnrt] set %s high priority failed with NNRT error %d; using device default",
                graph.name.c_str(),
                static_cast<int>(code));
        }

        if (cache_dir.empty()) {
            cache_dir = model_path + ".nnrt-cache";
        }
        const std::string schema_dir =
            "schema-" + std::to_string(kGraphCacheSchemaVersion);
        graph.cache_path = cache_dir + "/" + schema_dir + "/" + graph.name;
        std::filesystem::create_directories(graph.cache_path);
        code = OH_NNCompilation_SetCache(
            graph.compilation.get(),
            graph.cache_path.c_str(),
            args.cache_version);
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] configure %s compilation cache failed with NNRT error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }

        if (args.enable_profiling) {
            if (profiling_output_dir.empty()) {
                profiling_output_dir = cache_dir + "/profiling";
            }
            graph.profiling_path =
                profiling_output_dir + "/" + schema_dir + "/" + graph.name;
            std::filesystem::create_directories(graph.profiling_path);
            code = HMS_HiAIOptions_SetOmOptions(
                graph.compilation.get(),
                HIAI_OM_TYPE_PROFILING,
                graph.profiling_path.c_str());
            if (code != OH_NN_SUCCESS) {
                LOGE(
                    "[nnrt] enable %s CANN profiling failed with NNRT error %d",
                    graph.name.c_str(),
                    static_cast<int>(code));
                return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
            }
        }

        code = OH_NNCompilation_Build(graph.compilation.get());
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] compile %s graph on Kirin NPU failed with NNRT error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        return RWKV_SUCCESS;
    }

    void create_executor_tensors(
        RuntimeGraph &graph,
        size_t input_count,
        size_t output_count) {
        graph.executor.reset(OH_NNExecutor_Construct(graph.compilation.get()));
        if (!graph.executor) {
            throw std::runtime_error("failed to construct NNRT executor for " + graph.name);
        }

        graph.inputs.reserve(input_count);
        graph.outputs.reserve(output_count);
        graph.input_handles.reserve(input_count);
        graph.output_handles.reserve(output_count);
        for (size_t index = 0; index < input_count; ++index) {
            graph.inputs.push_back(
                create_runtime_tensor(graph.executor.get(), device_id, index, true));
            graph.input_handles.push_back(graph.inputs.back().tensor.get());
        }
        for (size_t index = 0; index < output_count; ++index) {
            graph.outputs.push_back(
                create_runtime_tensor(graph.executor.get(), device_id, index, false));
            graph.output_handles.push_back(graph.outputs.back().tensor.get());
        }
    }

    int run_graph(RuntimeGraph &graph) {
        if (!graph.executor) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        const OH_NN_ReturnCode code = OH_NNExecutor_RunSync(
            graph.executor.get(),
            graph.input_handles.data(),
            static_cast<uint32_t>(graph.input_handles.size()),
            graph.output_handles.data(),
            static_cast<uint32_t>(graph.output_handles.size()));
        if (code != OH_NN_SUCCESS) {
            LOGE(
                "[nnrt] %s RunSync failed with error %d",
                graph.name.c_str(),
                static_cast<int>(code));
            return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
        }
        return RWKV_SUCCESS;
    }

    void log_stage_stats(
        const std::string &name,
        const RuntimeTensor &tensor,
        size_t expected_count) {
        const size_t expected_bytes = expected_count * sizeof(half_float::half);
        if (tensor.data == nullptr || tensor.bytes != expected_bytes) {
            LOGE(
                "[nnrt-stage] %s has invalid buffer bytes=%zu expected=%zu",
                name.c_str(),
                tensor.bytes,
                expected_bytes);
            return;
        }
        const half_float::half *values =
            static_cast<const half_float::half *>(tensor.data);
        double sum = 0.0;
        double square_sum = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        float maximum_absolute = 0.0F;
        std::string first_values;
        const size_t preview_count = std::min<size_t>(8, expected_count);
        for (size_t index = 0; index < expected_count; ++index) {
            const float value = static_cast<float>(values[index]);
            sum += value;
            square_sum += static_cast<double>(value) * value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            maximum_absolute = std::max(maximum_absolute, std::fabs(value));
            if (index >= preview_count) {
                continue;
            }
            if (!first_values.empty()) {
                first_values += ",";
            }
            first_values += std::to_string(value);
        }
        LOGI(
            "[nnrt-stage] %s count=%zu sum=%.9f l2=%.9f min=%.9f max=%.9f max_abs=%.9f first8=%s",
            name.c_str(),
            expected_count,
            sum,
            std::sqrt(square_sum),
            minimum,
            maximum,
            maximum_absolute,
            first_values.c_str());
    }

    template <typename BuildGraph>
    std::unique_ptr<RuntimeGraph> compile_graph(
        const std::string &name,
        const std::string &model_path,
        BuildGraph build_graph) {
        auto graph = std::make_unique<RuntimeGraph>();
        graph->name = name;
        graph->model.reset(OH_NNModel_Construct());
        if (!graph->model) {
            throw std::runtime_error("failed to construct NNRT " + name + " model");
        }
        GraphBuilder builder(graph->model.get(), *file);
        GraphDefinition definition = build_graph(builder);
        graph->operation_names = std::move(definition.operation_names);
        validate_operation_support(*graph);
        const int compilation_result = configure_compilation(*graph, model_path);
        if (compilation_result != RWKV_SUCCESS) {
            return nullptr;
        }
        create_executor_tensors(*graph, definition.input_count, definition.output_count);
        if (!definition.state_bytes.empty() && state_bytes.empty()) {
            state_bytes = std::move(definition.state_bytes);
        }
        LOGI(
            "[nnrt] compiled %s with %zu operations",
            graph->name.c_str(),
            graph->operation_names.size());
        return graph;
    }

    int run_token(int token, Tensor1D &logits, bool compute_logits) {
        if (token < 0 || token >= kExpectedVocabSize ||
            !embedding_graph || layer_graphs.size() != kExpectedLayers || !head_graph) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        if (embedding_graph->inputs.size() != 1 || embedding_graph->outputs.empty() ||
            embedding_graph->outputs.size() > 2 ||
            embedding_graph->inputs[0].bytes != sizeof(int32_t) ||
            embedding_graph->outputs[0].bytes != static_cast<size_t>(kExpectedHiddenSize) * sizeof(uint16_t) ||
            (embedding_graph->outputs.size() == 2 &&
             embedding_graph->outputs[1].bytes != static_cast<size_t>(kExpectedHiddenSize) * sizeof(uint16_t))) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        *static_cast<int32_t *>(embedding_graph->inputs[0].data) = token;
        int result = run_graph(*embedding_graph);
        if (result != RWKV_SUCCESS) {
            return result;
        }

        const size_t vector_bytes = static_cast<size_t>(kExpectedHiddenSize) * sizeof(uint16_t);
        const bool log_stages = args.enable_stage_diagnostics && !diagnostics_emitted;
        if (log_stages) {
            if (embedding_graph->outputs.size() == 2) {
                log_stage_stats(
                    "embedding.raw_gather",
                    embedding_graph->outputs[1],
                    kExpectedHiddenSize);
            }
            log_stage_stats("embedding", embedding_graph->outputs[0], kExpectedHiddenSize);
        }
        const void *x = embedding_graph->outputs[0].data;
        const void *value_first = nullptr;
        for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
            RuntimeGraph &layer = *layer_graphs[static_cast<size_t>(layer_index)];
            const size_t expected_inputs = layer_index == 0 ? 4 : 5;
            const size_t expected_outputs = layer_index == 0 ? 5 : 4;
            if (layer.inputs.size() != expected_inputs ||
                layer.outputs.size() != expected_outputs ||
                layer.inputs[0].bytes != vector_bytes ||
                layer.outputs[0].bytes != vector_bytes) {
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
            }
            std::memcpy(layer.inputs[0].data, x, vector_bytes);
            if (layer_index > 0) {
                if (value_first == nullptr || layer.inputs[4].bytes != vector_bytes) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
                std::memcpy(layer.inputs[4].data, value_first, vector_bytes);
            }
            result = run_graph(layer);
            if (result != RWKV_SUCCESS) {
                return result;
            }
            if (log_stages) {
                log_stage_stats(
                    "layer-" + std::to_string(layer_index),
                    layer.outputs[0],
                    kExpectedHiddenSize);
            }
            if (layer_index == 0) {
                if (layer.outputs[4].bytes != vector_bytes) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
                value_first = layer.outputs[4].data;
            }
            x = layer.outputs[0].data;
        }

        if (compute_logits) {
            if (head_graph->inputs.size() != 1 || head_graph->outputs.size() != 1 ||
                head_graph->inputs[0].bytes != vector_bytes ||
                head_graph->outputs[0].bytes != static_cast<size_t>(kExpectedVocabSize) * sizeof(uint16_t)) {
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
            }
            std::memcpy(head_graph->inputs[0].data, x, vector_bytes);
            result = run_graph(*head_graph);
            if (result != RWKV_SUCCESS) {
                return result;
            }
            if (log_stages) {
                log_stage_stats("head", head_graph->outputs[0], kExpectedVocabSize);
                diagnostics_emitted = true;
            }
        }
        if (state_bytes.size() != 3) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        for (const std::unique_ptr<RuntimeGraph> &layer : layer_graphs) {
            for (size_t state_index = 0; state_index < state_bytes.size(); ++state_index) {
                if (layer->inputs[state_index + 1].bytes != state_bytes[state_index] ||
                    layer->outputs[state_index + 1].bytes != state_bytes[state_index]) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
            }
        }
        for (const std::unique_ptr<RuntimeGraph> &layer : layer_graphs) {
            for (size_t state_index = 0; state_index < state_bytes.size(); ++state_index) {
                std::memcpy(
                    layer->inputs[state_index + 1].data,
                    layer->outputs[state_index + 1].data,
                    state_bytes[state_index]);
            }
        }
        if (compute_logits) {
            logits = Tensor1D::make(
                head_graph->outputs[0].data,
                TensorDType::F16,
                kExpectedVocabSize);
        }
        return RWKV_SUCCESS;
    }

    int run_prefill_sequence(const int *tokens, Tensor1D &logits, bool compute_logits) {
        if (tokens == nullptr ||
            !prefill_embedding_graph ||
            prefill_layer_graphs.size() != kExpectedLayers ||
            layer_graphs.size() != kExpectedLayers ||
            !head_graph ||
            state_bytes.size() != 3) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        const size_t vector_bytes =
            static_cast<size_t>(kExpectedHiddenSize) * sizeof(uint16_t);
        const size_t sequence_bytes =
            static_cast<size_t>(kPrefillSequenceLength) * vector_bytes;
        if (prefill_embedding_graph->inputs.size() != 1 ||
            prefill_embedding_graph->outputs.size() != 1 ||
            prefill_embedding_graph->inputs[0].bytes !=
                static_cast<size_t>(kPrefillSequenceLength) * sizeof(int32_t) ||
            prefill_embedding_graph->outputs[0].bytes != sequence_bytes) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        std::memcpy(
            prefill_embedding_graph->inputs[0].data,
            tokens,
            static_cast<size_t>(kPrefillSequenceLength) * sizeof(int32_t));
        int result = run_graph(*prefill_embedding_graph);
        if (result != RWKV_SUCCESS) {
            return result;
        }

        const void *x = prefill_embedding_graph->outputs[0].data;
        const void *value_first = nullptr;
        for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
            RuntimeGraph &prefill_layer =
                *prefill_layer_graphs[static_cast<size_t>(layer_index)];
            RuntimeGraph &decode_layer =
                *layer_graphs[static_cast<size_t>(layer_index)];
            const size_t expected_inputs = layer_index == 0 ? 4 : 5;
            const size_t expected_outputs = layer_index == 0 ? 5 : 4;
            if (prefill_layer.inputs.size() != expected_inputs ||
                prefill_layer.outputs.size() != expected_outputs ||
                prefill_layer.inputs[0].bytes != sequence_bytes ||
                prefill_layer.outputs[0].bytes != sequence_bytes ||
                decode_layer.inputs.size() < 4) {
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
            }
            std::memcpy(prefill_layer.inputs[0].data, x, sequence_bytes);
            for (size_t state_index = 0; state_index < state_bytes.size(); ++state_index) {
                if (prefill_layer.inputs[state_index + 1].bytes != state_bytes[state_index] ||
                    prefill_layer.outputs[state_index + 1].bytes != state_bytes[state_index] ||
                    decode_layer.inputs[state_index + 1].bytes != state_bytes[state_index]) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
                std::memcpy(
                    prefill_layer.inputs[state_index + 1].data,
                    decode_layer.inputs[state_index + 1].data,
                    state_bytes[state_index]);
            }
            if (layer_index > 0) {
                if (value_first == nullptr || prefill_layer.inputs[4].bytes != sequence_bytes) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
                std::memcpy(prefill_layer.inputs[4].data, value_first, sequence_bytes);
            }
            result = run_graph(prefill_layer);
            if (result != RWKV_SUCCESS) {
                return result;
            }
            if (layer_index == 0) {
                if (prefill_layer.outputs[4].bytes != sequence_bytes) {
                    return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
                }
                value_first = prefill_layer.outputs[4].data;
            }
            x = prefill_layer.outputs[0].data;
        }

        if (compute_logits) {
            if (head_graph->inputs.size() != 1 || head_graph->outputs.size() != 1 ||
                head_graph->inputs[0].bytes != vector_bytes ||
                head_graph->outputs[0].bytes !=
                    static_cast<size_t>(kExpectedVocabSize) * sizeof(uint16_t)) {
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
            }
            const uint8_t *last_x = static_cast<const uint8_t *>(x) +
                static_cast<size_t>(kPrefillSequenceLength - 1) * vector_bytes;
            std::memcpy(head_graph->inputs[0].data, last_x, vector_bytes);
            result = run_graph(*head_graph);
            if (result != RWKV_SUCCESS) {
                return result;
            }
        }

        for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
            RuntimeGraph &prefill_layer =
                *prefill_layer_graphs[static_cast<size_t>(layer_index)];
            RuntimeGraph &decode_layer =
                *layer_graphs[static_cast<size_t>(layer_index)];
            for (size_t state_index = 0; state_index < state_bytes.size(); ++state_index) {
                std::memcpy(
                    decode_layer.inputs[state_index + 1].data,
                    prefill_layer.outputs[state_index + 1].data,
                    state_bytes[state_index]);
            }
        }
        if (compute_logits) {
            logits = Tensor1D::make(
                head_graph->outputs[0].data,
                TensorDType::F16,
                kExpectedVocabSize);
        }
        return RWKV_SUCCESS;
    }
};

nnrt_backend::nnrt_backend() : impl_(std::make_unique<Impl>()) {
    supported_batch_sizes = {1};
}

nnrt_backend::~nnrt_backend() {
    release_model();
}

int nnrt_backend::init(void *extra) {
    try {
        impl_->copy_args(extra);
        impl_->select_device();
        return RWKV_SUCCESS;
    } catch (const std::exception &error) {
        LOGE("[nnrt] init failed: %s", error.what());
        return RWKV_ERROR_INIT | RWKV_ERROR_BACKEND;
    } catch (...) {
        LOGE("[nnrt] init failed with an unknown native exception");
        return RWKV_ERROR_INIT | RWKV_ERROR_BACKEND;
    }
}

int nnrt_backend::load_model(std::string model_path, void *extra) {
    release_model();
    impl_->load_progress = 0.0F;
    try {
        const int init_result = init(extra);
        if (init_result != RWKV_SUCCESS) {
            return init_result;
        }

        impl_->file = std::make_unique<SafetensorsFile>(model_path);
        impl_->validate_model_info();
        impl_->load_progress = 0.05F;

        impl_->embedding_graph = impl_->compile_graph(
            "embedding",
            model_path,
            [](GraphBuilder &builder) { return builder.build_embedding(); });
        if (!impl_->embedding_graph) {
            release_model();
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        impl_->load_progress = 0.10F;

        impl_->layer_graphs.reserve(kExpectedLayers);
        for (int layer = 0; layer < kExpectedLayers; ++layer) {
            std::unique_ptr<RuntimeGraph> graph = impl_->compile_graph(
                "layer-" + std::to_string(layer),
                model_path,
                [layer](GraphBuilder &builder) { return builder.build_layer(layer); });
            if (!graph) {
                release_model();
                return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
            }
            impl_->layer_graphs.push_back(std::move(graph));
            impl_->load_progress = 0.10F +
                0.35F * static_cast<float>(layer + 1) / static_cast<float>(kExpectedLayers);
        }

        impl_->head_graph = impl_->compile_graph(
            "head",
            model_path,
            [](GraphBuilder &builder) { return builder.build_head(); });
        if (!impl_->head_graph) {
            release_model();
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        impl_->load_progress = 0.50F;

        impl_->prefill_embedding_graph = impl_->compile_graph(
            "prefill-embedding-16",
            model_path,
            [](GraphBuilder &builder) { return builder.build_embedding_sequence(); });
        if (!impl_->prefill_embedding_graph) {
            release_model();
            return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
        }
        impl_->load_progress = 0.55F;

        impl_->prefill_layer_graphs.reserve(kExpectedLayers);
        for (int layer = 0; layer < kExpectedLayers; ++layer) {
            std::unique_ptr<RuntimeGraph> graph = impl_->compile_graph(
                "prefill-layer-16-" + std::to_string(layer),
                model_path,
                [layer](GraphBuilder &builder) {
                    return builder.build_layer_sequence(layer);
                });
            if (!graph) {
                release_model();
                return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
            }
            impl_->prefill_layer_graphs.push_back(std::move(graph));
            impl_->load_progress = 0.55F +
                0.40F * static_cast<float>(layer + 1) / static_cast<float>(kExpectedLayers);
        }

        n_layers = kExpectedLayers;
        hidden_size = kExpectedHiddenSize;
        num_heads = kExpectedHeadCount;
        vocab_size = kExpectedVocabSize;
        version = 7;
        supported_batch_sizes = {1};
        zero_state();
        impl_->load_progress = 1.0F;
        size_t operation_count = impl_->embedding_graph->operation_names.size() +
                                 impl_->head_graph->operation_names.size();
        for (const std::unique_ptr<RuntimeGraph> &graph : impl_->layer_graphs) {
            operation_count += graph->operation_names.size();
        }
        operation_count += impl_->prefill_embedding_graph->operation_names.size();
        for (const std::unique_ptr<RuntimeGraph> &graph : impl_->prefill_layer_graphs) {
            operation_count += graph->operation_names.size();
        }
        LOGI(
            "[nnrt] loaded RWKV7 0.1B F16 with decode and T=16 prefill as 27 NNRT graphs with %zu operations bound to %s; verify driver partitions separately",
            operation_count,
            impl_->device_name.c_str());
        return RWKV_SUCCESS;
    } catch (const std::exception &error) {
        LOGE("[nnrt] model load failed: %s", error.what());
        release_model();
        return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
    } catch (...) {
        LOGE("[nnrt] model load failed with an unknown native exception");
        release_model();
        return RWKV_ERROR_MODEL | RWKV_ERROR_BACKEND;
    }
}

float nnrt_backend::get_load_progress() const {
    return impl_->load_progress.load();
}

int nnrt_backend::eval(int id, Tensor1D &logits) {
    const auto start = std::chrono::steady_clock::now();
    const int result = impl_->run_token(id, logits, true);
    const auto end = std::chrono::steady_clock::now();
    if (result == RWKV_SUCCESS) {
        const int64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (microseconds > 0) {
            impl_->decode_speed = 1000000.0 / static_cast<double>(microseconds);
        }
    }
    return result;
}

int nnrt_backend::eval(std::vector<int> ids, Tensor1D &logits) {
    if (ids.empty()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto update_live_prefill_speed = [&](size_t processed_tokens) {
        const auto now = std::chrono::steady_clock::now();
        const int64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        if (processed_tokens > 0 && microseconds > 0) {
            impl_->prefill_speed.store(
                static_cast<double>(processed_tokens) * 1000000.0 /
                static_cast<double>(microseconds),
                std::memory_order_relaxed);
        }
    };
    size_t index = 0;
    for (; index + kPrefillSequenceLength <= ids.size();
         index += kPrefillSequenceLength) {
        const bool compute_logits = index + kPrefillSequenceLength == ids.size();
        const int result = impl_->run_prefill_sequence(
            ids.data() + index, logits, compute_logits);
        if (result != RWKV_SUCCESS) {
            return result;
        }
        update_live_prefill_speed(index + kPrefillSequenceLength);
    }
    for (; index < ids.size(); ++index) {
        const bool compute_logits = index + 1 == ids.size();
        const int result = impl_->run_token(ids[index], logits, compute_logits);
        if (result != RWKV_SUCCESS) {
            return result;
        }
        update_live_prefill_speed(index + 1);
    }
    const auto end = std::chrono::steady_clock::now();
    const int64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (microseconds > 0) {
        impl_->prefill_speed.store(
            static_cast<double>(ids.size()) * 1000000.0 / static_cast<double>(microseconds),
            std::memory_order_relaxed);
    }
    return RWKV_SUCCESS;
}

int nnrt_backend::get_state(std::any &state) {
    if (impl_->layer_graphs.size() != kExpectedLayers || impl_->state_bytes.size() != 3) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto snapshot = std::make_shared<StateData>(
        static_cast<size_t>(kExpectedLayers) * impl_->state_bytes.size());
    for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
        const RuntimeGraph &layer = *impl_->layer_graphs[static_cast<size_t>(layer_index)];
        if (layer.inputs.size() < 4) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        for (size_t state_index = 0; state_index < impl_->state_bytes.size(); ++state_index) {
            const size_t snapshot_index =
                static_cast<size_t>(layer_index) * impl_->state_bytes.size() + state_index;
            (*snapshot)[snapshot_index].resize(impl_->state_bytes[state_index]);
            std::memcpy(
                (*snapshot)[snapshot_index].data(),
                layer.inputs[state_index + 1].data,
                impl_->state_bytes[state_index]);
        }
    }
    state = std::move(snapshot);
    return RWKV_SUCCESS;
}

int nnrt_backend::set_state(std::any state) {
    if (impl_->layer_graphs.size() != kExpectedLayers ||
        impl_->state_bytes.size() != 3 || !state.has_value()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::shared_ptr<StateData> snapshot;
    try {
        snapshot = std::any_cast<std::shared_ptr<StateData>>(state);
    } catch (const std::bad_any_cast &) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    const size_t expected_count =
        static_cast<size_t>(kExpectedLayers) * impl_->state_bytes.size();
    if (!snapshot || snapshot->size() != expected_count) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
        const RuntimeGraph &layer = *impl_->layer_graphs[static_cast<size_t>(layer_index)];
        if (layer.inputs.size() < 4) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        for (size_t state_index = 0; state_index < impl_->state_bytes.size(); ++state_index) {
            const size_t snapshot_index =
                static_cast<size_t>(layer_index) * impl_->state_bytes.size() + state_index;
            if ((*snapshot)[snapshot_index].size() != impl_->state_bytes[state_index] ||
                layer.inputs[state_index + 1].bytes != impl_->state_bytes[state_index]) {
                return RWKV_ERROR_INVALID_PARAMETERS;
            }
        }
    }
    for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
        RuntimeGraph &layer = *impl_->layer_graphs[static_cast<size_t>(layer_index)];
        for (size_t state_index = 0; state_index < impl_->state_bytes.size(); ++state_index) {
            const size_t snapshot_index =
                static_cast<size_t>(layer_index) * impl_->state_bytes.size() + state_index;
            std::memcpy(
                layer.inputs[state_index + 1].data,
                (*snapshot)[snapshot_index].data(),
                impl_->state_bytes[state_index]);
        }
    }
    return RWKV_SUCCESS;
}

int nnrt_backend::free_state(std::any state) {
    state.reset();
    return RWKV_SUCCESS;
}

int nnrt_backend::zero_state() {
    if (impl_->layer_graphs.size() != kExpectedLayers || impl_->state_bytes.size() != 3) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (const std::unique_ptr<RuntimeGraph> &layer : impl_->layer_graphs) {
        if (!layer || layer->inputs.size() < 4) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
        }
        for (size_t state_index = 0; state_index < impl_->state_bytes.size(); ++state_index) {
            if (layer->inputs[state_index + 1].bytes != impl_->state_bytes[state_index]) {
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_BACKEND;
            }
        }
    }
    for (const std::unique_ptr<RuntimeGraph> &layer : impl_->layer_graphs) {
        for (size_t state_index = 0; state_index < impl_->state_bytes.size(); ++state_index) {
            std::memset(
                layer->inputs[state_index + 1].data,
                0,
                impl_->state_bytes[state_index]);
        }
    }
    if (impl_->args.enable_stage_diagnostics) {
        impl_->diagnostics_emitted = false;
    }
    return RWKV_SUCCESS;
}

int nnrt_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    if (impl_->layer_graphs.size() != kExpectedLayers ||
        impl_->state_bytes.size() != 3 ||
        states.size() != static_cast<size_t>(kExpectedLayers)) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    const size_t wkv_elements = static_cast<size_t>(kExpectedHeadCount) *
                                kRwkv7HeadSize * kRwkv7HeadSize;
    for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
        const RuntimeGraph &layer = *impl_->layer_graphs[static_cast<size_t>(layer_index)];
        if (layer.inputs.size() < 4 ||
            states[static_cast<size_t>(layer_index)].size() != wkv_elements ||
            layer.inputs[1].bytes != impl_->state_bytes[0] ||
            layer.inputs[2].bytes != impl_->state_bytes[1] ||
            layer.inputs[3].bytes != impl_->state_bytes[2]) {
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
    }
    for (int layer_index = 0; layer_index < kExpectedLayers; ++layer_index) {
        RuntimeGraph &layer = *impl_->layer_graphs[static_cast<size_t>(layer_index)];
        std::memset(layer.inputs[1].data, 0, impl_->state_bytes[0]);
        std::memcpy(
            layer.inputs[2].data,
            states[static_cast<size_t>(layer_index)].data(),
            impl_->state_bytes[1]);
        std::memset(layer.inputs[3].data, 0, impl_->state_bytes[2]);
    }
    return RWKV_SUCCESS;
}

int nnrt_backend::release_model() {
    impl_->prefill_layer_graphs.clear();
    impl_->prefill_embedding_graph.reset();
    impl_->head_graph.reset();
    impl_->layer_graphs.clear();
    impl_->embedding_graph.reset();
    impl_->file.reset();
    impl_->state_bytes.clear();
    impl_->load_progress = 0.0F;
    impl_->diagnostics_emitted = false;
    return RWKV_SUCCESS;
}

int nnrt_backend::release() {
    return release_model();
}

bool nnrt_backend::is_available() {
    try {
        impl_->select_device();
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

double nnrt_backend::get_prefill_speed() {
    return impl_->prefill_speed.load(std::memory_order_relaxed);
}

double nnrt_backend::get_decode_speed() {
    return impl_->decode_speed.load(std::memory_order_relaxed);
}

void nnrt_backend::reset_speed_stats() {
    impl_->prefill_speed.store(-1.0, std::memory_order_relaxed);
    impl_->decode_speed.store(-1.0, std::memory_order_relaxed);
}

} // namespace rwkvmobile
