#ifndef RWKV_MOBILE_NNRT_SAFETENSORS_FILE_H
#define RWKV_MOBILE_NNRT_SAFETENSORS_FILE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rwkvmobile {

enum class SafetensorsDType {
    F16,
    F32,
    BF16,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    BOOL,
};

struct SafetensorsTensorInfo {
    std::string name;
    SafetensorsDType dtype = SafetensorsDType::F16;
    std::vector<int64_t> shape;
    uint64_t data_offset = 0;
    uint64_t data_size = 0;
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string &path);
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile &) = delete;
    SafetensorsFile &operator=(const SafetensorsFile &) = delete;

    bool has_tensor(const std::string &name) const;
    const SafetensorsTensorInfo *get_tensor_info(const std::string &name) const;
    const void *get_tensor_data(const std::string &name) const;
    std::vector<std::string> tensor_names() const;

    uint64_t header_size() const;
    uint64_t data_offset() const;
    uint64_t file_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

size_t safetensors_dtype_size(SafetensorsDType dtype);
const char *safetensors_dtype_name(SafetensorsDType dtype);

} // namespace rwkvmobile

#endif
