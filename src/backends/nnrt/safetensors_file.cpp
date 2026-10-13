#include "safetensors_file.h"

#include "json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rwkvmobile {
namespace {

constexpr uint64_t kLengthPrefixBytes = 8;
constexpr uint64_t kMaximumHeaderBytes = 64ULL * 1024ULL * 1024ULL;

uint64_t read_u64_le(const uint8_t *data) {
    uint64_t value = 0;
    for (size_t index = 0; index < sizeof(value); ++index) {
        value |= static_cast<uint64_t>(data[index]) << (index * 8);
    }
    return value;
}

uint64_t checked_numel(const std::vector<int64_t> &shape, const std::string &name) {
    uint64_t result = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("safetensors tensor has a negative dimension: " + name);
        }
        const uint64_t unsigned_dimension = static_cast<uint64_t>(dimension);
        if (unsigned_dimension != 0 && result > std::numeric_limits<uint64_t>::max() / unsigned_dimension) {
            throw std::runtime_error("safetensors tensor element count overflow: " + name);
        }
        result *= unsigned_dimension;
    }
    return result;
}

SafetensorsDType parse_dtype(const std::string &name) {
    if (name == "F16") return SafetensorsDType::F16;
    if (name == "F32") return SafetensorsDType::F32;
    if (name == "BF16") return SafetensorsDType::BF16;
    if (name == "I8") return SafetensorsDType::I8;
    if (name == "U8") return SafetensorsDType::U8;
    if (name == "I16") return SafetensorsDType::I16;
    if (name == "U16") return SafetensorsDType::U16;
    if (name == "I32") return SafetensorsDType::I32;
    if (name == "U32") return SafetensorsDType::U32;
    if (name == "I64") return SafetensorsDType::I64;
    if (name == "U64") return SafetensorsDType::U64;
    if (name == "BOOL") return SafetensorsDType::BOOL;
    throw std::runtime_error("unsupported safetensors dtype: " + name);
}

struct MappedFile {
    const uint8_t *data = nullptr;
    uint64_t size = 0;
#ifndef _WIN32
    int fd = -1;
#else
    std::vector<uint8_t> owned;
#endif

    explicit MappedFile(const std::string &path) {
#ifndef _WIN32
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open safetensors file: " + path);
        }
        struct stat file_stat {};
        if (fstat(fd, &file_stat) != 0 || file_stat.st_size < static_cast<off_t>(kLengthPrefixBytes)) {
            close(fd);
            fd = -1;
            throw std::runtime_error("failed to stat safetensors file: " + path);
        }
        size = static_cast<uint64_t>(file_stat.st_size);
        void *mapping = mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            close(fd);
            fd = -1;
            throw std::runtime_error("failed to mmap safetensors file: " + path);
        }
        data = static_cast<const uint8_t *>(mapping);
#else
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open safetensors file: " + path);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff end = input.tellg();
        if (end < static_cast<std::streamoff>(kLengthPrefixBytes)) {
            throw std::runtime_error("safetensors file is truncated: " + path);
        }
        size = static_cast<uint64_t>(end);
        input.seekg(0, std::ios::beg);
        owned.resize(static_cast<size_t>(size));
        input.read(reinterpret_cast<char *>(owned.data()), static_cast<std::streamsize>(owned.size()));
        if (!input) {
            throw std::runtime_error("failed to read safetensors file: " + path);
        }
        data = owned.data();
#endif
    }

    ~MappedFile() {
#ifndef _WIN32
        if (data != nullptr) {
            munmap(const_cast<uint8_t *>(data), static_cast<size_t>(size));
        }
        if (fd >= 0) {
            close(fd);
        }
#endif
    }
};

} // namespace

size_t safetensors_dtype_size(SafetensorsDType dtype) {
    switch (dtype) {
        case SafetensorsDType::F16:
        case SafetensorsDType::BF16:
        case SafetensorsDType::I16:
        case SafetensorsDType::U16:
            return 2;
        case SafetensorsDType::F32:
        case SafetensorsDType::I32:
        case SafetensorsDType::U32:
            return 4;
        case SafetensorsDType::I64:
        case SafetensorsDType::U64:
            return 8;
        case SafetensorsDType::I8:
        case SafetensorsDType::U8:
        case SafetensorsDType::BOOL:
            return 1;
    }
    return 0;
}

const char *safetensors_dtype_name(SafetensorsDType dtype) {
    switch (dtype) {
        case SafetensorsDType::F16: return "F16";
        case SafetensorsDType::F32: return "F32";
        case SafetensorsDType::BF16: return "BF16";
        case SafetensorsDType::I8: return "I8";
        case SafetensorsDType::U8: return "U8";
        case SafetensorsDType::I16: return "I16";
        case SafetensorsDType::U16: return "U16";
        case SafetensorsDType::I32: return "I32";
        case SafetensorsDType::U32: return "U32";
        case SafetensorsDType::I64: return "I64";
        case SafetensorsDType::U64: return "U64";
        case SafetensorsDType::BOOL: return "BOOL";
    }
    return "unknown";
}

struct SafetensorsFile::Impl {
    MappedFile file;
    uint64_t header_bytes = 0;
    uint64_t data_begin = 0;
    std::unordered_map<std::string, SafetensorsTensorInfo> tensors;

    explicit Impl(const std::string &path) : file(path) {
        header_bytes = read_u64_le(file.data);
        if (header_bytes == 0 || header_bytes > kMaximumHeaderBytes) {
            throw std::runtime_error("invalid safetensors header size: " + std::to_string(header_bytes));
        }
        if (header_bytes > file.size - kLengthPrefixBytes) {
            throw std::runtime_error("safetensors header extends beyond file: " + path);
        }
        data_begin = kLengthPrefixBytes + header_bytes;

        const std::string header(
            reinterpret_cast<const char *>(file.data + kLengthPrefixBytes),
            static_cast<size_t>(header_bytes));
        const nlohmann::json root = nlohmann::json::parse(header);
        if (!root.is_object()) {
            throw std::runtime_error("safetensors header must be a JSON object: " + path);
        }

        for (auto iterator = root.begin(); iterator != root.end(); ++iterator) {
            if (iterator.key() == "__metadata__") {
                continue;
            }
            const nlohmann::json &entry = iterator.value();
            if (!entry.is_object() || !entry.contains("dtype") ||
                !entry.contains("shape") || !entry.contains("data_offsets")) {
                throw std::runtime_error("invalid safetensors tensor entry: " + iterator.key());
            }

            SafetensorsTensorInfo info;
            info.name = iterator.key();
            info.dtype = parse_dtype(entry.at("dtype").get<std::string>());
            info.shape = entry.at("shape").get<std::vector<int64_t>>();
            const std::vector<uint64_t> offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[1] < offsets[0]) {
                throw std::runtime_error("invalid safetensors offsets: " + info.name);
            }
            info.data_offset = offsets[0];
            info.data_size = offsets[1] - offsets[0];

            if (info.data_offset > file.size - data_begin ||
                info.data_size > file.size - data_begin - info.data_offset) {
                throw std::runtime_error("safetensors tensor extends beyond file: " + info.name);
            }

            const uint64_t element_count = checked_numel(info.shape, info.name);
            const uint64_t element_size = safetensors_dtype_size(info.dtype);
            if (element_size != 0 && element_count > std::numeric_limits<uint64_t>::max() / element_size) {
                throw std::runtime_error("safetensors tensor byte count overflow: " + info.name);
            }
            if (element_count * element_size != info.data_size) {
                throw std::runtime_error("safetensors tensor byte count mismatch: " + info.name);
            }
            if (!tensors.emplace(info.name, std::move(info)).second) {
                throw std::runtime_error("duplicate safetensors tensor: " + iterator.key());
            }
        }
        if (tensors.empty()) {
            throw std::runtime_error("safetensors file contains no tensors: " + path);
        }
    }
};

SafetensorsFile::SafetensorsFile(const std::string &path) : impl_(std::make_unique<Impl>(path)) {}

SafetensorsFile::~SafetensorsFile() = default;

bool SafetensorsFile::has_tensor(const std::string &name) const {
    return impl_->tensors.find(name) != impl_->tensors.end();
}

const SafetensorsTensorInfo *SafetensorsFile::get_tensor_info(const std::string &name) const {
    const auto iterator = impl_->tensors.find(name);
    if (iterator == impl_->tensors.end()) {
        return nullptr;
    }
    return &iterator->second;
}

const void *SafetensorsFile::get_tensor_data(const std::string &name) const {
    const SafetensorsTensorInfo *info = get_tensor_info(name);
    if (info == nullptr) {
        return nullptr;
    }
    return impl_->file.data + impl_->data_begin + info->data_offset;
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->tensors.size());
    for (const auto &entry : impl_->tensors) {
        names.push_back(entry.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

uint64_t SafetensorsFile::header_size() const {
    return impl_->header_bytes;
}

uint64_t SafetensorsFile::data_offset() const {
    return impl_->data_begin;
}

uint64_t SafetensorsFile::file_size() const {
    return impl_->file.size;
}

} // namespace rwkvmobile
