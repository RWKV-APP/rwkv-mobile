// Standalone: c++ -O3 -std=c++17 -Isrc examples/test_sampler_argmax.cpp src/sampler.cpp -o test_sampler_argmax
#include "sampler.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

using half_float::half;
using namespace rwkvmobile;

int main() {
    NucleusSampler sampler;
    std::mt19937 random(42);
    auto check = [&](std::vector<half>& values, size_t offset, size_t size) {
        half* data = values.data() + offset;
        const int expected = size ? int(std::max_element(data, data + size) - data) : 0;
        auto tensor = Tensor1D::make(data, TensorDType::F16, size);
        assert(sampler.sample(tensor, size, 1.f, 1, .5f) == expected);
        assert(sampler.sample(tensor, size, 1.f, 128, 0.f) == expected);
    };

    // Every half encoding, including subnormals, infinities, signed zero and NaNs.
    std::vector<half> values(65537);
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint16_t raw = uint16_t(bits);
        std::memcpy(&values[bits + 1], &raw, sizeof(raw));
    }
    check(values, 1, 65536);
    for (int trial = 0; trial < 2000; ++trial) {
        const size_t offset = trial % 8;
        const size_t size = trial % 131;
        for (size_t i = 0; i < size; ++i) {
            const uint16_t raw = uint16_t(random());
            std::memcpy(&values[offset + i], &raw, sizeof(raw));
        }
        check(values, offset, size);
    }
    for (size_t size = 1; size <= 33; ++size) {
        std::fill(values.begin(), values.begin() + 40, half(0.f));
        check(values, 1, size); // Equal values must return the first index.
        values[1] = std::numeric_limits<half>::quiet_NaN();
        check(values, 1, size); // First NaN is retained by std::max_element.
    }
    for (const auto& encodings : std::vector<std::vector<uint16_t>>{
             {0x8000, 0x0000, 0x8000}, {0x0000, 0x8000, 0x0000},
             {0x8002, 0x8001, 0x8003}, {0x0001, 0x0003, 0x0002},
             {0xfc00, 0x7c01, 0xfbff, 0xfe00, 0x7c00, 0x7c00}}) {
        for (size_t i = 0; i < encodings.size(); ++i)
            std::memcpy(&values[i + 1], &encodings[i], sizeof(uint16_t));
        check(values, 1, encodings.size());
    }

    std::vector<half> batch(37 * 3, half(-1.f));
    batch[5] = batch[37 + 9] = batch[74 + 12] = half(1.f);
    auto batch_tensor = Tensor1D::make(batch.data(), TensorDType::F16, batch.size());
    assert((sampler.sample_batch(batch_tensor, 33, 37, 3, {1.f}, {1}, {0.f}) ==
            std::vector<int>{5, 9, 12}));

    for (half& value : values) value = half(float(int(random() % 20001) - 10000) / 1000.f);
    auto tensor = Tensor1D::make(values.data() + 1, TensorDType::F16, 65536);
    volatile int checksum = 0;
    for (int trial = 0; trial < 5; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) checksum += sampler.sample(tensor, 65536, 1.f, 1, 0.f);
        const auto end = std::chrono::steady_clock::now();
        std::cout << "f16_argmax_us="
                  << std::chrono::duration<double, std::micro>(end - start).count() / 1000
                  << '\n';
    }
    std::cout << "PASS exact reference, tails, ties, nonfinite values, batch; checksum=" << checksum << '\n';
}
