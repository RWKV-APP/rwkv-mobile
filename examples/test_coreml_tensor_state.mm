#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#import <objc/runtime.h>
#include "rwkv-coreml.mm"

using ModelLoader = MLModel *(*)(id, SEL, NSURL *, MLModelConfiguration *, NSError **);
static ModelLoader original_loader;
static MLModel *require_ne(id object, SEL selector, NSURL *url, MLModelConfiguration *config, NSError **error) {
    assert(config.computeUnits == MLComputeUnitsCPUAndNeuralEngine);
    return original_loader(object, selector, url, config, error);
}

static void save_finite(const void *bytes, size_t count, NSString *path) {
    assert(bytes != nullptr);
    const auto *values = static_cast<const half_float::half *>(bytes);
    for (size_t i = 0; i < count; ++i) assert(std::isfinite(float(values[i])));
    assert([[NSData dataWithBytes:bytes length:count * sizeof(*values)] writeToFile:path atomically:YES]);
}

// A real-model check: a GPU fallback must fail this NE regression test.
int main(int argc, char **argv) {
    assert(argc == 5);
    setvbuf(stdout, nullptr, _IONBF, 0);
    @autoreleasepool {
        Method method = class_getClassMethod([MLModel class], @selector(modelWithContentsOfURL:configuration:error:));
        assert(method != nullptr);
        original_loader = reinterpret_cast<ModelLoader>(method_setImplementation(method, reinterpret_cast<IMP>(require_ne)));
        auto *ctx = rwkv_coreml_new_context();
        assert(rwkv_coreml_init(ctx, argv[1], 0, 0) == 0);
        assert(ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR);
        assert(rwkv_coreml_is_prefill_ready(ctx));
        for (const auto &models : {ctx->model_decode, ctx->model_prefill}) {
            for (const auto *ptr : models) {
                MLModel *model = (__bridge MLModel *)ptr;
                assert(model.configuration.computeUnits == MLComputeUnitsCPUAndNeuralEngine);
            }
        }
        NSString *out = [NSString stringWithUTF8String:argv[3]];
        assert([[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil]);
        const bool prefill = std::string(argv[2]) == "prefill";
        const int width = prefill ? rwkv_coreml_get_prefill_seq_length(ctx) : 1;
        for (int step = 0; step < std::stoi(argv[4]); ++step) {
            std::vector<int> tokens;
            for (int j = 0; j < width; ++j) tokens.push_back(((step * width + j) * 997 + 261) % 65535);
            void *logits = prefill ? rwkv_coreml_prefill(ctx, tokens) : rwkv_coreml_decode(ctx, tokens[0]);
            save_finite(logits, ctx->vocab_size, [out stringByAppendingPathComponent:[NSString stringWithFormat:@"%d-logits.bin", step]]);
            for (int chunk = 0; chunk < ctx->num_chunks; ++chunk) {
                MLMultiArray *wkv = external_state_wkv(ctx, chunk);
                MLMultiArray *shift = external_state_tokenshift(ctx, chunk);
                save_finite(wkv.dataPointer, wkv.count, [out stringByAppendingPathComponent:[NSString stringWithFormat:@"%d-%d-wkv.bin", step, chunk]]);
                save_finite(shift.dataPointer, shift.count, [out stringByAppendingPathComponent:[NSString stringWithFormat:@"%d-%d-shift.bin", step, chunk]]);
                if (step == 0) NSLog(@"chunk=%d wkv shape=%@ strides=%@ shift shape=%@ strides=%@", chunk, wkv.shape, wkv.strides, shift.shape, shift.strides);
            }
            printf("NATIVE_NE_STEP %d %s\n", step, argv[2]);
        }
        const auto saved = rwkv_coreml_get_state(ctx);
        rwkv_coreml_zero_state(ctx);
        for (const auto &part : rwkv_coreml_get_state(ctx)) {
            for (const auto byte : part) assert(byte == 0);
        }
        rwkv_coreml_set_state(ctx, saved);
        assert(rwkv_coreml_get_state(ctx) == saved);
        const auto *next = static_cast<const half_float::half *>(rwkv_coreml_decode(ctx, 261));
        assert(next != nullptr);
        std::vector<half_float::half> first(next, next + ctx->vocab_size);
        rwkv_coreml_set_state(ctx, saved);
        next = static_cast<const half_float::half *>(rwkv_coreml_decode(ctx, 261));
        assert(next != nullptr && std::memcmp(first.data(), next, first.size() * sizeof(*next)) == 0);
        for (const auto value : first) assert(std::isfinite(float(value)));
        rwkv_coreml_free(ctx);
        puts("NATIVE_TENSOR_NE_OK");
    }
}
