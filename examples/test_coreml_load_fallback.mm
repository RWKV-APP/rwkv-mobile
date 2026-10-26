#include <cassert>
#include <cstdio>
#import <objc/runtime.h>
#include "../src/backends/coreml/rwkv-coreml.mm"

static std::vector<MLComputeUnits> attempts;
static int ne_success_at, gpu_success_at;
static NSString *expected_function;
static MLModel *stub_model;

static MLModel *stub_load(id, SEL, NSURL *, MLModelConfiguration *configuration, NSError **error) {
    assert([configuration.functionName isEqualToString:expected_function]);
    assert(configuration.allowLowPrecisionAccumulationOnGPU);
    const auto units = configuration.computeUnits;
    attempts.push_back(units);
    const int count = (int)std::count(attempts.begin(), attempts.end(), units);
    if ((units == MLComputeUnitsCPUAndNeuralEngine && count == ne_success_at) ||
        (units == MLComputeUnitsCPUAndGPU && count == gpu_success_at)) return stub_model;
    *error = [NSError errorWithDomain:@"CoreMLFallbackTest" code:11
                             userInfo:@{NSLocalizedDescriptionKey: @"original loader failure"}];
    return nil;
}

static void check(MLComputeUnits units, int ne_success, int gpu_success,
                  std::vector<MLComputeUnits> expected, bool succeeds, NSString *function = @"decode") {
    attempts.clear();
    ne_success_at = ne_success;
    gpu_success_at = gpu_success;
    expected_function = function;
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = units;
    configuration.functionName = function;
    configuration.allowLowPrecisionAccumulationOnGPU = YES;
    MLModel *model = load_coreml_model_with_retry([NSURL fileURLWithPath:@"/unused.mlmodelc"], configuration,
                                                @"unused.mlmodelc", function, 0, 1);
    assert((model == stub_model) == succeeds);
    assert(attempts == expected);
    assert(configuration.computeUnits == units);
    assert([configuration.functionName isEqualToString:function]);
}

int main() {
    @autoreleasepool {
        setenv("RWKV_COREML_LOAD_RETRY_ATTEMPTS", "2", 1);
        setenv("RWKV_COREML_LOAD_RETRY_DELAY_MS", "0", 1);
        stub_model = (MLModel *)[NSObject new];
        Method method = class_getClassMethod([MLModel class], @selector(modelWithContentsOfURL:configuration:error:));
        assert(method != nullptr);
        IMP original = method_setImplementation(method, (IMP)stub_load);
        const auto ne = MLComputeUnitsCPUAndNeuralEngine, gpu = MLComputeUnitsCPUAndGPU;
        check(ne, 1, 0, {ne}, true);
        check(ne, 2, 0, {ne, ne}, true);
        check(ne, 0, 1, {ne, ne, gpu}, true);
        check(ne, 0, 1, {ne, ne, gpu}, true, @"prefill");
        check(ne, 0, 0, {ne, ne, gpu, gpu}, false);
        check(gpu, 0, 0, {gpu, gpu}, false);
        check(MLComputeUnitsCPUOnly, 0, 0, {MLComputeUnitsCPUOnly, MLComputeUnitsCPUOnly}, false);
        check(MLComputeUnitsAll, 0, 0, {MLComputeUnitsAll, MLComputeUnitsAll}, false);
        method_setImplementation(method, original);
        assert(rwkvmobile::logger_get_log().find("original loader failure") != std::string::npos);
        assert(rwkvmobile::logger_get_log().find("compute_units=cpu-gpu") != std::string::npos);
        puts("CoreML load fallback: 8 cases passed");
    }
}
