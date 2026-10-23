#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include "commondef.h"
#include "c_api.h"
#include "qnn_backend.h"
namespace rwkvmobile { extern std::shared_ptr<qnn_backend_context> g_qnn_backend_context_ptr; }
using ContextFree = Qnn_ErrorHandle_t (*)(Qnn_ContextHandle_t, Qnn_ProfileHandle_t);
static ContextFree originalFree;
static bool inject = true;
static Qnn_ErrorHandle_t failOnce(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile) {
    if (inject) { inject = false; return QNN_CONTEXT_ERROR_MEM_ALLOC; }
    return originalFree(context, profile);
}
static void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    try {
        check(argc == 5, "model sdk tokenizer mode required");
        auto runtime = rwkvmobile_runtime_init();
        check(runtime != nullptr, "runtime init");
        const int model = rwkvmobile_runtime_load_model_with_extra(runtime, argv[1], "qnn", argv[3], argv[2]);
        check(model >= 0, "model load");
        auto& session = rwkvmobile::g_qnn_backend_context_ptr;
        check(session && session->ref_count == 1 && session->powerConfigActive, "active SDK ownership");
        if (std::strcmp(argv[4], "ownership") == 0) {
            auto* identity = session.get();
            check(rwkvmobile_runtime_release_model(runtime, model) == 0, "first model release");
            char names[1024] = {};
            check(rwkvmobile_runtime_get_available_backend_names(names, sizeof(names)) == 0, "backend inventory");
            check(session.get() == identity, "idle SDK session lost during temporary Runtime inventory");
            char small[1] = {};
            check(rwkvmobile_runtime_get_available_backend_names(small, sizeof(small)) != 0, "short inventory buffer rejected");
            check(session.get() == identity, "failed inventory retains SDK session");
            auto other = rwkvmobile_runtime_init();
            check(other != nullptr, "second runtime init");
            const int otherModel = rwkvmobile_runtime_load_model_with_extra(other, argv[1], "qnn", argv[3], argv[2]);
            check(otherModel >= 0 && session.get() == identity, "second runtime shares SDK session");
            check(rwkvmobile_runtime_release_model(other, otherModel) == 0, "second model release");
            check(rwkvmobile_runtime_release(runtime) == 0, "first runtime release");
            check(session.get() == identity, "remaining runtime retains idle SDK session");
            check(rwkvmobile_runtime_get_available_backend_names(names, sizeof(names)) == 0, "second backend inventory");
            check(session.get() == identity, "second inventory retains SDK session");
            check(rwkvmobile_runtime_release(other) == 0, "last runtime release");
            check(session.get() == identity && session->ref_count == 0 && !session->powerConfigActive, "process retains idle SDK session without model or power ownership");
            std::puts("QNN_RELEASE_FAULT_RETRY_PASSED mode=ownership");
            return 0;
        }
        originalFree = session->qnnFunctionPointers.qnnInterface.contextFree;
        session->qnnFunctionPointers.qnnInterface.contextFree = failOnce;
        if (std::strcmp(argv[4], "model") == 0) {
            check(rwkvmobile_runtime_release_model(runtime, model) != 0, "model release must report SDK failure");
            check(!inject && session && session->ref_count == 1 && session->powerConfigActive, "failed model retains SDK and power ownership");
            int token = 117;
            std::vector<float> logits(65536);
            check(rwkvmobile_runtime_eval_logits(runtime, model, &token, 1, logits.data(), logits.size()) != 0, "partly released model rejects inference");
            session->qnnFunctionPointers.qnnInterface.contextFree = originalFree;
            check(rwkvmobile_runtime_release_model(runtime, model) == 0, "model release retry");
            check(session && session->ref_count == 0 && !session->powerConfigActive, "retry releases power vote");
        } else {
            check(std::strcmp(argv[4], "runtime") == 0, "unknown mode");
            check(rwkvmobile_runtime_release(runtime) != 0, "runtime release must report SDK failure");
            check(!inject && session && session->ref_count == 1, "failed runtime remains valid for retry");
            session->qnnFunctionPointers.qnnInterface.contextFree = originalFree;
        }
        check(rwkvmobile_runtime_release(runtime) == 0, "runtime release retry");
        check(session && session->ref_count == 0 && !session->powerConfigActive, "process owns idle SDK session until orderly library shutdown");
        std::printf("QNN_RELEASE_FAULT_RETRY_PASSED mode=%s\n", argv[4]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "QNN_RELEASE_FAULT_FAILED %s\n", e.what());
        return 1;
    }
}
