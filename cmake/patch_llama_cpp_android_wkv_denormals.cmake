function(rwkv_mobile_patch_llama_cpp_android_wkv_denormals source_file)
    file(READ "${source_file}" source_text)
    string(REPLACE "\r\n" "\n" source_text "${source_text}")
    string(SHA256 source_hash "${source_text}")
    if(source_hash STREQUAL "1cd228bc9de0ae53afff23ef73f9243af346748e883e31f1adabc9e53418dc0b")
        return()
    endif()
    if(NOT source_hash STREQUAL "d044623eb06b9a93b216ab98eb15012df7776c5a496719a1cb21eb041eecc446")
        message(FATAL_ERROR "Pinned Android WKV7 CPU source differs; retain original file")
    endif()
    string(REPLACE [====[static void ggml_compute_forward_rwkv_wkv7_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {]====] [====[static void ggml_compute_forward_rwkv_wkv7_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    #if defined(__ANDROID__) && defined(__aarch64__)
        // Decayed RWKV7 state can trigger slow denormal arithmetic on mobile cores.
        struct ScopedWkvFtz {
            uint64_t saved;
            ScopedWkvFtz() {
                asm volatile("mrs %0, fpcr" : "=r"(saved));
                const uint64_t enabled = saved | (uint64_t(1) << 24);
                asm volatile("msr fpcr, %0" :: "r"(enabled) : "memory");
            }
            ~ScopedWkvFtz() {
                asm volatile("msr fpcr, %0" :: "r"(saved) : "memory");
            }
        } scoped_wkv_ftz;
    #endif
]====] source_text "${source_text}")
    string(SHA256 source_hash "${source_text}")
    if(NOT source_hash STREQUAL "1cd228bc9de0ae53afff23ef73f9243af346748e883e31f1adabc9e53418dc0b")
        message(FATAL_ERROR "Android WKV7 denormal patch verification failed")
    endif()
    file(CONFIGURE OUTPUT "${source_file}" CONTENT "${source_text}" @ONLY NEWLINE_STYLE LF)
endfunction()
