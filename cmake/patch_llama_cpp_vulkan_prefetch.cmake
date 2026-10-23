function(rwkv_mobile_patch_llama_cpp_vulkan_prefetch source_file)
    file(READ "${source_file}" source)
    string(REPLACE "\r\n" "\n" source "${source}")
    string(SHA256 source_sha "${source}")
    if(source_sha STREQUAL "23571175c40d4cd4148c16a4e212e4bbf3a0717ad39d0133b0a9b513650f3b1c")
        return()
    endif()
    if(NOT source_sha STREQUAL "3ddfec5f0f4911fa316dfd276bb66946c1955446d3ea2562e6cba5381ec8462d")
        message(FATAL_ERROR "Unexpected llama model source for Windows Vulkan prefetch patch")
    endif()
    set(before [=[    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);]=])
    set(after [=[    bool prefetch_mmap = true;
    if (arch == LLM_ARCH_RWKV7 && ml.use_mmap && !use_mlock && !ml.check_tensors &&
        act_gpu_layers == n_layer + 1 && devices.size() == 1 &&
        std::strcmp(ggml_backend_dev_name(devices[0].dev), "Vulkan0") == 0) {
        prefetch_mmap = false;
        LLAMA_LOG_DEBUG("%s: full-GPU RWKV7 mapped prefetch disabled\n", __func__);
    }
    ml.init_mappings(prefetch_mmap, use_mlock ? &pimpl->mlock_mmaps : nullptr);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    string(SHA256 patched_sha "${source}")
    if(NOT patched_sha STREQUAL "23571175c40d4cd4148c16a4e212e4bbf3a0717ad39d0133b0a9b513650f3b1c")
        message(FATAL_ERROR "Windows Vulkan prefetch patch did not match the pinned source")
    endif()
    file(WRITE "${source_file}" "${source}")
endfunction()
