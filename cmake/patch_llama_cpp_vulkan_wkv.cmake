# Split independent WKV7 value rows across two full32-lane workgroups.
function(rwkv_mobile_patch_llama_cpp_vulkan_wkv source_dir)
    file(READ "${source_dir}/ggml-vulkan.cpp" host)
    file(READ "${source_dir}/vulkan-shaders/wkv7.comp" shader)
    string(REPLACE "\r\n" "\n" host "${host}")
    string(REPLACE "\r\n" "\n" shader "${shader}")
    string(SHA256 host_hash "${host}")
    string(SHA256 shader_hash "${shader}")
    if(host_hash STREQUAL "c87e725dc6ac16760982b4f408a4dcabf2abda9e992d8422136e53ff001d43c3" AND shader_hash STREQUAL "216cc6fb676d6f35e61b52e1048afddd8eacc4029267fa2867be2767f6d190eb")
        return()
    endif()
    if(NOT host_hash STREQUAL "ac3470ce197444cf4269b09adc68f4391f9e2f92f27518f52091b17a0113be41" OR NOT shader_hash STREQUAL "184ad4a22b750b45c45e5e42dc04f9ad37f83a6c23b62a5d5c2b447d4099b496")
        message(FATAL_ERROR "Pinned Vulkan WKV7 source differs; retain original files")
    endif()
    string(REPLACE [====[        (uint32_t)(pc.B * pc.H),
        1,
        1
    };]====] [====[        (uint32_t)(pc.B * pc.H),
        version == 7 ? 2u : 1u,
        1
    };]====] host "${host}")
    string(REPLACE [====[#define BLOCK_SIZE 64]====] [====[#define BLOCK_SIZE 64
#define ROW_TILE 32]====] shader "${shader}")
    string(REPLACE [====[layout(local_size_x = BLOCK_SIZE,]====] [====[layout(local_size_x = ROW_TILE,]====] shader "${shader}")
    string(REPLACE [====[const uint tid = gl_LocalInvocationID.x;]====] [====[const uint tid = gl_WorkGroupID.y * ROW_TILE + gl_LocalInvocationID.x;]====] shader "${shader}")
    string(REPLACE [====[        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];]====] [====[        for (uint j = gl_LocalInvocationID.x; j < BLOCK_SIZE; j += ROW_TILE) {
            const uint coefficient = t - tid + j;
            _r[j] = r[coefficient];
            _w[j] = w[coefficient];
            _k[j] = k[coefficient];
            _a[j] = a[coefficient];
            _b[j] = b[coefficient];
        }]====] shader "${shader}")
    string(SHA256 host_hash "${host}")
    string(SHA256 shader_hash "${shader}")
    if(NOT host_hash STREQUAL "c87e725dc6ac16760982b4f408a4dcabf2abda9e992d8422136e53ff001d43c3" OR NOT shader_hash STREQUAL "216cc6fb676d6f35e61b52e1048afddd8eacc4029267fa2867be2767f6d190eb")
        message(FATAL_ERROR "Vulkan WKV7 tiling patch verification failed")
    endif()
    file(CONFIGURE OUTPUT "${source_dir}/ggml-vulkan.cpp" CONTENT "${host}" @ONLY NEWLINE_STYLE LF)
    file(CONFIGURE OUTPUT "${source_dir}/vulkan-shaders/wkv7.comp" CONTENT "${shader}" @ONLY NEWLINE_STYLE LF)
endfunction()
