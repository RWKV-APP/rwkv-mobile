function(rwkv_mobile_enable_selective_avx512_cpu)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
            OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64|X64)$"
            OR CMAKE_CROSSCOMPILING
            OR NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "Selective AVX512 is an experimental native Linux x86-64 build option")
    endif()
    if(GGML_NATIVE OR GGML_AVX512 OR GGML_CPU_ALL_VARIANTS OR GGML_BACKEND_DL OR NOT TARGET ggml-cpu)
        message(FATAL_ERROR "Selective AVX512 requires the single portable ggml-cpu target and baseline global ISA flags")
    endif()
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("
        int main() {
            __builtin_cpu_init();
            return !(__builtin_cpu_supports(\"avx512f\") &&
                     __builtin_cpu_supports(\"avx512bw\") &&
                     __builtin_cpu_supports(\"avx512dq\") &&
                     __builtin_cpu_supports(\"avx512vl\") &&
                     __builtin_cpu_supports(\"avx512vnni\"));
        }
    " RWKV_MOBILE_HOST_SUPPORTS_SELECTIVE_AVX512)
    if(NOT RWKV_MOBILE_HOST_SUPPORTS_SELECTIVE_AVX512)
        message(FATAL_ERROR "The current host cannot execute the requested AVX512/VNNI matrix kernels")
    endif()
    get_target_property(cpu_source_dir ggml-cpu SOURCE_DIR)
    get_target_property(cpu_sources ggml-cpu SOURCES)
    set(cpu_source_paths)
    foreach(cpu_source IN LISTS cpu_sources)
        get_filename_component(cpu_source_path "${cpu_source}" ABSOLUTE BASE_DIR "${cpu_source_dir}")
        list(APPEND cpu_source_paths "${cpu_source_path}")
    endforeach()
    set(matrix_sources
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/hbm.cpp
        ggml-cpu/quants.c
        ggml-cpu/traits.cpp
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/arch/x86/quants.c
        ggml-cpu/arch/x86/repack.cpp)
    foreach(source IN LISTS matrix_sources)
        get_filename_component(source_path "${source}" ABSOLUTE BASE_DIR "${cpu_source_dir}")
        if(NOT source_path IN_LIST cpu_source_paths)
            message(FATAL_ERROR "The pinned CPU source layout changed: ${source}")
        endif()
        set_property(SOURCE "${source_path}" DIRECTORY "${cpu_source_dir}" APPEND PROPERTY
            COMPILE_DEFINITIONS GGML_AVX512 GGML_AVX512_VNNI)
        set_property(SOURCE "${source_path}" DIRECTORY "${cpu_source_dir}" APPEND PROPERTY
            COMPILE_OPTIONS -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni)
    endforeach()
    message(STATUS "Experimental host-specific matrix ISA enabled; vec/ops/unary/binary arithmetic keeps baseline flags")
endfunction()
