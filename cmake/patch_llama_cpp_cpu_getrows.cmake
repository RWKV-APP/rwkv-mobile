function(rwkv_mobile_patch_llama_cpp_cpu_getrows source_file)
    file(READ "${source_file}" source_text)
    file(READ "${source_file}" line_endings HEX)
    string(FIND "${line_endings}" "0d0a" crlf_at)
    if (NOT crlf_at EQUAL -1)
        string(REPLACE "\r\n" "\n" source_text "${source_text}")
    endif()
    string(FIND "${source_text}" "static void ggml_compute_forward_get_rows_f32(" function_start)
    string(FIND "${source_text}" "void ggml_compute_forward_get_rows(" function_end)
    if (function_start EQUAL -1 OR function_end LESS function_start)
        message(FATAL_ERROR "Pinned llama.cpp F32 row-copy function no longer matches")
    endif()
    math(EXPR function_length "${function_end} - ${function_start}")
    string(SUBSTRING "${source_text}" ${function_start} ${function_length} function_text)
    set(original_range "    const int ir0 = dr*ith;\n    const int ir1 = MIN(ir0 + dr, nr);")
    set(replacement_range [=[    // Large recurrent-state rows otherwise leave the existing workers idle.
    const bool split_columns = nc >= 16384 && nr < nth;
    const int ir0 = split_columns ? 0 : dr*ith;
    const int ir1 = split_columns ? (int)nr : MIN(ir0 + dr, nr);
    const int64_t col0 = split_columns ? nc * ith / nth : 0;
    const int64_t col1 = split_columns ? nc * (ith + 1) / nth : nc;]=])
    set(previous_range "${replacement_range}")
    string(REPLACE "    const bool split_columns = nc >= 16384 && nr < nth;"
        "#if defined(__x86_64__) || defined(_M_X64)\n    const bool split_columns = nc >= 16384 && nr < nth;\n#else\n    const bool split_columns = false;\n#endif"
        replacement_range "${replacement_range}")
    set(original_copy [=[        ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (float *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));]=])
    set(replacement_copy [=[        if (split_columns) {
            memcpy((float *)((char *)dst->data + i10*nb1 + i11*nb2 + i12*nb3) + col0,
                   (float *)((char *)src0->data + i01*nb01 + i11*nb02 + i12*nb03) + col0,
                   (size_t)(col1 - col0) * sizeof(float));
        } else {
            ggml_vec_cpy_f32(nc,
                    (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                    (float *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
        }]=])
    string(FIND "${function_text}" "${replacement_range}" patched_range)
    string(FIND "${function_text}" "${replacement_copy}" patched_copy)
    if (NOT patched_range EQUAL -1 AND NOT patched_copy EQUAL -1)
        return()
    endif()
    string(FIND "${function_text}" "${previous_range}" previous_at)
    if (NOT previous_at EQUAL -1 AND NOT patched_copy EQUAL -1 AND patched_range EQUAL -1)
        string(REPLACE "${previous_range}" "${replacement_range}" patched_function "${function_text}")
    else()
        string(FIND "${function_text}" "${original_range}" range_at)
        string(FIND "${function_text}" "${original_copy}" copy_at)
        if (range_at EQUAL -1 OR copy_at EQUAL -1 OR NOT patched_range EQUAL -1 OR NOT patched_copy EQUAL -1)
            message(FATAL_ERROR "Pinned llama.cpp F32 row-copy body no longer matches the reviewed patch")
        endif()
        string(REPLACE "${original_range}" "${replacement_range}" patched_function "${function_text}")
        string(REPLACE "${original_copy}" "${replacement_copy}" patched_function "${patched_function}")
    endif()
    string(REPLACE "${function_text}" "${patched_function}" patched_text "${source_text}")
    if (NOT crlf_at EQUAL -1)
        string(REPLACE "\n" "\r\n" patched_text "${patched_text}")
    endif()
    file(WRITE "${source_file}" "${patched_text}")
    message(STATUS "Applied x86-only large F32 row-copy optimization in pinned llama.cpp")
endfunction()
