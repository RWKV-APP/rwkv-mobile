function(rwkv_mobile_patch_llama_cpp_vulkan_mmv source_file)
    file(READ "${source_file}" source)
    string(REPLACE "\r\n" "\n" source "${source}")
    string(SHA256 source_sha "${source}")
    if(source_sha STREQUAL "88d4476e5ae81a2b56f93d2ebb947427b36d042dddbe7b849ae180ea46bcdfa1")
        return()
    endif()
    if(NOT source_sha STREQUAL "ac3470ce197444cf4269b09adc68f4391f9e2f92f27518f52091b17a0113be41")
        message(FATAL_ERROR "Unexpected pinned Vulkan source for MMV row experiment")
    endif()
    set(before [=[    uint32_t rm_iq = 2 * rm_kq;]=])
    set(after [=[    uint32_t rm_iq = 2 * rm_kq;

    uint32_t rwkv_mmv_rows = 0;
    const char * rwkv_mmv_rows_env = getenv("RWKV_VULKAN_MMV_ROWS");
    if (rwkv_mmv_rows_env && device->vendor_id == VK_VENDOR_ID_NVIDIA &&
        strcmp(device->properties.deviceName.data(), "NVIDIA GeForce RTX 4090") == 0) {
        rwkv_mmv_rows = strcmp(rwkv_mmv_rows_env, "2") == 0 ? 2u :
                        strcmp(rwkv_mmv_rows_env, "4") == 0 ? 4u : 0u;
    }
    if (rwkv_mmv_rows) {
        GGML_LOG_INFO("ggml_vulkan: experimental RWKV MMV rows=%u cols=1 Q4_K/Q6_K\n", rwkv_mmv_rows);
    }
    // Vary independent output rows without changing the K reduction.
    const auto rwkv_mmv_row_count = [rwkv_mmv_rows](uint32_t original, uint32_t columns) {
        return rwkv_mmv_rows && columns == 1 ? rwkv_mmv_rows : original;
    };]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    set(after [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rwkv_mmv_row_count(rm_kq, i+1), 1, 1}, {wg_size_subgroup16, rwkv_mmv_row_count(rm_kq, i+1), i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    set(after [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rwkv_mmv_row_count(rm_kq, i+1), 1, 1}, {wg_size_subgroup16, rwkv_mmv_row_count(rm_kq, i+1), i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    set(after [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rwkv_mmv_row_count(rm_kq, i+1), 1, 1}, {wg_size_subgroup16, rwkv_mmv_row_count(rm_kq, i+1), i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    set(after [=[            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rwkv_mmv_row_count(rm_kq, i+1), 1, 1}, {wg_size_subgroup16, rwkv_mmv_row_count(rm_kq, i+1), i+1}, 1, true, use_subgroups16, force_subgroup_size16);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);]=])
    set(after [=[                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rwkv_mmv_row_count(rm_kq_int, i+1), 1, 1}, {wg_size_subgroup_int, 1*rwkv_mmv_row_count(rm_kq_int, i+1), i+1}, 1, true, use_subgroups, subgroup_size_int);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    set(before [=[                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);]=])
    set(after [=[                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rwkv_mmv_row_count(rm_kq_int, i+1), 1, 1}, {wg_size_subgroup_int, 1*rwkv_mmv_row_count(rm_kq_int, i+1), i+1}, 1, true, use_subgroups, subgroup_size_int);]=])
    string(REPLACE "${before}" "${after}" source "${source}")
    string(SHA256 patched_sha "${source}")
    if(NOT patched_sha STREQUAL "88d4476e5ae81a2b56f93d2ebb947427b36d042dddbe7b849ae180ea46bcdfa1")
        message(FATAL_ERROR "MMV row experiment did not match the pinned source")
    endif()
    file(WRITE "${source_file}" "${source}")
endfunction()
