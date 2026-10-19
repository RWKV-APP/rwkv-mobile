# Preserve reviewed RWKV CUDA transforms in the pinned llama.cpp source.
function(rwkv_mobile_patch_llama_cpp_cuda_grouped source_dir)
    file(READ "${source_dir}/ggml-cuda.cu" source_0)
    file(READ "${source_dir}/ggml-cuda.cu" line_endings_0 HEX)
    string(FIND "${line_endings_0}" "0d0a" crlf_0)
    string(REPLACE "\r\n" "\n" source_0 "${source_0}")
    string(SHA256 hash_0 "${source_0}")
    if (NOT hash_0 STREQUAL "5f95865a3d2c878bfa62cde910935de0e05abf33054e2c8bee89a1cb2bab36b1" AND NOT hash_0 STREQUAL "52353449b105ca25d0d51444cdb7f1483a2e5267f73fd050cc4f57c3421cccdb" AND NOT hash_0 STREQUAL "093be203f9a345ecc7958b315af40f56c04227d1af8239b47d27f55b4f1e45f4" AND NOT hash_0 STREQUAL "62905093cb583f4f98c651c7ddd1f2e882a1695cec42e494db223b2d6971ff56")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file ggml-cuda.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/mmvq.cu" source_1)
    file(READ "${source_dir}/mmvq.cu" line_endings_1 HEX)
    string(FIND "${line_endings_1}" "0d0a" crlf_1)
    string(REPLACE "\r\n" "\n" source_1 "${source_1}")
    string(SHA256 hash_1 "${source_1}")
    if (NOT hash_1 STREQUAL "d1696a310eef474461b08d4ffdf1a69656c834823eae7f863c9a050692b551d7" AND NOT hash_1 STREQUAL "c9f1d6dbd56d38088fa807d2128e33fee40c53540a99f366d6ac14489adbb424")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file mmvq.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/mmvq.cuh" source_2)
    file(READ "${source_dir}/mmvq.cuh" line_endings_2 HEX)
    string(FIND "${line_endings_2}" "0d0a" crlf_2)
    string(REPLACE "\r\n" "\n" source_2 "${source_2}")
    string(SHA256 hash_2 "${source_2}")
    if (NOT hash_2 STREQUAL "b9f077366d3256e5e75ed700f74fec337b2d580493fa282eec8f60effff1521b" AND NOT hash_2 STREQUAL "39ad77a9bf04aa9a4168211a8ecc4acc038b9a09f23c907d92b9950548a85ace")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file mmvq.cuh no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/quantize.cu" source_3)
    file(READ "${source_dir}/quantize.cu" line_endings_3 HEX)
    string(FIND "${line_endings_3}" "0d0a" crlf_3)
    string(REPLACE "\r\n" "\n" source_3 "${source_3}")
    string(SHA256 hash_3 "${source_3}")
    if (NOT hash_3 STREQUAL "752c02fc5a17c1f884ba6acc6eed9419da7fc1d79bb6aa5f45a9837874744029" AND NOT hash_3 STREQUAL "bfcd4f9213da7db7b8aadf7a9ced7de339db7e53c283f6a32c2ec8f77731993d")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file quantize.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/quantize.cuh" source_4)
    file(READ "${source_dir}/quantize.cuh" line_endings_4 HEX)
    string(FIND "${line_endings_4}" "0d0a" crlf_4)
    string(REPLACE "\r\n" "\n" source_4 "${source_4}")
    string(SHA256 hash_4 "${source_4}")
    if (NOT hash_4 STREQUAL "017f8592498246d68f8b66e35b7d0de2d1d315440ebaeeaffdcaa8477d02fd8f" AND NOT hash_4 STREQUAL "c9a54fe72c061009c3abd349998fcd96782d5e52a35e808ee526219b431ba412")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file quantize.cuh no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/../../../src/models/rwkv7-base.cpp" source_5)
    file(READ "${source_dir}/../../../src/models/rwkv7-base.cpp" line_endings_5 HEX)
    string(FIND "${line_endings_5}" "0d0a" crlf_5)
    string(REPLACE "\r\n" "\n" source_5 "${source_5}")
    string(SHA256 hash_5 "${source_5}")
    if (NOT hash_5 STREQUAL "b74cd5ab735a15c2554e41a4a8a305a89a192ed330280f01c2d78ca94225c603" AND NOT hash_5 STREQUAL "825315fa79e93d3b50c1bdbdaca27e16f2384729ff2db3e3eb145cf9b19cc05e")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file ../../../src/models/rwkv7-base.cpp no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/unary.cu" source_6)
    file(READ "${source_dir}/unary.cu" line_endings_6 HEX)
    string(FIND "${line_endings_6}" "0d0a" crlf_6)
    string(REPLACE "\r\n" "\n" source_6 "${source_6}")
    string(SHA256 hash_6 "${source_6}")
    if (NOT hash_6 STREQUAL "4924d7b6d29770739e80e9ac4c1fbf71db11fa1be3a5f268faf4e68234a43613" AND NOT hash_6 STREQUAL "96a4043da05bfaabf45ed313733ffacd22f8b54131d4fffd616c2f4aac20cac9")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file unary.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/unary.cuh" source_7)
    file(READ "${source_dir}/unary.cuh" line_endings_7 HEX)
    string(FIND "${line_endings_7}" "0d0a" crlf_7)
    string(REPLACE "\r\n" "\n" source_7 "${source_7}")
    string(SHA256 hash_7 "${source_7}")
    if (NOT hash_7 STREQUAL "281a094dc34dd98a7e3d297c27a6d919a1bf482da2c76700ff6d56c4a31fef1d" AND NOT hash_7 STREQUAL "b8f60528cb74a8b49ec5f9a370a6b494d2ccc32f2866f6862c216c4b69ebf000")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file unary.cuh no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/norm.cu" source_8)
    file(READ "${source_dir}/norm.cu" line_endings_8 HEX)
    string(FIND "${line_endings_8}" "0d0a" crlf_8)
    string(REPLACE "\r\n" "\n" source_8 "${source_8}")
    string(SHA256 hash_8 "${source_8}")
    if (NOT hash_8 STREQUAL "4274f28bc6a26f35de21fcf3d025ecf2db7c3af469f9b3114a4b951d398a7947" AND NOT hash_8 STREQUAL "2e17ff4572805d68a370476320fabd22bedd7801ee2f36bbe79d62905a7d17c6")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file norm.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/norm.cuh" source_9)
    file(READ "${source_dir}/norm.cuh" line_endings_9 HEX)
    string(FIND "${line_endings_9}" "0d0a" crlf_9)
    string(REPLACE "\r\n" "\n" source_9 "${source_9}")
    string(SHA256 hash_9 "${source_9}")
    if (NOT hash_9 STREQUAL "dca47c2d6c15bed7b4297624b841f683fd6d3d46ad2c2fd12eb6a55ce1ce8b39" AND NOT hash_9 STREQUAL "039f299377cf34e57a70b1f65956285e1f1b5dacf3a16f108a41144f0e3f96a0")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file norm.cuh no longer matches the reviewed patch")
    endif()
    if (hash_0 STREQUAL "5f95865a3d2c878bfa62cde910935de0e05abf33054e2c8bee89a1cb2bab36b1")
        set(original [====[
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    const bool repeated_mix = ggml_can_fuse_subgraph(cgraph, i, {GGML_OP_SUB, GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD}, {i + 3});
    const bool channel_mix = !repeated_mix && ggml_can_fuse(cgraph, i, {GGML_OP_SUB, GGML_OP_MUL, GGML_OP_ADD});
    if (repeated_mix || channel_mix) {
]====])
        set(replacement [====[
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    if (node->op == GGML_OP_MUL_MAT && node->src[1] && ggml_nrows(node->src[1]) == 1) {
        ggml_tensor * projections[3] = {};
        int count = 0, last = i;
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        bool valid = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        for (int j = i; valid && j < cgraph->n_nodes && j < i + 8 && count < 3; ++j) {
            ggml_tensor * current = cgraph->nodes[j];
            if (current->flags & GGML_TENSOR_FLAG_OUTPUT) { valid = false; break; }
            if (current->op == GGML_OP_VIEW || current->op == GGML_OP_RESHAPE ||
                current->op == GGML_OP_PERMUTE || current->op == GGML_OP_TRANSPOSE || current->op == GGML_OP_NONE) continue;
            if (current->op != GGML_OP_MUL_MAT) { valid = false; break; }
            const ggml_tensor * weight = current->src[0];
            const ggml_tensor * input = current->src[1];
            if (!weight || !input || weight->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 ||
                current->type != GGML_TYPE_F32 || !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) ||
                !ggml_is_contiguous(current) || weight->ne[2] != 1 || weight->ne[3] != 1 ||
                ggml_nrows(input) != 1 || ggml_nrows(current) != 1 ||
                weight->ne[0] != input->ne[0] || weight->ne[1] != current->ne[0] ||
                weight->ne[0] > INT_MAX - MATRIX_ROW_PADDING || weight->ne[1] > INT_MAX) { valid = false; break; }
            const ggml_tensor * tensors[] = {weight, input, current};
            for (const ggml_tensor * tensor : tensors) {
                if (!tensor->data || !tensor->buffer || tensor->buffer->buft != ggml_backend_cuda_buffer_type(cuda_ctx->device)) valid = false;
            }
            if (!valid || ggml_backend_buffer_get_usage(weight->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) { valid = false; break; }
            if (count && !ggml_are_same_shape(weight, projections[0]->src[0])) { valid = false; break; }
            projections[count++] = current;
            last = j;
        }
        const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
            const uintptr_t x = (uintptr_t)a->data, y = (uintptr_t)b->data;
            return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
        };
        if (valid && count == 3) {
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
                if (overlaps(projections[a], projections[b]->src[0]) ||
                    overlaps(projections[a], projections[b]->src[1]) ||
                    (a != b && overlaps(projections[a], projections[b]))) valid = false;
            }
            if (valid) {
                ggml_cuda_mul_mat_vec_q_group3(*cuda_ctx, projections);
                return last - i;
            }
        }
    }
#endif
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    const bool repeated_mix = ggml_can_fuse_subgraph(cgraph, i, {GGML_OP_SUB, GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD}, {i + 3});
    const bool channel_mix = !repeated_mix && ggml_can_fuse(cgraph, i, {GGML_OP_SUB, GGML_OP_MUL, GGML_OP_ADD});
    if (repeated_mix || channel_mix) {
]====])
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        string(SHA256 patched_hash "${source_0}")
        if (NOT patched_hash STREQUAL "52353449b105ca25d0d51444cdb7f1483a2e5267f73fd050cc4f57c3421cccdb")
            message(FATAL_ERROR "CUDA patch verification failed for ggml-cuda.cu; source retained")
        endif()
    endif()
    if (hash_1 STREQUAL "d1696a310eef474461b08d4ffdf1a69656c834823eae7f863c9a050692b551d7")
        set(original [====[
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;
]====])
        set(replacement [====[
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

struct rwkv_q4_group3 {
    const void * weights[3];
    const void * inputs[3];
    float * outputs[3];
};

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void rwkv_mul_mat_vec_q_group3(
        const rwkv_q4_group3 group,
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    vx = blockIdx.z == 0 ? group.weights[0] : blockIdx.z == 1 ? group.weights[1] : group.weights[2];
    vy = blockIdx.z == 0 ? group.inputs[0] : blockIdx.z == 1 ? group.inputs[1] : group.inputs[2];
    dst = blockIdx.z == 0 ? group.outputs[0] : blockIdx.z == 1 ? group.outputs[1] : group.outputs[2];

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = 0;
#endif

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;
]====])
        string(REPLACE "${original}" "${replacement}" source_1 "${source_1}")
        set(original [====[

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
]====])
        set(replacement [====[

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


void ggml_cuda_mul_mat_vec_q_group3(ggml_backend_cuda_context & ctx, ggml_tensor * const nodes[3]) {
    const int cols = nodes[0]->src[0]->ne[0];
    const int rows = nodes[0]->src[0]->ne[1];
    const int padded = GGML_PAD(cols, MATRIX_ROW_PADDING);
    const size_t bytes = padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q0(ctx.pool(), bytes), q1(ctx.pool(), bytes), q2(ctx.pool(), bytes);
    char * quantized[] = {q0.get(), q1.get(), q2.get()};
    rwkv_q4_group3 group{};
    const float * inputs[3];
    void * outputs[3];
    for (int j = 0; j < 3; ++j) {
        inputs[j] = (const float *)nodes[j]->src[1]->data;
        outputs[j] = quantized[j];
        group.weights[j] = nodes[j]->src[0]->data;
        group.inputs[j] = quantized[j];
        group.outputs[j] = (float *)nodes[j]->data;
    }
    rwkv_quantize_q8_1_group3_cuda(inputs, outputs, cols, padded, ctx.stream());
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING);
    const int warp = ggml_cuda_info().devices[ctx.device].warp_size;
    const auto table = get_device_table_id(cc);
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = get_vdr_mmvq(GGML_TYPE_Q4_K);
    const int nwarps = calc_nwarps(GGML_TYPE_Q4_K, 1, table);
    const bool small = nwarps > 1 && cols/qk < nwarps*(vdr*warp/qi);
    auto dims = calc_launch_params<GGML_TYPE_Q4_K>(1, rows, 1, 1, warp, table, small);
    dims.first.z = 3;
    const auto zero = make_uint3(0, 0, 0);
    const auto one = init_fastdiv_values(1);
    const ggml_cuda_mm_fusion_args_device fusion{};
    if (small) {
        rwkv_mul_mat_vec_q_group3<GGML_TYPE_Q4_K,1,false,true><<<dims.first,dims.second,0,ctx.stream()>>>(
            group, nullptr, nullptr, nullptr, fusion, nullptr, cols, zero, cols/qk, padded/QK8_1,
            rows, one, 1, 1, 1, one, 1, 1, 1, 0);
    } else {
        rwkv_mul_mat_vec_q_group3<GGML_TYPE_Q4_K,1,false,false><<<dims.first,dims.second,0,ctx.stream()>>>(
            group, nullptr, nullptr, nullptr, fusion, nullptr, cols, zero, cols/qk, padded/QK8_1,
            rows, one, 1, 1, 1, one, 1, 1, 1, 0);
    }
}
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_1 "${source_1}")
        string(SHA256 patched_hash "${source_1}")
        if (NOT patched_hash STREQUAL "c9f1d6dbd56d38088fa807d2128e33fee40c53540a99f366d6ac14489adbb424")
            message(FATAL_ERROR "CUDA patch verification failed for mmvq.cu; source retained")
        endif()
    endif()
    if (hash_2 STREQUAL "b9f077366d3256e5e75ed700f74fec337b2d580493fa282eec8f60effff1521b")
        set(original [====[
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
]====])
        set(replacement [====[
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

void ggml_cuda_mul_mat_vec_q_group3(ggml_backend_cuda_context & ctx, ggml_tensor * const nodes[3]);
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_2 "${source_2}")
        string(SHA256 patched_hash "${source_2}")
        if (NOT patched_hash STREQUAL "39ad77a9bf04aa9a4168211a8ecc4acc038b9a09f23c907d92b9950548a85ace")
            message(FATAL_ERROR "CUDA patch verification failed for mmvq.cuh; source retained")
        endif()
    endif()
    if (hash_3 STREQUAL "752c02fc5a17c1f884ba6acc6eed9419da7fc1d79bb6aa5f45a9837874744029")
        set(original [====[
        quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}
]====])
        set(replacement [====[
        quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

static __global__ void rwkv_quantize_q8_1_group3(
        const float * x0, const float * x1, const float * x2,
        void * y0, void * y1, void * y2, const int64_t ne00, const int64_t ne0) {
    const float * x = blockIdx.z == 0 ? x0 : blockIdx.z == 1 ? x1 : x2;
    void * vy = blockIdx.z == 0 ? y0 : blockIdx.z == 1 ? y1 : y2;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i_cont = i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

void rwkv_quantize_q8_1_group3_cuda(const float * const inputs[3], void * const outputs[3],
                                  int64_t cols, int64_t padded, cudaStream_t stream) {
    GGML_ASSERT(cols > 0 && padded >= cols && padded % QK8_1 == 0);
    const dim3 blocks((padded + CUDA_QUANTIZE_BLOCK_SIZE - 1)/CUDA_QUANTIZE_BLOCK_SIZE, 1, 3);
    rwkv_quantize_q8_1_group3<<<blocks, dim3(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1), 0, stream>>>(
        inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2], cols, padded);
}
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_3 "${source_3}")
        string(SHA256 patched_hash "${source_3}")
        if (NOT patched_hash STREQUAL "bfcd4f9213da7db7b8aadf7a9ced7de339db7e53c283f6a32c2ec8f77731993d")
            message(FATAL_ERROR "CUDA patch verification failed for quantize.cu; source retained")
        endif()
    endif()
    if (hash_4 STREQUAL "017f8592498246d68f8b66e35b7d0de2d1d315440ebaeeaffdcaa8477d02fd8f")
        set(original [====[
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);
]====])
        set(replacement [====[
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

void rwkv_quantize_q8_1_group3_cuda(const float * const inputs[3], void * const outputs[3], int64_t cols, int64_t padded, cudaStream_t stream);
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_4 "${source_4}")
        string(SHA256 patched_hash "${source_4}")
        if (NOT patched_hash STREQUAL "c9a54fe72c061009c3abd349998fcd96782d5e52a35e808ee526219b431ba412")
            message(FATAL_ERROR "CUDA patch verification failed for quantize.cuh; source retained")
        endif()
    endif()
    if (hash_5 STREQUAL "b74cd5ab735a15c2554e41a4a8a305a89a192ed330280f01c2d78ca94225c603")
        set(original [====[

    ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
]====])
        set(replacement [====[

    ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
    if (layer.time_mix_receptance->type == GGML_TYPE_Q4_K && layer.time_mix_key->type == GGML_TYPE_Q4_K && layer.time_mix_value->type == GGML_TYPE_Q4_K) {
        ggml_build_forward_expand(gf, r);
        ggml_build_forward_expand(gf, k);
        ggml_build_forward_expand(gf, v);
    }
#endif
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
]====])
        string(REPLACE "${original}" "${replacement}" source_5 "${source_5}")
        string(SHA256 patched_hash "${source_5}")
        if (NOT patched_hash STREQUAL "825315fa79e93d3b50c1bdbdaca27e16f2384729ff2db3e3eb145cf9b19cc05e")
            message(FATAL_ERROR "CUDA patch verification failed for ../../../src/models/rwkv7-base.cpp; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_0}")
    if (fusion_hash STREQUAL "52353449b105ca25d0d51444cdb7f1483a2e5267f73fd050cc4f57c3421cccdb")
        set(original [====[
}

#endif
// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    if (node->op == GGML_OP_MUL_MAT && node->src[1] && ggml_nrows(node->src[1]) == 1) {
        ggml_tensor * projections[3] = {};
        int count = 0, last = i;
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        bool valid = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        for (int j = i; valid && j < cgraph->n_nodes && j < i + 8 && count < 3; ++j) {
            ggml_tensor * current = cgraph->nodes[j];
            if (current->flags & GGML_TENSOR_FLAG_OUTPUT) { valid = false; break; }
            if (current->op == GGML_OP_VIEW || current->op == GGML_OP_RESHAPE ||
]====])
        set(replacement [====[
}

#endif
// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
        ggml_can_fuse(cgraph, i, {GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY})) {
        ggml_tensor * scaled = cgraph->nodes[i+1];
        ggml_tensor * output = cgraph->nodes[i+2];
        const ggml_tensor * input = node->src[0];
        bool valid = input && scaled->src[0] == node && output->src[0] == scaled &&
            ggml_get_unary_op(output) == GGML_UNARY_OP_EXP &&
            !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(scaled->flags & GGML_TENSOR_FLAG_OUTPUT);
        const ggml_tensor * tensors[] = {input, node, scaled, output};
        for (const ggml_tensor * tensor : tensors) {
            valid = valid && tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                tensor->data && tensor->buffer && tensor->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) &&
                ggml_are_same_shape(tensor, output);
        }
        if (valid) {
            const uintptr_t x = (uintptr_t)input->data, y = (uintptr_t)output->data;
            const bool overlaps = x <= y ? y-x < ggml_nbytes(input) : x-y < ggml_nbytes(output);
            valid = !overlaps || x == y;
            const int64_t count = ggml_nelements(output);
            valid = valid && count > 0 && (count+255)/256 <= INT_MAX;
            if (valid) {
                ggml_cuda_rwkv_sigmoid_scale_exp_exact(*cuda_ctx, input, scaled, output);
                return 2;
            }
        }
    }
#endif
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    if (node->op == GGML_OP_MUL_MAT && node->src[1] && ggml_nrows(node->src[1]) == 1) {
        ggml_tensor * projections[3] = {};
        int count = 0, last = i;
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        bool valid = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        for (int j = i; valid && j < cgraph->n_nodes && j < i + 8 && count < 3; ++j) {
            ggml_tensor * current = cgraph->nodes[j];
            if (current->flags & GGML_TENSOR_FLAG_OUTPUT) { valid = false; break; }
            if (current->op == GGML_OP_VIEW || current->op == GGML_OP_RESHAPE ||
]====])
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        string(SHA256 fusion_hash "${source_0}")
        if (NOT fusion_hash STREQUAL "093be203f9a345ecc7958b315af40f56c04227d1af8239b47d27f55b4f1e45f4")
            message(FATAL_ERROR "CUDA fusion patch verification failed for ggml-cuda.cu; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_6}")
    if (fusion_hash STREQUAL "4924d7b6d29770739e80e9ac4c1fbf71db11fa1be3a5f268faf4e68234a43613")
        set(original [====[

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}
]====])
        set(replacement [====[

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __global__ void rwkv_sigmoid_scale_exp_exact(
        const float * input, float * output, int64_t count, float scale, float bias) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count) return;
    float value = op_sigmoid(input[i]);
    // Keep the rounding boundaries of the original separate kernels.
    asm volatile("mov.b32 %0, %1;" : "=f"(value) : "f"(value));
    value = scale*value+bias;
    asm volatile("mov.b32 %0, %1;" : "=f"(value) : "f"(value));
    output[i] = op_exp(value);
}

void ggml_cuda_rwkv_sigmoid_scale_exp_exact(ggml_backend_cuda_context & ctx,
        const ggml_tensor * input, const ggml_tensor * scaled, ggml_tensor * output) {
    float scale, bias;
    memcpy(&scale, (const float *)scaled->op_params, sizeof(float));
    memcpy(&bias, (const float *)scaled->op_params+1, sizeof(float));
    const int64_t count = ggml_nelements(output);
    rwkv_sigmoid_scale_exp_exact<<<(count+255)/256,256,0,ctx.stream()>>>(
        (const float *)input->data, (float *)output->data, count, scale, bias);
}
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_6 "${source_6}")
        string(SHA256 fusion_hash "${source_6}")
        if (NOT fusion_hash STREQUAL "96a4043da05bfaabf45ed313733ffacd22f8b54131d4fffd616c2f4aac20cac9")
            message(FATAL_ERROR "CUDA fusion patch verification failed for unary.cu; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_7}")
    if (fusion_hash STREQUAL "281a094dc34dd98a7e3d297c27a6d919a1bf482da2c76700ff6d56c4a31fef1d")
        set(original [====[

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
]====])
        set(replacement [====[

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_rwkv_sigmoid_scale_exp_exact(ggml_backend_cuda_context & ctx,
        const ggml_tensor * input, const ggml_tensor * scaled, ggml_tensor * output);
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_7 "${source_7}")
        string(SHA256 fusion_hash "${source_7}")
        if (NOT fusion_hash STREQUAL "b8f60528cb74a8b49ec5f9a370a6b494d2ccc32f2866f6862c216c4b69ebf000")
            message(FATAL_ERROR "CUDA fusion patch verification failed for unary.cuh; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_0}")
    if (fusion_hash STREQUAL "093be203f9a345ecc7958b315af40f56c04227d1af8239b47d27f55b4f1e45f4")
        set(original [====[
            }
            const int64_t count = ggml_nelements(add);
            compatible = compatible && count > 0 && (count + 255) / 256 <= INT_MAX;
            if (compatible) {
                rwkv_repeat_lerp_f32_exact<<<(count + 255) / 256,256,0,cuda_ctx->stream()>>>(
                    (const float *)previous->data, (const float *)current->data, (const float *)weight->data,
                    (float *)add->data, ggml_nelements(current), current->ne[0], count);
                return repeated_mix ? 3 : 2;
            }
        }
    }

    if (ggml_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * add = cgraph->nodes[i + 2];
        const ggml_tensor * weight = mul->src[0] == node ? mul->src[1] : mul->src[0];
        const ggml_tensor * bias = add->src[0] == mul ? add->src[1] : add->src[0];
        bool compatible = true;
        const ggml_tensor * tensors[] = {node, node->src[0], weight, bias, mul, add};
        for (const ggml_tensor * tensor : tensors) {
            compatible = compatible && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
        }
        compatible = compatible && ggml_are_same_shape(node, add) && ggml_are_same_shape(node, mul)
                     && weight->ne[0] == node->ne[0] && ggml_nrows(weight) == 1
]====])
        set(replacement [====[
            }
            const int64_t count = ggml_nelements(add);
            compatible = compatible && count > 0 && (count + 255) / 256 <= INT_MAX;
            if (compatible) {
                rwkv_repeat_lerp_f32_exact<<<(count + 255) / 256,256,0,cuda_ctx->stream()>>>(
                    (const float *)previous->data, (const float *)current->data, (const float *)weight->data,
                    (float *)add->data, ggml_nelements(current), current->ne[0], count);
                return repeated_mix ? 3 : 2;
            }
        }
    }


    if (node->op == GGML_OP_NORM && ggml_can_fuse_subgraph(cgraph, i,
            {GGML_OP_NORM, GGML_OP_RESHAPE, GGML_OP_MUL, GGML_OP_ADD}, {i+3})) {
        const ggml_tensor * reshaped = cgraph->nodes[i+1];
        const ggml_tensor * mul = cgraph->nodes[i+2];
        ggml_tensor * output = cgraph->nodes[i+3];
        const ggml_tensor * weight = mul->src[0] == reshaped ? mul->src[1] : mul->src[0];
        const ggml_tensor * bias = output->src[0] == mul ? output->src[1] : output->src[0];
        const ggml_tensor * input = node->src[0];
        bool valid = input && weight && bias && reshaped->src[0] == node &&
            (mul->src[0] == reshaped || mul->src[1] == reshaped) &&
            (output->src[0] == mul || output->src[1] == mul) &&
            !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(reshaped->flags & GGML_TENSOR_FLAG_OUTPUT) &&
            !(mul->flags & GGML_TENSOR_FLAG_OUTPUT);
        for (const ggml_tensor * tensor : {input, (const ggml_tensor *)node, reshaped, mul, (const ggml_tensor *)output, weight, bias}) {
            valid = valid && tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                tensor->data && tensor->buffer && tensor->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device);
        }
        if (valid) {
            valid = node->ne[0] == 64 && node->ne[1] > 0 && node->ne[1] <= INT_MAX/64 &&
                node->ne[2] > 0 && node->ne[2] <= INT_MAX && node->ne[3] == 1 &&
                ggml_are_same_shape(input,node) && ggml_are_same_shape(reshaped,mul) && ggml_are_same_shape(mul,output) &&
                output->ne[0] == node->ne[0]*node->ne[1] && output->ne[1] == node->ne[2] &&
                output->ne[2] == 1 && output->ne[3] == 1 &&
                weight->ne[0] == output->ne[0] && bias->ne[0] == output->ne[0] &&
                ggml_nrows(weight) == 1 && ggml_nrows(bias) == 1;
            const auto overlaps=[](const ggml_tensor * a,const ggml_tensor * b) {
                const uintptr_t x=(uintptr_t)a->data,y=(uintptr_t)b->data;
                return x<=y ? y-x<ggml_nbytes(a) : x-y<ggml_nbytes(b);
            };
            valid = valid && (!overlaps(input,output) || input->data == output->data) &&
                !overlaps(weight,output) && !overlaps(bias,output);
            if (valid) {
                ggml_cuda_op_group_norm_affine_exact(*cuda_ctx,node,weight,bias,output);
                return 3;
            }
        }
    }

    if (ggml_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * add = cgraph->nodes[i + 2];
        const ggml_tensor * weight = mul->src[0] == node ? mul->src[1] : mul->src[0];
        const ggml_tensor * bias = add->src[0] == mul ? add->src[1] : add->src[0];
        bool compatible = true;
        const ggml_tensor * tensors[] = {node, node->src[0], weight, bias, mul, add};
        for (const ggml_tensor * tensor : tensors) {
            compatible = compatible && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
        }
        compatible = compatible && ggml_are_same_shape(node, add) && ggml_are_same_shape(node, mul)
                     && weight->ne[0] == node->ne[0] && ggml_nrows(weight) == 1
]====])
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        string(SHA256 fusion_hash "${source_0}")
        if (NOT fusion_hash STREQUAL "62905093cb583f4f98c651c7ddd1f2e882a1695cec42e494db223b2d6971ff56")
            message(FATAL_ERROR "CUDA fusion patch verification failed for ggml-cuda.cu; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_8}")
    if (fusion_hash STREQUAL "4274f28bc6a26f35de21fcf3d025ecf2db7c3af469f9b3114a4b951d398a7947")
        set(original [====[
#include "norm.cuh"
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
template <int block_size, bool affine = false>
#else
template <int block_size>
#endif
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        const int64_t stride_sample, const float eps, const float * weight = nullptr, const float * bias = nullptr) {
#else
        const int64_t stride_sample, const float eps) {
#endif
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
]====])
        set(replacement [====[
#include "norm.cuh"
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
template <int block_size, bool affine = false, bool grouped_affine = false>
#else
template <int block_size>
#endif
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        const int64_t stride_sample, const float eps, const float * weight = nullptr, const float * bias = nullptr) {
#else
        const int64_t stride_sample, const float eps) {
#endif
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
]====])
        string(REPLACE "${original}" "${replacement}" source_8 "${source_8}")
        set(original [====[
    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        if constexpr (affine) {
            const float value = (x[col] - mean) * inv_std;
            dst[col] = __fadd_rn(__fmul_rn(value, weight[col]), bias[col]);
        } else {
#endif
        dst[col] = (x[col] - mean) * inv_std;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        }
#endif
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
]====])
        set(replacement [====[
    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        if constexpr (affine) {
            const float value = (x[col] - mean) * inv_std;
            const int64_t affine_col = col + (grouped_affine ? int64_t(blockIdx.x) * ncols : 0);
            dst[col] = __fadd_rn(__fmul_rn(value, weight[affine_col]), bias[affine_col]);
        } else {
#endif
        dst[col] = (x[col] - mean) * inv_std;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        }
#endif
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
]====])
        string(REPLACE "${original}" "${replacement}" source_8 "${source_8}")
        set(original [====[
    const float * x = (const float *)src->data;
    float * output = (float *)dst->data;
    const float * w = (const float *)weight->data;
    const float * b = (const float *)bias->data;
    const int64_t s1 = src->nb[1] / sizeof(float), s2 = src->nb[2] / sizeof(float), s3 = src->nb[3] / sizeof(float);
    if (ncols < 1024) {
        norm_f32<WARP_SIZE, true><<<blocks, WARP_SIZE, 0, ctx.stream()>>>(x, output, ncols, s1, s2, s3, eps, w, b);
    } else {
        norm_f32<1024, true><<<blocks, 1024, 32 * sizeof(float2), ctx.stream()>>>(x, output, ncols, s1, s2, s3, eps, w, b);
    }
}

#endif
void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

]====])
        set(replacement [====[
    const float * x = (const float *)src->data;
    float * output = (float *)dst->data;
    const float * w = (const float *)weight->data;
    const float * b = (const float *)bias->data;
    const int64_t s1 = src->nb[1] / sizeof(float), s2 = src->nb[2] / sizeof(float), s3 = src->nb[3] / sizeof(float);
    if (ncols < 1024) {
        norm_f32<WARP_SIZE, true><<<blocks, WARP_SIZE, 0, ctx.stream()>>>(x, output, ncols, s1, s2, s3, eps, w, b);
    } else {
        norm_f32<1024, true><<<blocks, 1024, 32 * sizeof(float2), ctx.stream()>>>(x, output, ncols, s1, s2, s3, eps, w, b);
    }
}

void ggml_cuda_op_group_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                                          const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst) {
    const ggml_tensor * src = norm->src[0];
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const dim3 blocks(src->ne[1], src->ne[2], src->ne[3]);
    const int64_t s1 = src->nb[1]/sizeof(float), s2 = src->nb[2]/sizeof(float), s3 = src->nb[3]/sizeof(float);
    norm_f32<WARP_SIZE, true, true><<<blocks, WARP_SIZE, 0, ctx.stream()>>>(
        (const float *)src->data, (float *)dst->data, src->ne[0], s1, s2, s3, eps,
        (const float *)weight->data, (const float *)bias->data);
}

#endif
void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

]====])
        string(REPLACE "${original}" "${replacement}" source_8 "${source_8}")
        string(SHA256 fusion_hash "${source_8}")
        if (NOT fusion_hash STREQUAL "2e17ff4572805d68a370476320fabd22bedd7801ee2f36bbe79d62905a7d17c6")
            message(FATAL_ERROR "CUDA fusion patch verification failed for norm.cu; source retained")
        endif()
    endif()
    string(SHA256 fusion_hash "${source_9}")
    if (fusion_hash STREQUAL "dca47c2d6c15bed7b4297624b841f683fd6d3d46ad2c2fd12eb6a55ce1ce8b39")
        set(original [====[
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

void ggml_cuda_op_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst);
#endif
]====])
        set(replacement [====[
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

void ggml_cuda_op_group_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst);
void ggml_cuda_op_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst);
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_9 "${source_9}")
        string(SHA256 fusion_hash "${source_9}")
        if (NOT fusion_hash STREQUAL "039f299377cf34e57a70b1f65956285e1f1b5dacf3a16f108a41144f0e3f96a0")
            message(FATAL_ERROR "CUDA fusion patch verification failed for norm.cuh; source retained")
        endif()
    endif()
    if (NOT hash_0 STREQUAL "62905093cb583f4f98c651c7ddd1f2e882a1695cec42e494db223b2d6971ff56")
        set(rwkv_newline_style LF)
        if (NOT crlf_0 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/ggml-cuda.cu" CONTENT "${source_0}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_1 STREQUAL "d1696a310eef474461b08d4ffdf1a69656c834823eae7f863c9a050692b551d7")
        set(rwkv_newline_style LF)
        if (NOT crlf_1 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/mmvq.cu" CONTENT "${source_1}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_2 STREQUAL "b9f077366d3256e5e75ed700f74fec337b2d580493fa282eec8f60effff1521b")
        set(rwkv_newline_style LF)
        if (NOT crlf_2 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/mmvq.cuh" CONTENT "${source_2}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_3 STREQUAL "752c02fc5a17c1f884ba6acc6eed9419da7fc1d79bb6aa5f45a9837874744029")
        set(rwkv_newline_style LF)
        if (NOT crlf_3 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/quantize.cu" CONTENT "${source_3}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_4 STREQUAL "017f8592498246d68f8b66e35b7d0de2d1d315440ebaeeaffdcaa8477d02fd8f")
        set(rwkv_newline_style LF)
        if (NOT crlf_4 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/quantize.cuh" CONTENT "${source_4}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_5 STREQUAL "b74cd5ab735a15c2554e41a4a8a305a89a192ed330280f01c2d78ca94225c603")
        set(rwkv_newline_style LF)
        if (NOT crlf_5 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/../../../src/models/rwkv7-base.cpp" CONTENT "${source_5}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_6 STREQUAL "4924d7b6d29770739e80e9ac4c1fbf71db11fa1be3a5f268faf4e68234a43613")
        set(rwkv_newline_style LF)
        if (NOT crlf_6 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/unary.cu" CONTENT "${source_6}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_7 STREQUAL "281a094dc34dd98a7e3d297c27a6d919a1bf482da2c76700ff6d56c4a31fef1d")
        set(rwkv_newline_style LF)
        if (NOT crlf_7 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/unary.cuh" CONTENT "${source_7}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_8 STREQUAL "4274f28bc6a26f35de21fcf3d025ecf2db7c3af469f9b3114a4b951d398a7947")
        set(rwkv_newline_style LF)
        if (NOT crlf_8 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/norm.cu" CONTENT "${source_8}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_9 STREQUAL "dca47c2d6c15bed7b4297624b841f683fd6d3d46ad2c2fd12eb6a55ce1ce8b39")
        set(rwkv_newline_style LF)
        if (NOT crlf_9 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/norm.cuh" CONTENT "${source_9}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
endfunction()
