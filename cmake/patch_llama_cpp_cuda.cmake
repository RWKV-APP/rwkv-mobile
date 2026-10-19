# Preserve reviewed RWKV CUDA transforms in the pinned llama.cpp source.
function(rwkv_mobile_patch_llama_cpp_cuda source_dir)
    file(READ "${source_dir}/norm.cu" source_0)
    file(READ "${source_dir}/norm.cu" line_endings_0 HEX)
    string(FIND "${line_endings_0}" "0d0a" crlf_0)
    string(REPLACE "\r\n" "\n" source_0 "${source_0}")
    string(SHA256 hash_0 "${source_0}")
    if (NOT hash_0 STREQUAL "32e6065c3b32a3461d1e9f301331d1a4af90d123807db924878b40dafc2497b4" AND NOT hash_0 STREQUAL "4274f28bc6a26f35de21fcf3d025ecf2db7c3af469f9b3114a4b951d398a7947" AND NOT hash_0 STREQUAL "2e17ff4572805d68a370476320fabd22bedd7801ee2f36bbe79d62905a7d17c6")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file norm.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/norm.cuh" source_1)
    file(READ "${source_dir}/norm.cuh" line_endings_1 HEX)
    string(FIND "${line_endings_1}" "0d0a" crlf_1)
    string(REPLACE "\r\n" "\n" source_1 "${source_1}")
    string(SHA256 hash_1 "${source_1}")
    if (NOT hash_1 STREQUAL "0f2d6b743ff349012424cddc8866eb0fa504ea16b7a58cb47b31b392d1f78a73" AND NOT hash_1 STREQUAL "dca47c2d6c15bed7b4297624b841f683fd6d3d46ad2c2fd12eb6a55ce1ce8b39" AND NOT hash_1 STREQUAL "039f299377cf34e57a70b1f65956285e1f1b5dacf3a16f108a41144f0e3f96a0")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file norm.cuh no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/ggml-cuda.cu" source_2)
    file(READ "${source_dir}/ggml-cuda.cu" line_endings_2 HEX)
    string(FIND "${line_endings_2}" "0d0a" crlf_2)
    string(REPLACE "\r\n" "\n" source_2 "${source_2}")
    string(SHA256 hash_2 "${source_2}")
    if (NOT hash_2 STREQUAL "f0c281d33df005093b25c9827822c11d4d3031bfe915ffd6aa385fd608f4ae4d" AND NOT hash_2 STREQUAL "5f95865a3d2c878bfa62cde910935de0e05abf33054e2c8bee89a1cb2bab36b1" AND NOT hash_2 STREQUAL "52353449b105ca25d0d51444cdb7f1483a2e5267f73fd050cc4f57c3421cccdb" AND NOT hash_2 STREQUAL "093be203f9a345ecc7958b315af40f56c04227d1af8239b47d27f55b4f1e45f4" AND NOT hash_2 STREQUAL "62905093cb583f4f98c651c7ddd1f2e882a1695cec42e494db223b2d6971ff56")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file ggml-cuda.cu no longer matches the reviewed patch")
    endif()
    file(READ "${source_dir}/wkv.cu" source_3)
    file(READ "${source_dir}/wkv.cu" line_endings_3 HEX)
    string(FIND "${line_endings_3}" "0d0a" crlf_3)
    string(REPLACE "\r\n" "\n" source_3 "${source_3}")
    string(SHA256 hash_3 "${source_3}")
    if (NOT hash_3 STREQUAL "25e0b0bf660b09cfed312a28d17e14699f85f98a063c6b33537966ac8eff19b0" AND NOT hash_3 STREQUAL "ec9ede6c9c5f6216fc07792e18a9b80443f2bd1d358bca5489b76c02ed7f2e12")
        message(FATAL_ERROR "Pinned llama.cpp CUDA file wkv.cu no longer matches the reviewed patch")
    endif()
    if (hash_0 STREQUAL "32e6065c3b32a3461d1e9f301331d1a4af90d123807db924878b40dafc2497b4")
        set(original [====[
#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

]====])
        set(replacement [====[
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
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        set(original [====[
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

]====])
        set(replacement [====[
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

]====])
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        set(original [====[
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
]====])
        set(replacement [====[
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
void ggml_cuda_op_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                                    const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst) {
    const ggml_tensor * src = norm->src[0];
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const dim3 blocks(src->ne[1], src->ne[2], src->ne[3]);
    const int ncols = src->ne[0];
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
]====])
        string(REPLACE "${original}" "${replacement}" source_0 "${source_0}")
        string(SHA256 patched_hash "${source_0}")
        if (NOT patched_hash STREQUAL "4274f28bc6a26f35de21fcf3d025ecf2db7c3af469f9b3114a4b951d398a7947")
            message(FATAL_ERROR "CUDA patch verification failed for norm.cu; source retained")
        endif()
    endif()
    if (hash_1 STREQUAL "0f2d6b743ff349012424cddc8866eb0fa504ea16b7a58cb47b31b392d1f78a73")
        set(original [====[
void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
]====])
        set(replacement [====[
void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY

void ggml_cuda_op_norm_affine_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * weight, const ggml_tensor * bias, ggml_tensor * dst);
#endif
]====])
        string(REPLACE "${original}" "${replacement}" source_1 "${source_1}")
        string(SHA256 patched_hash "${source_1}")
        if (NOT patched_hash STREQUAL "dca47c2d6c15bed7b4297624b841f683fd6d3d46ad2c2fd12eb6a55ce1ce8b39")
            message(FATAL_ERROR "CUDA patch verification failed for norm.cuh; source retained")
        endif()
    endif()
    if (hash_2 STREQUAL "f0c281d33df005093b25c9827822c11d4d3031bfe915ffd6aa385fd608f4ae4d")
        set(original [====[
    return false;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

]====])
        set(replacement [====[
    return false;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
__global__ static void rwkv_mul_add_f32_exact(const float * a, const float * b, const float * bias, float * dst, int64_t count) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __fadd_rn(__fmul_rn(a[i], b[i]), bias[i]);
}


__global__ static void rwkv_repeat_lerp_f32_exact(const float * previous, const float * current,
        const float * weight, float * output, int64_t base_count, int64_t width, int64_t count) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t input = i % base_count;
        const int64_t coefficient = i % width + i / base_count * width;
        const float delta = __fsub_rn(previous[input], current[input]);
        output[i] = __fadd_rn(__fmul_rn(delta, weight[coefficient]), current[input]);
    }
}

#endif
// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

]====])
        string(REPLACE "${original}" "${replacement}" source_2 "${source_2}")
        set(original [====[
    }

    ggml_tensor * node = cgraph->nodes[i];

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
]====])
        set(replacement [====[
    }

    ggml_tensor * node = cgraph->nodes[i];
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY


    const bool repeated_mix = ggml_can_fuse_subgraph(cgraph, i, {GGML_OP_SUB, GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD}, {i + 3});
    const bool channel_mix = !repeated_mix && ggml_can_fuse(cgraph, i, {GGML_OP_SUB, GGML_OP_MUL, GGML_OP_ADD});
    if (repeated_mix || channel_mix) {
        const ggml_tensor * previous = node->src[0];
        const ggml_tensor * current = node->src[1];
        const ggml_tensor * repeat = repeated_mix ? cgraph->nodes[i + 1] : node;
        const ggml_tensor * mul = cgraph->nodes[i + (repeated_mix ? 2 : 1)];
        ggml_tensor * add = cgraph->nodes[i + (repeated_mix ? 3 : 2)];
        const ggml_tensor * weight = mul->src[1];
        bool compatible = (!repeated_mix || repeat->src[0] == node) && mul->src[0] == repeat &&
            add->src[0] == mul && add->src[1] == current && previous && current && weight;
        const ggml_tensor * tensors[] = {node, previous, current, repeat, mul, add, weight};
        for (const ggml_tensor * tensor : tensors) {
            compatible = compatible && tensor && tensor->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(tensor) && tensor->data && tensor->buffer &&
                tensor->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device);
        }
        if (compatible) {
            compatible = ggml_are_same_shape(previous, current) && ggml_are_same_shape(node, current) &&
                current->ne[3] == 1 && (repeated_mix ? (repeat->ne[3] == 5 || repeat->ne[3] == 6) : repeat->ne[3] == 1) &&
                repeat->ne[0] == current->ne[0] && repeat->ne[1] == current->ne[1] && repeat->ne[2] == current->ne[2] &&
                ggml_are_same_shape(repeat, mul) && ggml_are_same_shape(mul, add) &&
                weight->ne[0] == current->ne[0] && weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == repeat->ne[3];
            const ggml_tensor * intermediates[] = {node, repeat, mul};
            for (const ggml_tensor * tensor : intermediates) {
                if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) compatible = false;
            }
            const uintptr_t output = (uintptr_t)add->data;
            const size_t bytes = ggml_nbytes(add);
            for (const ggml_tensor * input : {previous, current, weight}) {
                const uintptr_t start = (uintptr_t)input->data;
                if (start <= output ? output - start < ggml_nbytes(input) : start - output < bytes) compatible = false;
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
                     && bias->ne[0] == node->ne[0] && ggml_nrows(bias) == 1;
        const uintptr_t output = (uintptr_t)add->data;
        const size_t bytes = ggml_nbytes(add);
        const ggml_tensor * inputs[] = {node->src[0], weight, bias};
        for (const ggml_tensor * input : inputs) {
            const uintptr_t begin = (uintptr_t)input->data;
            const bool overlaps = begin < output + bytes && output < begin + ggml_nbytes(input);
            if (overlaps && (input != node->src[0] || begin != output)) compatible = false;
        }
        if (compatible) {
            ggml_cuda_op_norm_affine_exact(*cuda_ctx, node, weight, bias, add);
            return 2;
        }
    }

    if (ggml_can_fuse(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD })) {
        ggml_tensor * add = cgraph->nodes[i + 1];
        const ggml_tensor * bias = add->src[0] == node ? add->src[1] : add->src[0];
        bool compatible = true;
        const ggml_tensor * tensors[] = {node, node->src[0], node->src[1], bias, add};
        for (const ggml_tensor * tensor : tensors) {
            compatible = compatible && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && ggml_are_same_shape(tensor, node);
        }
        const uintptr_t output_begin = (uintptr_t)add->data;
        const size_t bytes = ggml_nbytes(add);
        const ggml_tensor * inputs[] = {node->src[0], node->src[1], bias};
        for (const ggml_tensor * input : inputs) {
            const uintptr_t input_begin = (uintptr_t)input->data;
            if (input_begin != output_begin && input_begin < output_begin + bytes && output_begin < input_begin + bytes) {
                compatible = false;
            }
        }
        if (compatible) {
            const int64_t count = ggml_nelements(add);
            rwkv_mul_add_f32_exact<<<(count + 255) / 256, 256, 0, cuda_ctx->stream()>>>(
                (const float *)node->src[0]->data, (const float *)node->src[1]->data,
                (const float *)bias->data, (float *)add->data, count);
            return 1;
        }
    }
#endif

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
]====])
        string(REPLACE "${original}" "${replacement}" source_2 "${source_2}")
        string(SHA256 patched_hash "${source_2}")
        if (NOT patched_hash STREQUAL "5f95865a3d2c878bfa62cde910935de0e05abf33054e2c8bee89a1cb2bab36b1")
            message(FATAL_ERROR "CUDA patch verification failed for ggml-cuda.cu; source retained")
        endif()
    endif()
    if (hash_3 STREQUAL "25e0b0bf660b09cfed312a28d17e14699f85f98a063c6b33537966ac8eff19b0")
        set(original [====[
    }
}

template <int block_size>
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
]====])
        set(replacement [====[
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
template <int block_size, int rows_per_block = block_size>
#else
template <int block_size>
#endif
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
    static_assert(block_size % rows_per_block == 0);
    const int tid = blockIdx.y * rows_per_block + threadIdx.x;
#else
    const int tid = threadIdx.x;
#endif
    const int bid = blockIdx.x;

    const int head_size = block_size;
]====])
        string(REPLACE "${original}" "${replacement}" source_3 "${source_3}")
        set(original [====[

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        __syncthreads();

        float sa = 0;
]====])
        set(replacement [====[

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        if constexpr (rows_per_block == block_size) {
#endif
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        } else {
            for (int j = threadIdx.x; j < head_size; j += rows_per_block) {
                _r[j] = r[t - tid + j];
                _w[j] = w[t - tid + j];
                _k[j] = k[t - tid + j];
                _a[j] = a[t - tid + j];
                _b[j] = b[t - tid + j];
            }
        }
#endif
        __syncthreads();

        float sa = 0;
]====])
        string(REPLACE "${original}" "${replacement}" source_3 "${source_3}")
        set(original [====[
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    }
}
]====])
        set(replacement [====[
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        if (T == B) {
            rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE, 16><<<dim3(B * H, C / H / 16), 16, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
        } else
#endif
        {
#endif
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        }
#endif
    } else {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        if (T == B) {
            rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2, 16><<<dim3(B * H, C / H / 16), 16, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
        } else
#endif
        {
#endif
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // RWKV_CUDA_ONLY
        }
#endif
    }
}
]====])
        string(REPLACE "${original}" "${replacement}" source_3 "${source_3}")
        string(SHA256 patched_hash "${source_3}")
        if (NOT patched_hash STREQUAL "ec9ede6c9c5f6216fc07792e18a9b80443f2bd1d358bca5489b76c02ed7f2e12")
            message(FATAL_ERROR "CUDA patch verification failed for wkv.cu; source retained")
        endif()
    endif()
    if (hash_0 STREQUAL "32e6065c3b32a3461d1e9f301331d1a4af90d123807db924878b40dafc2497b4")
        set(rwkv_newline_style LF)
        if (NOT crlf_0 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/norm.cu" CONTENT "${source_0}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_1 STREQUAL "0f2d6b743ff349012424cddc8866eb0fa504ea16b7a58cb47b31b392d1f78a73")
        set(rwkv_newline_style LF)
        if (NOT crlf_1 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/norm.cuh" CONTENT "${source_1}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_2 STREQUAL "f0c281d33df005093b25c9827822c11d4d3031bfe915ffd6aa385fd608f4ae4d")
        set(rwkv_newline_style LF)
        if (NOT crlf_2 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/ggml-cuda.cu" CONTENT "${source_2}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
    if (hash_3 STREQUAL "25e0b0bf660b09cfed312a28d17e14699f85f98a063c6b33537966ac8eff19b0")
        set(rwkv_newline_style LF)
        if (NOT crlf_3 EQUAL -1)
            set(rwkv_newline_style CRLF)
        endif()
        file(CONFIGURE OUTPUT "${source_dir}/wkv.cu" CONTENT "${source_3}"
            @ONLY NEWLINE_STYLE "${rwkv_newline_style}")
    endif()
endfunction()
