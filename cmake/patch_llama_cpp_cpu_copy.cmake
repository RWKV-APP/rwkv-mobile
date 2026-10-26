function(rwkv_mobile_patch_llama_cpp_cpu_copy cpu_dir)
    set(ops_path "${cpu_dir}/ops.cpp")
    set(tasks_path "${cpu_dir}/ggml-cpu.c")
    if(NOT EXISTS "${ops_path}" OR NOT EXISTS "${tasks_path}")
        message(FATAL_ERROR "Pinned CPU copy sources missing")
    endif()
    file(READ "${ops_path}" ops)
    file(READ "${tasks_path}" tasks)
    file(READ "${ops_path}" ops_endings HEX)
    file(READ "${tasks_path}" tasks_endings HEX)
    string(REPLACE "\r" "" ops "${ops}")
    string(REPLACE "\r" "" tasks "${tasks}")
    set(ops_original "${ops}")
    set(tasks_original "${tasks}")
    set(copy_helpers [==[

// Pure byte copies only; retain the original kernels for strided or overlapping layouts.
static bool rwkv_cpu_copy_disjoint(const ggml_tensor * src, const ggml_tensor * dst) {
    const uintptr_t a = (uintptr_t) src->data, b = (uintptr_t) dst->data;
    return a < b ? b - a >= ggml_nbytes(src) : a - b >= ggml_nbytes(dst);
}

static bool rwkv_cpu_repeat_contiguous(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst) ||
        !rwkv_cpu_copy_disjoint(src, dst) || ggml_blck_size(src->type) != 1) return false;
    const size_t elem = ggml_type_size(src->type);
    if (elem != 2 && elem != 4) return false;
    const int64_t rows = ggml_nrows(dst);
    const int64_t begin = rows * params->ith / params->nth;
    const int64_t end = rows * (params->ith + 1) / params->nth;
    const size_t bytes = src->ne[0] * elem;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t i1 = row % dst->ne[1];
        const int64_t i2 = row / dst->ne[1] % dst->ne[2];
        const int64_t i3 = row / (dst->ne[1] * dst->ne[2]);
        const char * x = (const char *) src->data + (i1 % src->ne[1]) * src->nb[1] +
            (i2 % src->ne[2]) * src->nb[2] + (i3 % src->ne[3]) * src->nb[3];
        char * y = (char *) dst->data + row * dst->nb[1];
        for (int64_t col = 0; col < dst->ne[0]; col += src->ne[0]) memcpy(y + col * elem, x, bytes);
    }
    return true;
}

static bool rwkv_cpu_concat_contiguous(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0], * b = dst->src[1];
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst) ||
        ggml_blck_size(a->type) != 1 || !rwkv_cpu_copy_disjoint(a, dst) ||
        !rwkv_cpu_copy_disjoint(b, dst)) return false;
    const size_t elem = ggml_type_size(a->type);
    if (elem != 1 && elem != 2 && elem != 4) return false;
    const int dim = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(dim >= 0 && dim < 4);
    const int64_t rows = ggml_nrows(dst);
    const int64_t begin = rows * params->ith / params->nth;
    const int64_t end = rows * (params->ith + 1) / params->nth;
    for (int64_t row = begin; row < end; ++row) {
        int64_t c[4] = {0, row % dst->ne[1], row / dst->ne[1] % dst->ne[2], row / (dst->ne[1] * dst->ne[2])};
        char * y = (char *) dst->data + row * dst->nb[1];
        if (dim == 0) {
            const char * x0 = (const char *) a->data + c[1]*a->nb[1] + c[2]*a->nb[2] + c[3]*a->nb[3];
            const char * x1 = (const char *) b->data + c[1]*b->nb[1] + c[2]*b->nb[2] + c[3]*b->nb[3];
            memcpy(y, x0, a->ne[0] * elem);
            memcpy(y + a->ne[0] * elem, x1, b->ne[0] * elem);
        } else {
            const ggml_tensor * src = a;
            if (c[dim] >= a->ne[dim]) { src = b; c[dim] -= a->ne[dim]; }
            const char * x = (const char *) src->data + c[1]*src->nb[1] + c[2]*src->nb[2] + c[3]*src->nb[3];
            memcpy(y, x, dst->ne[0] * elem);
        }
    }
    return true;
}

]==])
    if(NOT ops MATCHES "static bool rwkv_cpu_repeat_contiguous")
    set(old_repeat [==[
void ggml_compute_forward_repeat(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];]==])
    set(new_repeat [==[
void ggml_compute_forward_repeat(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (rwkv_cpu_repeat_contiguous(params, dst)) return;]==])
    string(FIND "${ops}" "${old_repeat}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Pinned CPU repeat entrypoint changed")
    endif()
    set(old_concat [==[
void ggml_compute_forward_concat(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];]==])
    set(new_concat [==[
void ggml_compute_forward_concat(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (rwkv_cpu_concat_contiguous(params, dst)) return;]==])
    string(FIND "${ops}" "${old_concat}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Pinned CPU concat entrypoint changed")
    endif()
    string(REPLACE "void ggml_compute_forward_repeat(\n" "${copy_helpers}void ggml_compute_forward_repeat(\n" ops "${ops}")
    string(REPLACE "${old_repeat}" "${new_repeat}" ops "${ops}")
    string(REPLACE "${old_concat}" "${new_concat}" ops "${ops}")
    endif()
    set(old_tasks [==[
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:]==])
    set(new_tasks [==[
        case GGML_OP_REPEAT:
            {
                n_tasks = ggml_nbytes(node) >= 65536 && ggml_is_contiguous(node) &&
                    ggml_is_contiguous(node->src[0]) ? n_threads : 1;
            } break;
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:]==])
    if(NOT tasks MATCHES "n_tasks = ggml_nbytes\\(node\\) >= 65536")
        string(FIND "${tasks}" "${old_tasks}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "Pinned CPU REPEAT task policy changed")
        endif()
        string(REPLACE "${old_tasks}" "${new_tasks}" tasks "${tasks}")
    endif()
    set(copy_payload "${ops}")
    if(ops_endings MATCHES "0d0a")
        set(copy_newline CRLF)
    else()
        set(copy_newline LF)
    endif()
    if(NOT "${ops}" STREQUAL "${ops_original}")
    file(CONFIGURE OUTPUT "${ops_path}" CONTENT "@copy_payload@" @ONLY NEWLINE_STYLE "${copy_newline}")
    endif()
    set(copy_payload "${tasks}")
    if(tasks_endings MATCHES "0d0a")
        set(copy_newline CRLF)
    else()
        set(copy_newline LF)
    endif()
    if(NOT "${tasks}" STREQUAL "${tasks_original}")
    file(CONFIGURE OUTPUT "${tasks_path}" CONTENT "@copy_payload@" @ONLY NEWLINE_STYLE "${copy_newline}")
    endif()
endfunction()
