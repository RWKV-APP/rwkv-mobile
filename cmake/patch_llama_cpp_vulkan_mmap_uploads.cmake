function(rwkv_mobile_patch_llama_cpp_vulkan_mmap_uploads source_file)
    file(READ "${source_file}" source)
    string(REPLACE "\r\n" "\n" source "${source}")
    string(SHA256 source_sha "${source}")
    if(source_sha STREQUAL "359f9e35531bc425bc3e9db0adfb1a15d8d356a9c0d4c2054d03227615a13477")
        return()
    endif()
    if(NOT source_sha STREQUAL "e487eeb7849bcb5ff1be2c90c360cded0dad3af34b844150c711d6b779a8c4a6")
        message(FATAL_ERROR "Unexpected llama model loader source for Vulkan mmap uploads")
    endif()
    set(before_0 [=[    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t> events;]=])
    set(after_0 [=[    struct upload_event_deleter {
        void operator()(ggml_backend_event_t event) const {
            ggml_backend_event_synchronize(event);
            ggml_backend_event_free(event);
        }
    };
    ggml_backend_ptr upload_owner;
    std::vector<ggml_backend_buffer_ptr> host_buffers;
    std::vector<std::unique_ptr<ggml_backend_event, upload_event_deleter>> events;]=])
    string(REPLACE "${before_0}" "${after_0}" source "${source}")
    set(before_1 [=[        if (use_mmap || check_tensors) {]=])
    set(after_1 [=[        if (check_tensors || (use_mmap && arch_name != "rwkv7")) {]=])
    string(REPLACE "${before_1}" "${after_1}" source "${source}")
    set(before_2 [=[        if (buft != ggml_backend_dev_buffer_type(dev)) {]=])
    set(after_2 [=[        if (use_mmap && strcmp(ggml_backend_dev_name(dev), "Vulkan0") != 0) {
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {]=])
    string(REPLACE "${before_2}" "${after_2}" source "${source}")
    set(before_3 [=[            host_buffers.emplace_back(buf);]=])
    set(after_3 [=[            host_buffers.emplace_back(ggml_backend_buffer_ptr(buf));]=])
    string(REPLACE "${before_3}" "${after_3}" source "${source}")
    set(before_4 [=[            events.emplace_back(event);]=])
    set(after_4 [=[            events.emplace_back(std::unique_ptr<ggml_backend_event, upload_event_deleter>(event));]=])
    string(REPLACE "${before_4}" "${after_4}" source "${source}")
    set(before_5 [=[    }(__func__);

    if (upload_backend) {]=])
    set(after_5 [=[    }(__func__);
    upload_owner.reset(upload_backend);

    if (upload_backend) {]=])
    string(REPLACE "${before_5}" "${after_5}" source "${source}")
    set(before_6 [=[                        ggml_backend_event_synchronize(events[buffer_idx]);]=])
    set(after_6 [=[                        ggml_backend_event_synchronize(events[buffer_idx].get());]=])
    string(REPLACE "${before_6}" "${after_6}" source "${source}")
    set(before_7 [=[                        ggml_backend_event_record(events[buffer_idx], upload_backend);]=])
    set(after_7 [=[                        ggml_backend_event_record(events[buffer_idx].get(), upload_backend);]=])
    string(REPLACE "${before_7}" "${after_7}" source "${source}")
    set(before_8 [=[            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {]=])
    set(after_8 [=[            } else if (upload_backend && !ggml_backend_buffer_is_host(cur->buffer)) {
                LLAMA_LOG_DEBUG("%s: staged mapped GPU tensor %s (%zu bytes)\n", __func__, ggml_get_name(cur), n_size);
                // Keep mapped CPU weights while staging GPU transfers.
                for (size_t offset = 0; offset < n_size;) {
                    const size_t chunk = std::min(buffer_size, n_size - offset);
                    ggml_backend_event_synchronize(events[buffer_idx].get());
                    std::memcpy(host_ptrs[buffer_idx], data + offset, chunk);
                    ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], offset, chunk);
                    ggml_backend_event_record(events[buffer_idx].get(), upload_backend);
                    offset += chunk;
                    buffer_idx = (buffer_idx + 1) % n_buffers;
                }
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {]=])
    string(REPLACE "${before_8}" "${after_8}" source "${source}")
    set(before_9 [=[    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);]=])
    set(after_9 [=[    // Drain transfers before mapped GPU-only weight regions are released.
    events.clear();
    host_buffers.clear();
    upload_owner.reset();]=])
    string(REPLACE "${before_9}" "${after_9}" source "${source}")
    string(SHA256 patched_sha "${source}")
    if(NOT patched_sha STREQUAL "359f9e35531bc425bc3e9db0adfb1a15d8d356a9c0d4c2054d03227615a13477")
        message(FATAL_ERROR "Vulkan mmap upload patch did not match the pinned source")
    endif()
    file(WRITE "${source_file}" "${source}")
endfunction()
