set(sampler "${PALM_INFRA_SOURCE_DIR}/engine/sampler.cpp")
file(READ "${sampler}" contents)
if (contents MATCHES "#if defined\\(_WIN32\\)")
    return()
endif()

set(before [[
    return static_cast<float>(rand_r(seed)) /
           static_cast<float>(RAND_MAX);
]])
set(after [[
#if defined(_WIN32)
    // rand_r is POSIX-only. Keep the sampler state local to the caller on
    // Windows as well, using the traditional ANSI C LCG rather than the
    // process-global rand() state.
    unsigned int value = *seed;
    value = value * 1103515245u + 12345u;
    *seed = value;
    return static_cast<float>((value / 65536u) % 32768u) / 32767.0f;
#else
    return static_cast<float>(rand_r(seed)) /
           static_cast<float>(RAND_MAX);
#endif
]])
string(REPLACE "${before}" "${after}" patched "${contents}")
if (patched STREQUAL contents)
    message(FATAL_ERROR "Palm sampler layout changed at the pinned revision")
endif()
file(WRITE "${sampler}" "${patched}")
