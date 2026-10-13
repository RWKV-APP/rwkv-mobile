find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --no-index --ignore-space-change --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
if(already_applied EQUAL 0)
    return()
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --no-index --ignore-space-change --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --no-index --ignore-space-change "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)
