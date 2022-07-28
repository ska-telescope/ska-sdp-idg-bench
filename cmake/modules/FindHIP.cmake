# CMake find_package() Module for HIP  C++ runtime API
# https://github.com/ROCm-Developer-Tools/HIP
#
# Based on:
# https://github.com/ROCm-Developer-Tools/HIP/blob/main/cmake/FindHIP.cmake

if(NOT HIP_ROOT_DIR)
    # Search in user specified path first
    find_path(
        HIP_ROOT_DIR
        NAMES bin/hipconfig
        PATHS
        "$ENV{ROCM_PATH}/hip"
        ENV HIP_PATH
        /opt/rocm/hip
        NO_DEFAULT_PATH
        )
    if(NOT EXISTS ${HIP_ROOT_DIR})
        if(HIP_FIND_REQUIRED)
            message(FATAL_ERROR "Specify HIP_ROOT_DIR")
        elseif(NOT HIP_FIND_QUIETLY)
            message("HIP_ROOT_DIR not found or specified")
        endif()
    endif()
    # And push it back to the cache
    set(HIP_ROOT_DIR ${HIP_ROOT_DIR} CACHE PATH "HIP installed location" FORCE)
endif()

# Find HIP include directory
find_path(
    HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    HINTS ${HIP_ROOT_DIR}
    PATH_SUFFIXES include
)

# Find HIPCC executable
find_program(
    HIP_HIPCC_EXECUTABLE
    NAMES hipcc
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
if(NOT HIP_HIPCC_EXECUTABLE)
    # Now search in default paths
    find_program(HIP_HIPCC_EXECUTABLE hipcc)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    HIP
    REQUIRED_VARS
    HIP_ROOT_DIR
    HIP_INCLUDE_DIR
    HIP_HIPCC_EXECUTABLE
    )
