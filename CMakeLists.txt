cmake_minimum_required(VERSION 3.17)

project(idg-bench CXX)

include(CMakeDependentOption)
include(CheckLanguage)
include(ExternalProject)

option(BUILD_LIBPOWERSENSOR OFF)

option(BUILD_CPU "Build CPU version (check GPU correctness)" ON)
set(GPU_BRANDS NVIDIA AMD)
set(GPU_BRAND
    NVIDIA
    CACHE STRING "GPU brands")
set_property(CACHE GPU_BRAND PROPERTY STRINGS ${GPU_BRANDS})

cmake_dependent_option(BUILD_CUDA "Build with CUDA" OFF
                       "${GPU_BRAND} STREQUAL NVIDIA" OFF)

cmake_dependent_option(BUILD_HIP "Build with HIP" OFF
                       "BUILD_CUDA OR ${GPU_BRAND} STREQUAL AMD" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Default build type." FORCE)
endif()

set(CMAKE_CXX_FLAGS "-O3 -fno-math-errno -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -fno-math-errno -march=native -DDEBUG")

if(BUILD_LIBPOWERSENSOR)
  set(EXTERNAL_INSTALL_LOCATION_LIBPOWERSENSOR
      ${CMAKE_SOURCE_DIR}/external/libpowersensor)
  ExternalProject_Add(
    libpowersensor
    GIT_REPOSITORY https://gitlab.com/astron-misc/libpowersensor
    GIT_TAG master
    # add other arguments if needed such as build with CUDA and find the
    # necessary packages above such as FFTW
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION_LIBPOWERSENSOR}
      -DBUILD_LIKWID_POWERSENSOR=OFF -DBUILD_ARDUINO_POWERSENSOR=OFF
      -DBUILD_NVML_POWERSENSOR=${BUILD_CUDA}
    # Uncomment the next line if you want to always build the external project
    # BUILD_ALWAYS 1
  )
  include_directories(${EXTERNAL_INSTALL_LOCATION_LIBPOWERSENSOR}/include)
  link_directories(${EXTERNAL_INSTALL_LOCATION_LIBPOWERSENSOR}/lib)
endif()

if(BUILD_CUDA AND "${GPU_BRAND}" STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O3 -DDEBUG")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)

    # Automatically find CUDA CAPABILITY This is necessary for old CUDA versions
    # New CUDA versions e.g. 11.6.0 support -arch=native
    include(FindCUDA/select_compute_arch)
    cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

    if(NOT DEFINED CUDA_ARCH_LIST)
      message(FATAL_ERROR "CUDA DEVICE NOT FOUND")
    else()
      message("FOUND CUDA CAPABILITY: ${CUDA_ARCH_LIST}")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  else()
    message(FATAL_ERROR "No CUDA support")
  endif()
endif()

if(BUILD_HIP AND "${GPU_BRAND}" STREQUAL "AMD")
  find_package(hip REQUIRED)
endif()

add_subdirectory(app)

add_subdirectory(tests)
