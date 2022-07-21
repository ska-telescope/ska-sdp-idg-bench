cmake_minimum_required(VERSION 3.17)

project(idg-bench CXX)

include(CMakeDependentOption)
include(CheckLanguage)
include(ExternalProject)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

option(BUILD_LIBPOWERSENSOR OFF)

if(BUILD_LIBPOWERSENSOR)
   find_package(PowerSensor REQUIRED)
endif()

if(${POWERSENSOR_FOUND})
   set(HAVE_POWERSENSOR TRUE)
   include_directories(${POWERSENSOR_INCLUDE_DIR})
   link_directories(${POWERSENSOR_LIBRARY})
endif()

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

if(BUILD_CUDA AND "${GPU_BRAND}" STREQUAL "NVIDIA")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O3 -DDEBUG")
  check_language(CUDA)
  enable_language(CUDA)
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
    message("Found CUDA capability: ${CUDA_ARCH_LIST}")
  endif()

  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

if(BUILD_HIP)
  find_package(HIP REQUIRED)
  if("${GPU_BRAND}" STREQUAL "NVIDIA")
    # FIXME
    #set(ENV{HIP_PLATFORM} "nvidia")
    message("Run \"export HIP_PLATFORM=nvidia\" before calling \"make\".")
  endif()

endif()

add_subdirectory(app)

add_subdirectory(tests)
