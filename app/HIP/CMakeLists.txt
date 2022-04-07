project(lib-hip CUDA CXX)

file(GLOB hip-sources "./kernels/*.hip.cpp" "./*.hip.cpp")

if("${GPU_BRAND}" STREQUAL "NVIDIA")
  set(CMAKE_CUDA_COMPILER hipcc)
  set(GPU_ARCH sm_${CUDA_ARCH_LIST})
  set_source_files_properties(${hip-sources} PROPERTIES LANGUAGE CUDA)
  add_compile_options(-D__HIP_PLATFORM_NVIDIA__ -D__CUDACC__
                      --gpu-architecture=${GPU_ARCH})
  include_directories(${PROJECT_NAME} PUBLIC $ENV{CUDA_ROOT}/include
                      $ENV{HIP_PATH}/include)
elseif("${GPU_BRAND}" STREQUAL "AMD")
  set(CMAKE_CXX_COMPILER $ENV{HIP_PATH}/bin/hipcc)
  set_source_files_properties(${hip-sources} PROPERTIES LANGUAGE CXX)
endif()

add_compile_options(-std=c++17)
include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/app
                    ${CMAKE_SOURCE_DIR}/app/HIP)
add_library(${PROJECT_NAME} SHARED ${hip-sources})

if(BUILD_LIBPOWERSENSOR)
  target_compile_options(${PROJECT_NAME} PRIVATE -DENABLE_POWERSENSOR)
  target_link_libraries(${PROJECT_NAME} PUBLIC powersensor)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC lib-common)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
