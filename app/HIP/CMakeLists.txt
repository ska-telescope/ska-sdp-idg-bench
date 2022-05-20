project(lib-hip CXX CUDA) 

set(CMAKE_CUDA_COMPILER hipcc)

# Build the hip utilities library
set_source_files_properties(util.cpp PROPERTIES LANGUAGE CUDA)
add_library(lib-hip OBJECT util.cpp)
target_include_directories(lib-hip PUBLIC ${CMAKE_SOURCE_DIR}/app
                                           ${CUDAToolkit_INCLUDE_DIRS}
                                           $ENV{HIP_PATH}/include)
set(GPU_ARCH sm_${CUDA_ARCH_LIST})
add_compile_options(-D__HIP_PLATFORM_NVIDIA__ -D__CUDACC__
                                           --gpu-architecture=${GPU_ARCH})
                     
if(BUILD_LIBPOWERSENSOR)
  target_compile_options(lib-hip PRIVATE -DENABLE_POWERSENSOR)
endif()


# Find all kernel sources, the KERNEL_SRC variable
# is also used in tests/CMakeLists.txt
file(GLOB KERNEL_SRC "./kernels/*.hip.cpp")


# Iterate all kernels
foreach(kernelsourcepath ${KERNEL_SRC})
  # Get filename without path
  get_filename_component(kernelsourcefile ${kernelsourcepath} NAME)
  
  set_source_files_properties(${kernelsourcepath} PROPERTIES LANGUAGE CUDA)

  # Get name of library by removing extension and prepending programming model
  string(REPLACE ".hip.cpp" "" kernelname hip-${kernelsourcefile}-kernel)

  # Build the library
  add_library(${kernelname} OBJECT ${kernelsourcepath})
  target_include_directories(${kernelname} PUBLIC ${CMAKE_SOURCE_DIR}/app
                      ${CMAKE_SOURCE_DIR}/app/HIP ${CMAKE_SOURCE_DIR}/app/common $ENV{HIP_PATH}/include)
  target_compile_options(${kernelname} PRIVATE -DFUNCTION_ATTRIBUTES=__device__)
endforeach(kernelsourcepath ${KERNEL_SRC})