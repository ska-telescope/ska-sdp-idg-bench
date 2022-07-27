project(lib-cuda CXX)

# Build the CUDA utilities library
add_library(lib-cuda OBJECT util.cpp)
target_include_directories(lib-cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..
                                           ${CUDAToolkit_INCLUDE_DIRS})
if(BUILD_LIBPOWERSENSOR)
  target_compile_options(lib-cuda PRIVATE -DENABLE_POWERSENSOR)
endif()

# Find all kernel sources, the KERNEL_SRC variable
# is also used in tests/CMakeLists.txt
file(GLOB KERNEL_SRC "./kernels/*.cu")

# Iterate all kernels
foreach(kernelsourcepath ${KERNEL_SRC})
  # Get filename without path
  get_filename_component(kernelsourcefile ${kernelsourcepath} NAME)

  # Get name of library by removing extension and prepending programming model
  string(REPLACE ".cu" "" kernelname cuda-${kernelsourcefile}-kernel)

  # Build the library
  add_library(${kernelname} OBJECT ${kernelsourcepath})
  target_include_directories(${kernelname} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..
                      ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../common)
  target_compile_options(${kernelname} PRIVATE -DFUNCTION_ATTRIBUTES=__device__)
endforeach(kernelsourcepath ${KERNEL_SRC})
