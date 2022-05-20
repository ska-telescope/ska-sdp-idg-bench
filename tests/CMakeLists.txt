
if(BUILD_CUDA)
  # KERNEL_SRC is set by app/CUDA/CMakeLists.txt
  file(GLOB KERNEL_SRC "${CMAKE_SOURCE_DIR}/app/CUDA/kernels/*.cu")

  # Iterate all kernels
  foreach(kernelsourcepath ${KERNEL_SRC})

    # Get filename without path
    get_filename_component(kernelsourcename ${kernelsourcepath} NAME)

    # Get name of executable by removing extension and prepending programming model
    string(REPLACE ".cu" "" testname cuda-${kernelsourcename})

    # Determine whether the kernel is a gridder or degridder kernel
    if(${testname} MATCHES ".*-degridder_.*")
      set(testcommonsource "degridder_common.cpp")
    elseif(${testname} MATCHES ".*-gridder_.*")
      set(testcommonsource "gridder_common.cpp")
    else()
      message(FATAL_ERROR "${kernelsourcename} is neither a gridder or degridder.")
    endif()

    # Build the executable
    add_executable(${testname} ${testcommonsource})
    target_include_directories(${testname} PUBLIC ${CMAKE_SOURCE_DIR}/app
                                                  ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${testname} lib-cpu
                                      $<TARGET_OBJECTS:lib-cuda>
                                      $<TARGET_OBJECTS:${testname}-kernel>
                                      CUDA::cudart CUDA::cuda_driver)
    target_compile_options(${testname} PRIVATE -DBUILD_CUDA)
    if(BUILD_LIBPOWERSENSOR)
      target_link_libraries(${testname} powersensor)
    endif()
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endforeach(kernelsourcepath ${KERNEL_SRC})
endif()

# TODO: Update HIP like CUDA
file(GLOB TESTS_SRC "./*.cpp")

if(BUILD_HIP)
  foreach(testsourcefile ${TESTS_SRC})
    get_filename_component(testsourcefile ${testsourcefile} NAME)
    string(REPLACE ".cpp" "" testname hip-${testsourcefile})
    add_executable(${testname} ${testsourcefile})
    target_include_directories(${testname} PUBLIC ${CMAKE_SOURCE_DIR}/app
                                                  ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(${testname} lib-cpu lib-hip)
    target_compile_options(${testname} PRIVATE -DBUILD_HIP)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endforeach(testsourcefile ${TESTS_SRC})
endif()
