add_subdirectory(common)

file(GLOB lib-headers "./*.hpp")
install(FILES ${lib-headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

if(BUILD_CPU)
  add_subdirectory(CPU)
endif()

if(BUILD_CUDA)
  add_subdirectory(CUDA)
endif()

if(BUILD_HIP)
  add_subdirectory(HIP)
endif()