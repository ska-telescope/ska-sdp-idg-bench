project(lib-cuda CXX)

file(GLOB cuda-sources "./kernels/*.cu" "./*cu")

add_library(${PROJECT_NAME} SHARED ${cuda-sources})
include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/app
                    ${CMAKE_SOURCE_DIR}/app/CUDA)
target_link_libraries(${PROJECT_NAME} PUBLIC lib-common)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)