project(lib-cpu CXX)

file(GLOB cpu-sources "./kernels/*.cpp" "./*.cpp")

add_library(${PROJECT_NAME} SHARED ${cpu-sources})
include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../
                    ${CMAKE_CURRENT_LIST_DIR}/../../common ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC lib-common)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
