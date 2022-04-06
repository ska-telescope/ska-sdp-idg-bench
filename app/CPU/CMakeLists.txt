project(lib-cpu CXX)

file(GLOB cpu-sources "./*.cpp")

add_library(${PROJECT_NAME} SHARED ${cpu-sources})
include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/app
                    ${CMAKE_SOURCE_DIR}/app/common)
target_link_libraries(${PROJECT_NAME} PUBLIC lib-common)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)