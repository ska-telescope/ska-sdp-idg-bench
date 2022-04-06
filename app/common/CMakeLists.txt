project(lib-common CXX)

file(GLOB common-sources "./*.cpp")

add_library(${PROJECT_NAME} SHARED ${common-sources})
include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/app
                    ${CMAKE_SOURCE_DIR}/app/common)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)