add_executable(serial_and_parallel serial_and_parallel.cpp)

target_include_directories(serial_and_parallel PRIVATE ../src)

target_link_libraries(serial_and_parallel PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(serial_and_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)