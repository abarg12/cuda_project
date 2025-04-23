add_executable(compare_serial_parallel compare_serial_parallel.cpp)

target_include_directories(compare_serial_parallel PRIVATE ../src)

target_link_libraries(compare_serial_parallel PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(compare_serial_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)