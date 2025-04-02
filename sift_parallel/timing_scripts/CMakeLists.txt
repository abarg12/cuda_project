add_executable(serial_and_parallel_timing serial_and_parallel_timing.cpp)

target_include_directories(serial_and_parallel_timing PRIVATE ../src)

target_link_libraries(serial_and_parallel_timing PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(find_keypoints PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)