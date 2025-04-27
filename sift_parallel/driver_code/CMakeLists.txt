add_executable(compare_serial_parallel compare_serial_parallel.cpp)
add_executable(time_all time_all.cpp)

target_include_directories(compare_serial_parallel PRIVATE ../src)
target_include_directories(time_all PRIVATE ../src)

target_link_libraries(compare_serial_parallel PRIVATE
                      img
                      sift
)
target_link_libraries(time_all PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(compare_serial_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(time_all PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)