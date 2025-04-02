cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O3")

# Set CUDA flags
cmake_policy(SET CMP0104 NEW)  # Set policy to avoid CUDA_ARCHITECTURES warning
set(COMPUTE_CAPABILITY 60)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}")
# Define CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

project(sift_cpp LANGUAGES CXX CUDA)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(timing_scripts)