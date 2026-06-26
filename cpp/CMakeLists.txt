cmake_minimum_required(VERSION 3.18)
project(shwfs_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/tensorrt/include
)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/tensorrt/lib
)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Install TensorRT and set include/library paths if needed.")
endif()

add_executable(shwfs_trt main.cpp)
target_include_directories(shwfs_trt PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(shwfs_trt PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)
target_compile_options(shwfs_trt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
