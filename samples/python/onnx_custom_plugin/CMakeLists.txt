# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CustomHardMax LANGUAGES CXX CUDA)

if(NOT MSVC)
  # Enable all compile warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
  set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
  set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
  set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
  set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_CUDART_LIB cudart HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(CUDART_LIB ${_CUDART_LIB})

find_library(_CUBLAS_LIB cublas HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

find_library(_CUDA_LIB cuda HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib/stubs lib64/stubs)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

# -------- BUILDING --------

add_definitions(-DTENSORRT_BUILD_LIB)

# Add include directories
get_filename_component(SAMPLES_COMMON_DIR ${CMAKE_SOURCE_DIR}/../../common/ ABSOLUTE)
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/plugin/ ${SAMPLES_COMMON_DIR})

# Define Hardmax plugin library target
add_library(customHardmaxPlugin MODULE
  ${SAMPLES_COMMON_DIR}/logger.cpp
  ${CMAKE_SOURCE_DIR}/plugin/customHardmaxPlugin.cpp
  ${CMAKE_SOURCE_DIR}/plugin/customHardmaxPlugin.h
)

# Use C++11
target_compile_features(customHardmaxPlugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(customHardmaxPlugin PRIVATE ${NVINFER_LIB})
target_link_libraries(customHardmaxPlugin PRIVATE ${CUDART_LIB})
target_link_libraries(customHardmaxPlugin PRIVATE ${CUBLAS_LIB})
target_link_libraries(customHardmaxPlugin PRIVATE ${CUDA_LIB})
