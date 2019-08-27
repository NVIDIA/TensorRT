# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
--expt-relaxed-constexpr \
--expt-extended-lambda \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-O3")

set(BERT_LIBS 
    cudart
    cublas
    nvinfer
    pthread
    z
    )

include_directories(
    ../../include
    ../../samples/common
    /usr/local/cuda-10.1/targets/x86_64-linux/include
    ./
    ./plugins
    ./layers
    ./util
    ../../third_party/cub/
    /workspace/cub/
    /workspace/cutlass/
    )

link_directories(
    /usr/local/cuda-10.1/targets/x86_64-linux/lib
    /tensorrt/lib
    )

add_library(common SHARED
    ../../samples/common/logger.cpp
    util/dataUtils.cpp
    )

add_library(bert_plugins SHARED
    plugins/geluPlugin.cu
    plugins/skipLayerNormPlugin.cu
    plugins/qkvToContextPlugin.cu
    plugins/embLayerNormPlugin.cu
    )

target_link_libraries(bert_plugins 
    ${BERT_LIBS}
    )

target_link_libraries(common 
    ${BERT_LIBS}
    )

add_executable(sample_bert
    sampleBERT.cpp
    )

target_compile_features(sample_bert PUBLIC cxx_std_11)

target_link_libraries(sample_bert 
    common
    bert_plugins
    )

