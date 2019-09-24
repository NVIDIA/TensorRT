# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

ARG CUDA_VERSION=10.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu16.04

LABEL maintainer="NVIDIA CORPORATION"

# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

# Install Cmake
RUN cd /tmp &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh &&\
    chmod +x cmake-3.14.4-Linux-x86_64.sh &&\
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license &&\
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Set environment and working directory
ENV TRT_RELEASE /tensorrt
ENV TRT_LIB_DIR $TRT_RELEASE/lib
ENV TRT_SOURCE /workspace/TensorRT
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$TRT_LIB_DIR
WORKDIR /workspace

RUN ["/bin/bash"]
