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

#ARG CUDA_VERSION=10.1
#FROM ubuntu:18.04 
FROM nvidia/cuda:10.0-devel-ubuntu18.04

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

        
COPY docker/jetpack_files /pdk_files
COPY scripts/stubify.sh /pdk_files

# Install CUDA cross compile toolchain
RUN dpkg -i /pdk_files/cuda-repo-cross-aarch64-10-0-local-10.0.326_1.0-1_all.deb /pdk_files/cuda-repo-ubuntu1804-10-0-local-10.0.326-410.108_1.0-1_amd64.deb \
    && apt-get update \
    && apt-get install -y cuda-cross-aarch64 \
    && rm -rf /var/lib/apt/lists/*

# Unpack cudnn
RUN dpkg -x /pdk_files/libcudnn7_7.5.0.56-1+cuda10.0_arm64.deb /pdk_files/cudnn  \
    && dpkg -x /pdk_files/libcudnn7-dev_7.5.0.56-1+cuda10.0_arm64.deb /pdk_files/cudnn \
    && cd /pdk_files/cudnn/usr/include/aarch64-linux-gnu \
    && cd /pdk_files/cudnn/usr/lib/aarch64-linux-gnu \
    && ln -s libcudnn.so.7 libcudnn.so \
    && cd /pdk_files/cudnn \
    && ln -s usr/include/aarch64-linux-gnu include \
    && ln -s usr/lib/aarch64-linux-gnu lib \ 
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_v7.h /usr/include/cudnn.h

# Unpack libnvinfer
#
RUN dpkg -x /pdk_files/libnvinfer6_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-dev_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvparsers6_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvparsers-dev_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-plugin6_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-plugin-dev_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvonnxparsers6_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvonnxparsers-dev_6.0.1-1+cuda10.0_arm64.deb /pdk_files/tensorrt 

# create stub libraries 
RUN cd /pdk_files/tensorrt \
    && ln -s usr/include/aarch64-linux-gnu include \
    && ln -s usr/lib/aarch64-linux-gnu lib \
    && cd lib \
    && mkdir stubs \
    && for x in nvinfer nvparsers nvinfer_plugin nvonnxparser; \
       do                                                     \
            CC=aarch64-linux-gnu-gcc /pdk_files/stubify.sh lib${x}.so stubs/lib${x}.so \
       ; done

# Set environment and working directory
ENV TRT_RELEASE /pdk_files/tensorrt
ENV TRT_LIB_DIR $TRT_RELEASE/lib
ENV TRT_SOURCE /workspace/TensorRT
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$TRT_LIB_DIR
WORKDIR /workspace

RUN ["/bin/bash"]
