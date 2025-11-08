#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARG CUDA_VERSION=13.0.0
ARG OS_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 10.14.1.48
ENV DEBIAN_FRONTEND=noninteractive

# Setup user account and edit default account
RUN if id "ubuntu" &>/dev/null; then \
      usermod -u 1234 ubuntu && \
      groupmod -g 1234 ubuntu; \
    fi
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried libraries + aarch64 toolchains
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    build-essential \
    g++-aarch64-linux-gnu

# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel \
    python3-venv &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Create python3 virtualenv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-Linux-x86_64.sh && \
    chmod +x cmake-3.27.9-Linux-x86_64.sh && \
    ./cmake-3.27.9-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.27.9-Linux-x86_64.sh

# Install CUDA cross compile toolchain
RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-cross-sbsa-ubuntu2404-13-0-local_13.0.0-1_all.deb && \
    dpkg -i cuda-repo-cross-sbsa-ubuntu2404-13-0-local_13.0.0-1_all.deb && \
    cp /var/cuda-repo-cross-sbsa-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-cross-sbsa-13-0

# Unpack libnvinfer.

RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.aarch64-gnu.cuda-13.0.tar.gz && \
    tar -xf TensorRT-10.14.1.48.Linux.aarch64-gnu.cuda-13.0.tar.gz && \
    cp -a TensorRT-10.14.1.48/lib/*.so* /usr/lib/aarch64-linux-gnu

# Link required library
RUN cd /usr/aarch64-linux-gnu/lib && ln -sf librt.so.1 librt.so

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/aarch64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
