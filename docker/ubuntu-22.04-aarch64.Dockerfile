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

ARG CUDA_VERSION=12.5.0

# Multi-arch container support available in non-cudnn containers.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV TRT_VERSION 10.2.0.19
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT. This will also pull in CUDNN
RUN ver="${CUDA_VERSION%.*}" &&\
    if [ "${ver%.*}" = "12" ] ; then \
        ver="12.5"; \
    fi &&\
    v="${TRT_VERSION}-1+cuda${ver}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get -y install libnvinfer10=${v} libnvonnxparsers10=${v}  libnvinfer-plugin10=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v} libnvinfer-dispatch10=${v} libnvinfer-dispatch-dev=${v} libnvinfer-lean10=${v} \
        libnvinfer-lean-dev=${v} libnvinfer-vc-plugin10=${v} libnvinfer-vc-plugin-dev=${v} \
        libnvinfer-headers-dev=${v} libnvinfer-headers-plugin-dev=${v};

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-aarch64.sh && \
    chmod +x cmake-3.21.4-linux-aarch64.sh && \
    ./cmake-3.21.4-linux-aarch64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.21.4-linux-aarch64.sh

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_arm64.zip && unzip ngccli_arm64.zip && chmod u+x ngc-cli/ngc && rm ngccli_arm64.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/aarch64-linux-gnu/
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
