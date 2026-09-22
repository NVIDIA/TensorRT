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

ARG CUDA_VERSION=13.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION=11.3.0.99
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 22.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
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
    build-essential \
    libnccl2 \
    libnccl-dev \
    openmpi-bin \
    libopenmpi-dev \
    zstd \
    ccache

# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.31.11/cmake-3.31.11-Linux-x86_64.sh && \
    chmod +x cmake-3.31.11-Linux-x86_64.sh && \
    ./cmake-3.31.11-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.31.11-Linux-x86_64.sh

# Install gtest
RUN cd /tmp && \
    git clone https://github.com/google/googletest.git -b v1.14.0 && \
    cd googletest && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    rm -rf /tmp/googletest

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Install TensorRT
ARG CUDA_VERSION
COPY docker/downloadTRT.sh /tmp/downloadTRT.sh
RUN case "${CUDA_VERSION}" in \
        13.*) TRT_CUDA_VERSION=13.4 ;; \
        12.*) TRT_CUDA_VERSION=12.9 ;; \
        *) echo "Unsupported CUDA_VERSION: ${CUDA_VERSION}" && exit 1 ;; \
    esac && \
    /tmp/downloadTRT.sh --x86 --cuda "${TRT_CUDA_VERSION}"

# Set environment and working directory
ENV TRT_ROOT=/opt/TensorRT-$TRT_VERSION
ENV TRT_LIBPATH=/opt/TensorRT-$TRT_VERSION/lib
ENV TRT_OSSPATH=/workspace/TensorRT
ENV PATH="/workspace/TensorRT/build/out:${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
