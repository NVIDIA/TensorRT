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

ARG CUDA_VERSION=12.9.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 10.13.0.35
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
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

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

# Install TensorRT
RUN if [ "${CUDA_VERSION:0:2}" = "11" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && tar -xf TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && cp -a TensorRT-10.13.0.35/lib/*.so* /usr/lib/x86_64-linux-gnu \
    && pip install TensorRT-10.13.0.35/python/tensorrt-10.13.0.35-cp310-none-linux_x86_64.whl ;\
    elif [ "${CUDA_VERSION:0:2}" = "12" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && tar -xf TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && cp -a TensorRT-10.13.0.35/lib/*.so* /usr/lib/x86_64-linux-gnu \
    && pip install TensorRT-10.13.0.35/python/tensorrt-10.13.0.35-cp310-none-linux_x86_64.whl ;\
    else \
    echo "Invalid CUDA_VERSION"; \
    exit 1; \
    fi

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-Linux-x86_64.sh && \
    chmod +x cmake-3.27.9-Linux-x86_64.sh && \
    ./cmake-3.27.9-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.27.9-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="/workspace/TensorRT/build/out:${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
