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

FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 10.14.1.48
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG wheel trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried packages
RUN dnf -y groupinstall "Development Tools"
RUN dnf -y install \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    wget \
    perl-core \
    git \
    pkg-config \
    unzip \
    sudo

# Install python3
RUN dnf install -y python38 python38-devel &&\
    cd /usr/bin && ln -s /usr/bin/pip3.8 pip;


# Install TensorRT
RUN if [ "${CUDA_VERSION:0:2}" = "13" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz \
    && tar -xf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz \
    && cp -a TensorRT-10.14.1.48/lib/*.so* /usr/lib64 \
    && pip install TensorRT-10.14.1.48/python/tensorrt-10.14.1.48-cp38-none-linux_x86_64.whl ;\
    elif [ "${CUDA_VERSION:0:2}" = "12" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && tar -xf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && cp -a TensorRT-10.14.1.48/lib/*.so* /usr/lib64 \
    && pip install TensorRT-10.14.1.48/python/tensorrt-10.14.1.48-cp38-none-linux_x86_64.whl ;\
    else \
    echo "Invalid CUDA_VERSION"; \
    exit 1; \
    fi

# Install PyPI packages
RUN pip install --upgrade pip
RUN pip install setuptools>=41.0.0
RUN pip install numpy
RUN pip install jupyter jupyterlab

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-Linux-x86_64.sh && \
    chmod +x cmake-3.27.9-Linux-x86_64.sh && \
    ./cmake-3.27.9-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.27.9-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

RUN ln -s /usr/bin/python3 /usr/bin/python

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib64
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="/workspace/TensorRT/build/out:${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
