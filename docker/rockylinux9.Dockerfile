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

FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux9
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION=11.3.0.99
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG wheel trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install python3
RUN dnf install -y python39 python3-devel && \
    cd /usr/bin && rm pip && ln -s /usr/bin/pip3.9 pip;

# Install PyPI packages
RUN pip install --upgrade pip
RUN pip install setuptools>=41.0.0
RUN pip install numpy
RUN pip install jupyter jupyterlab

# Install requried packages
RUN dnf -y --nobest groupinstall "Development Tools"
RUN dnf -y install \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    wget \
    perl-core \
    git \
    pkg-config \
    unzip \
    sudo \
    libnccl \
    libnccl-devel \
    openmpi \
    openmpi-devel \
    zstd \
    epel-release

RUN dnf -y install ccache

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

RUN ln -s /usr/bin/python3 /usr/bin/python

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
