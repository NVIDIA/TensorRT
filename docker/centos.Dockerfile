# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

ARG CUDA_VERSION=11.0
ARG OS_VERSION=7
ARG NVCR_SUFFIX=
FROM nvidia/cuda:${CUDA_VERSION}-devel-centos${OS_VERSION}${NVCR_SUFFIX}

LABEL maintainer="NVIDIA CORPORATION"

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG wheel trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried libraries
RUN yum -y install \
    libcurl4-openssl-dev \
    wget \
    zlib-devel \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-devel \
    python3-wheel \
    unzip \
    sudo \
    make \
    build-essential

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install PyPI packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Set environment and working directory
ENV TRT_RELEASE /tensorrt
ENV TRT_SOURCE /workspace/TensorRT
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
