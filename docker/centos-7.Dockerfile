#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG CUDA_VERSION=12.2.0

# TODO: Update - unused in 22.09
FROM nvidia/cuda:${CUDA_VERSION}-devel-centos7
LABEL maintainer="NVIDIA CORPORATION"

ARG CUDA_VERSION_MAJOR_MINOR=12.2
ENV NV_CUDNN_VERSION 8.9.6.50-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda${CUDA_VERSION_MAJOR_MINOR}
ENV NV_CUDNN_PACKAGE_DEV libcudnn8-devel-${NV_CUDNN_VERSION}.cuda${CUDA_VERSION_MAJOR_MINOR}

ENV TRT_VERSION 9.2.0.5
SHELL ["/bin/bash", "-c"]

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG wheel trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried packages
RUN yum -y groupinstall "Development Tools"
RUN yum -y install \
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
RUN yum install -y python36 python3-devel

# yum needs to use python2
RUN sed -i "1s/python/python2/" /usr/bin/yum

# Install TensorRT
RUN if [ "${CUDA_VERSION:0:2}" = "11" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-11.8.tar.gz \
        && tar -xf tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-11.8.tar.gz \
        && cp -a TensorRT-9.2.0.5/lib/*.so* /usr/lib64 \
        && pip3 install TensorRT-9.2.0.5/python/tensorrt-9.2.0.post11.dev5-cp36-none-linux_x86_64.whl ;\
elif [ "${CUDA_VERSION:0:2}" = "12" ]; then \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz \
        && tar -xf tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz \
        && cp -a TensorRT-9.2.0.5/lib/*.so* /usr/lib64 \
        && pip3 install TensorRT-9.2.0.5/python/tensorrt-9.2.0.post12.dev5-cp36-none-linux_x86_64.whl ;\
else \
    echo "Invalid CUDA_VERSION"; \
    exit 1; \
fi

# Install dev-toolset-8 for g++ version that supports c++14
RUN yum -y install centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum -y install devtoolset-8

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install numpy
RUN pip3 install jupyter jupyterlab

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib64
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
# Use devtoolset-8 as default compiler
ENV PATH="/opt/rh/devtoolset-8/root/bin:${PATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
