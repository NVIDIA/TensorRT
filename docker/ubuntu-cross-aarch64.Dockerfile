#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG CUDA_VERSION=11.4.1
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.5.3.1
ENV DEBIAN_FRONTEND=noninteractive

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
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

# Skip installing PyPI packages and NGC client on cross-build container

COPY docker/jetpack_files /pdk_files
COPY scripts/stubify.sh /pdk_files

# Update CUDA signing keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# Install CUDA cross compile toolchain
RUN dpkg -i /pdk_files/cuda-repo-cross-aarch64*.deb /pdk_files/cuda-repo-ubuntu*_amd64.deb \
    && apt-get update \
    && apt-get install -y cuda-cross-aarch64 \
    && rm -rf /var/lib/apt/lists/*

# Unpack cudnn
RUN  dpkg -x /pdk_files/cudnn-local-repo*.deb /pdk_files/cudnn_extract \
    && dpkg -x /pdk_files/cudnn_extract/var/cudnn-local-repo*/libcudnn[7-8]_*-1+cuda11.[0-9]_arm64.deb /pdk_files/cudnn \
    && dpkg -x /pdk_files/cudnn_extract/var/cudnn-local-repo*/libcudnn[7-8]-dev_*-1+cuda11.[0-9]_arm64.deb /pdk_files/cudnn \
    && cd /pdk_files/cudnn/usr/lib/aarch64-linux-gnu \
    && cd /pdk_files/cudnn \
    && ln -s usr/include/aarch64-linux-gnu include \
    && ln -s usr/lib/aarch64-linux-gnu lib \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_adv_infer_v[7-9].h /usr/include/cudnn_adv_infer.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_adv_train_v[7-9].h /usr/include/cudnn_adv_train.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_backend_v[7-9].h /usr/include/cudnn_backend.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_cnn_infer_v[7-9].h /usr/include/cudnn_cnn_infer.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_cnn_train_v[7-9].h /usr/include/cudnn_cnn_train.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_ops_infer_v[7-9].h /usr/include/cudnn_ops_infer.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_ops_train_v[7-9].h /usr/include/cudnn_ops_train.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_v[7-9].h /usr/include/cudnn.h \
    && ln -s /pdk_files/cudnn/usr/include/aarch64-linux-gnu/cudnn_version_v[7-9].h /usr/include/cudnn_version.h

# Unpack libnvinfer
RUN dpkg -x /pdk_files/libnvinfer[0-8]_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-dev_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvparsers[6-8]_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvparsers-dev_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-plugin[6-8]_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvinfer-plugin-dev_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvonnxparsers[6-8]_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt \
    && dpkg -x /pdk_files/libnvonnxparsers-dev_*-1+cuda11.[0-9]_arm64.deb /pdk_files/tensorrt

# Clean up debs
RUN rm -rf /pdk_files/*.deb

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
ENV TRT_LIBPATH /pdk_files/tensorrt/lib
ENV TRT_OSSPATH /workspace/TensorRT
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
