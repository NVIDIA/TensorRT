# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

ARG CUDA_VERSION=10.2
ARG OS_VERSION=18.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

LABEL maintainer="NVIDIA CORPORATION"

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
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

# Install PyPI packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Put the lib I had to get off of Jetson
COPY docker/libnvrtc.so.10.2 /usr/local/cuda-10.2/targets/aarch64-linux/lib/

# Get nvidia libs (they came from SDK manager, I've unpacked and copied to S3
RUN cd / && wget https://edge-impulse-cdn-prod-new.s3-eu-west-1.amazonaws.com/pdk_files.zip
RUN unzip pdk_files.zip

COPY scripts/stubify.sh /pdk_files

# Unpack cudnn
RUN cd /pdk_files/cudnn/usr/lib/aarch64-linux-gnu \
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

# create stub libraries
RUN cd /pdk_files/tensorrt \
    && ln -s usr/include/aarch64-linux-gnu include \
    && ln -s usr/lib/aarch64-linux-gnu lib 

# Get the cross compiler
RUN cd /pdk_files && wget https://edge-impulse-cdn-prod-new.s3-eu-west-1.amazonaws.com/cuda-repo-cross-aarch64-10-2-local-10.2.89_1.0-1_all.deb

# # Install CUDA cross compile toolchain
RUN sudo chmod 777 /pdk_files/cuda-repo-cross-aarch64-10-2-local-10.2.89_1.0-1_all.deb && \
    dpkg -i /pdk_files/cuda-repo-cross-aarch64*.deb \
    && apt-get update \
    && apt-get install -y cuda-cross-aarch64 \
    && rm -rf /var/lib/apt/lists/*

# TensorRT (wrapper API, "OSS")
COPY . /workspace/TensorRT/
RUN cd /workspace/TensorRT && \
    git submodule update --init --recursive

# Set environment and working directory
ENV TRT_LIBPATH /pdk_files/tensorrt/lib
ENV TRT_OSSPATH /workspace/TensorRT

# Build libraries
RUN cd ${TRT_OSSPATH} && \
    mkdir -p build && cd build && \
    cmake .. -DTRT_LIB_DIR=${TRT_LIBPATH} -DTRT_OUT_DIR=/workspace/TensorRT/build/out -DCMAKE_TOOLCHAIN_FILE=${TRT_OSSPATH}/cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=10.2 \
    -DGPU_ARCHS="53" -DCUDNN_LIB=../../../pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so \
    -DCUBLAS_LIB=/usr/lib/aarch64-linux-gnu/libcublas.so -DCUBLASLT_LIB=/usr/lib/x86_64-linux-gnu/libcublasLt.so \
    -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF && \
    make ei -j

USER trtuser
RUN ["/bin/bash"]
