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
ARG OS_VERSION=18.04
ARG NVCR_SUFFIX=
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}${NVCR_SUFFIX}

LABEL maintainer="NVIDIA CORPORATION"

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown -R trtuser:trtuser /workspace

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
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    g++-powerpc64le-linux-gnu \
    libc6-powerpc-cross

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh
# Download ppc Cudnn, Cublas, Cudart, RT, nvprof
# TODO Remove once packages are added to cuda cross compiler
RUN wget http://cuda-repo/release-candidates/Libraries/cuDNN/v8.0/8.0.2.5_20200617_28575977/11.0.x-r445/Installer/Ubuntu18_04-ppc64le/libcudnn8_8.0.2.5-1+cuda11.0_ppc64el.deb && \
    wget http://cuda-repo/release-candidates/Libraries/cuDNN/v8.0/8.0.2.5_20200617_28575977/11.0.x-r445/Installer/Ubuntu18_04-ppc64le/libcudnn8-dev_8.0.2.5-1+cuda11.0_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/libcublas-dev-11-0_11.0.0.191-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/libcublas-11-0_11.0.0.191-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-cudart-11-0_11.0.171-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-cudart-dev-11-0_11.0.171-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvrtc-11-0_11.0.167-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvrtc-dev-11-0_11.0.167-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvcc-11-0_11.0.167-1_ppc64el.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-nvprof-11-0_11.0.167-1_ppc64el.deb

# Unpack Cublas
RUN dpkg -x libcublas-11-0_11.0.0.191-1_ppc64el.deb cublas && \
    dpkg -x libcublas-dev-11-0_11.0.0.191-1_ppc64el.deb cublas && \
    cp -r cublas/* /

# Unpack Cudart
RUN dpkg -x cuda-cudart-11-0_11.0.171-1_ppc64el.deb cudart && \
    dpkg -x cuda-cudart-dev-11-0_11.0.171-1_ppc64el.deb cudart && \
    cp -r cudart/* /

# Unpack RT
RUN dpkg -x cuda-nvrtc-11-0_11.0.167-1_ppc64el.deb rt && \
    dpkg -x cuda-nvrtc-dev-11-0_11.0.167-1_ppc64el.deb rt && \
    cp -r rt/* /

# Unpack Cudnn
RUN dpkg -x libcudnn8_8.0.2.5-1+cuda11.0_ppc64el.deb cudnn && \
    dpkg -x libcudnn8-dev_8.0.2.5-1+cuda11.0_ppc64el.deb cudnn && \
    cp -r cudnn/* /

# Unpack NVCC, and copy headers
RUN dpkg -x cuda-nvcc-11-0_11.0.167-1_ppc64el.deb nvcc && \
    cp -r nvcc/usr/local/cuda-11.0/targets/ppc64le-linux/include/*  /usr/local/cuda-11.0/targets/ppc64le-linux/include/

# Install nvprof
RUN dpkg -x cuda-nvprof-11-0_11.0.167-1_ppc64el.deb prof && \
    cp -r prof/* /

# Clean up temporary files
RUN rm -rf cublas cudart rt prof nvcc
RUN rm *.deb

WORKDIR /workspace
ENV TRT_RELEASE /tensorrt
ENV TRT_SOURCE /workspace/TensorRT
USER trtuser
RUN ["/bin/bash"]
