# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

FROM nvcr.io/nvidia/tensorrt:19.05-py3
ARG myuid
ARG mygid

RUN  echo $myuid
RUN  echo $mygid

# TODO: Depending on the docker version, this might work without mapping the user for home dir access
RUN groupadd -r -g ${mygid} nb && useradd -r -u ${myuid} -g ${mygid} -ms /bin/bash nb

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y pbzip2 pv bzip2  sudo gcc-7 g++-7  zlib1g-dev g++-4.9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                                 --slave /usr/bin/g++ g++ /usr/bin/g++-7  && \
                                 update-alternatives --config gcc

RUN wget https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh && \
    sh cmake-3.14.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir
RUN pip install tensorflow==1.13.1 && pip install horovod

RUN echo 'nb:abc123' | chpasswd

RUN chown nb /workspace

USER nb

WORKDIR /workspace

RUN git clone https://github.com/NVlabs/cub.git
RUN git clone https://github.com/NVIDIA/cutlass.git

