#!/bin/bash

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

# Make sure to run this script from directory up (bert_tensorrt) as:
#     sh scripts/create_docker_container.sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname ${SCRIPT})
TENSORRT_DIR=${SCRIPT_DIR}/../../../

# Build the docker image using the provided Docker file
DOCKERFILE_DIR=${TENSORRT_DIR}/demo/BERT
docker build --build-arg myuid=$(id -u) --build-arg mygid=$(id -g) --rm -t bert-tensorrt $DOCKERFILE_DIR

# Launch the docker container based on the image we just created
docker run -it --rm \
    --name bert-tensorrt \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=1 \
    --ulimit stack=67108864 \
    -u $(id -u):$(id -g) \
    -v ${HOME}/models:/models \
    -v ${TENSORRT_DIR}:/workspace/TensorRT \
    bert-tensorrt
