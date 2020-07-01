#!/usr/bin/env bash
#
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
#

arg_dockerfile=docker/ubuntu
arg_imagename=tensorrt-ubuntu
arg_gpus=all
arg_trtrelease=$TRT_RELEASE
arg_trtsource=$TRT_SOURCE
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --tag) arg_imagename="$2"; shift;;
  --gpus) arg_gpus="$2"; shift;;
  --release) arg_trtrelease="$2"; shift;;
  --source) arg_trtsource="$2"; shift;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help or -h          : Print this help menu."
    echo " --tag     <imagename> : Image name for the generated container."
    echo " --gpus    <number>    : Number of GPUs visible in container. Set 'none' to disable, and 'all' to make all visible."
    echo " --release <path>      : Path to TensorRT release build."
    echo " --source  <path>      : Path to TensorRT open source codebase."
    exit;
fi

extra_args=""
if [ "$arg_gpus" != "none" ]; then
    extra_args="$extra_args --gpus $arg_gpus"
fi

docker_args="$extra_args -v $arg_trtrelease:/tensorrt -v $arg_trtsource:/workspace/TensorRT -it $arg_imagename:latest"

echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
