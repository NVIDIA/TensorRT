#!/bin/bash
#
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
#

# Setup default parameters (if no command-line parameters given)
VERSION='v1.1'

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_squad.sh [v2_0|v1_1]"
            exit 0
            ;;
        v2_0) VERSION='v2.0'
            ;;
        v1_1) VERSION='v1.1'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Download the SQuAD training and dev datasets
echo "Downloading SQuAD-${VERSION} training and dev datasets"
mkdir -p squad
pushd squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-${VERSION}.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-${VERSION}.json
popd
