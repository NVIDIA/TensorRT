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


# Setup default parameters (if no command-line parameters given)
MODEL='base'
FT_PRECISION='fp16'
SEQ_LEN='384'

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname ${SCRIPT})
TENSORRT_DIR=${SCRIPT_DIR}/../../../

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh build_examples.sh [base|large] [fp16|fp32] [128|384]"
            exit 0
            ;;
        base) MODEL='base'
            ;;
        large) MODEL='large'
            ;;
        fp16) FT_PRECISION='fp16'
            ;;
        fp32) FT_PRECISION='fp32'
            ;;
        128) SEQ_LEN='128'
            ;;
        384) SEQ_LEN='384'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

echo "Downloading BERT-${MODEL} with fine-tuned precision ${FT_PRECISION} and sequence length ${SEQ_LEN} from NGC"

# Build the BERT plugins
echo "Building TensorRT plugins for BERT"

BERT_DIR=${TENSORRT_DIR}/demo/BERT
cd $BERT_DIR
mkdir -p build
cd build
cmake ..
make -j

# Download/install the NVIDIA NGC CLI
echo "Downloading NGC CLI for retrieving the BERT fine-tuned model"
BIN_DIR=/workspace/bin
mkdir -p $BIN_DIR
cd $BIN_DIR
wget https://ngc.nvidia.com/downloads/ngccli_bat_linux.zip
unzip ngccli_bat_linux.zip
chmod u+x ngc
# Remove zip file and checksum file
rm ngccli_bat_linux.zip ngc.md5
export PATH=$PATH:$BIN_DIR
echo "Setup NGC configuration for download"
echo "no-apikey\nascii\nno-org\nno-team\nno-ace\n" | ngc config set

# Download the BERT fine-tuned model
mkdir -p /workspace/models/fine-tuned
cd /workspace/models/fine-tuned
ngc registry model download-version nvidia/bert_tf_v2_${MODEL}_${FT_PRECISION}_${SEQ_LEN}:2
