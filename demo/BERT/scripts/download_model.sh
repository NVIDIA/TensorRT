#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
VERSION='v2'
MODEL='large'
FT_PRECISION='fp16'
SEQ_LEN='128'
FW='tf'

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_model.sh [tf|pyt] [base|large] [fp16|fp32] [128|384] [v2|v1_1]"
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
        v2) VERSION='v2'
            ;;
        v1_1) VERSION='v1_1'
            ;;
        tf) FW='tf'
            ;;
        pyt) FW='pyt'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Prepare the download directory
mkdir -p /workspace/TensorRT/demo/BERT/models/fine-tuned
cd /workspace/TensorRT/demo/BERT/models/fine-tuned

# Download the BERT fine-tuned model
echo "Downloading BERT-${FW} ${MODEL} checkpoints with precision ${FT_PRECISION} and sequence length ${SEQ_LEN} and fine-tuned for SQuAD ${VERSION} from NGC"
if [ "${FW}" = 'tf' ]; then
    ngc registry model download-version nvidia/bert_tf_${VERSION}_${MODEL}_${FT_PRECISION}_${SEQ_LEN}:2
elif [ "${FW}" = 'pyt' ]; then
    if [ "${MODEL}" != 'large' ] || [ "${VERSION}" != 'v1_1' ]; then
        echo "Skipping. Currently only BERT-large checkpoint fine-tuned for SQuAD v1.1 available in QAT (PyTorch) workflow."
    else
        ngc registry model download-version nvidia/bert_pyt_onnx_large_qa_squad11_amp_fake_quant:1
    fi
else
    echo "Invalid framework specified for checkpoint. Run download_model.sh -h for help."
fi
