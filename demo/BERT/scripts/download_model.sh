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
SQUAD='2'
MODEL='large'
SEQ_LEN='128'
FW='tf'
WTYPE='dense'
PREC='fp16'

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_model.sh [tf|pyt] [base|large|megatron-large] [128|384] [v2|v1_1] [sparse] [int8-qat]"
            exit 0
            ;;
        base) MODEL='base'
            ;;
        large) MODEL='large'
            ;;
        megatron-large) MODEL='megatron'
            ;;
        128) SEQ_LEN='128'
            ;;
        384) SEQ_LEN='384'
            ;;
        v2) SQUAD='2'
            ;;
        v1_1) SQUAD='11'
            ;;
        tf) FW='tf'
            ;;
        pyt) FW='pyt'
            ;;
        int8-qat) PREC='int8qat'
            ;;
        sparse) WTYPE='sparse'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Prepare the download directory
mkdir -p models/fine-tuned
pushd models/fine-tuned

# Download the BERT fine-tuned model
echo "Downloading BERT-${FW} ${MODEL} checkpoints for sequence length ${SEQ_LEN} and fine-tuned for SQuAD ${SQUAD}."
if [ "${FW}" = 'tf' ]; then
    CKPT=bert_${FW}_ckpt_${MODEL}_qa_squad${SQUAD}_amp_${SEQ_LEN}
    CKPT_VERSION=19.03.1
elif [ "${FW}" = 'pyt' ]; then
    if [ "${MODEL}" == 'megatron' ]; then
        CKPT=bert_${FW}_statedict_megatron_${WTYPE}_${PREC}
        CKPT_VERSION=21.03.0
    elif [ "${MODEL}" != 'large' ] || [ "${SQUAD}" != '11' ]; then
        echo "ERROR: Only BERT-large checkpoint fine-tuned for SQuAD v1.1 available in the QAT (PyTorch) workflow."
    else
        CKPT=bert_${FW}_onnx_${MODEL}_qa_squad${SQUAD}_amp_fake_quant
        CKPT_VERSION=1
    fi
else
    echo "Invalid framework specified for checkpoint. Run download_model.sh -h for help."
fi

if [ -n "$CKPT" ]; then
    if [ -d "${CKPT}_v${CKPT_VERSION}" ]; then
        echo "Checkpoint directory ${PWD}/${CKPT}_v${CKPT_VERSION} already exists. Skip download."
    else
        ngc registry model download-version nvidia/${CKPT}:${CKPT_VERSION}
    fi
fi

popd
