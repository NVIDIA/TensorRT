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

# Usage: run_benchmark(batch_sizes, model_variant: (base/large), precision: (int8/int8-qat/fp16/fp32), sequence_length, max_batch_size, gpu_arch)
run_benchmark() {
BATCH_SIZES="${1}"
MODEL_VARIANT="${2}"
PRECISION="${3}"
SEQUENCE_LENGTH="${4}"
MAX_BATCH="${5}"
GPU_ARCH="${6}"

CHECKPOINTS_DIR="models/fine-tuned/bert_tf_ckpt_${MODEL_VARIANT}_qa_squad2_amp_${SEQUENCE_LENGTH}_v19.03.1"
SQUAD_DIR="BERT/squad"
ENGINE_NAME="engines/bert_${MODEL_VARIANT}_${PRECISION}_bs${MAX_BATCH}_seqlen${SEQUENCE_LENGTH}_benchmark.engine"
# QAT Checkpoint - available only for BERT-Large
QAT_CHECKPOINT="models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx"
CUDAGRAPH_PERFBIN="build/perf"
TIMING_CACHE_FILE="build.tcf"

echo "==== Benchmarking BERT ${MODEL_VARIANT} ${PRECISION} SEQLEN ${SEQUENCE_LENGTH} on ${GPU_ARCH} ===="
if [ ! -f ${ENGINE_NAME} ]; then
    if [ ! -d ${CHECKPOINTS_DIR} ]; then
        echo "Downloading checkpoints: scripts/download_model.sh ${MODEL_VARIANT} ${SEQUENCE_LENGTH}"
        scripts/download_model.sh "${MODEL_VARIANT}" "${SEQUENCE_LENGTH}"
    fi;
    if [ "${PRECISION}" == "int8-qat" ]; then
        if [ ${MODEL_VARIANT} != "large" ]; then
            echo "Skipping: BERT-base not supported for int8 (QAT)"
            return
        fi;
        if [ ! -f ${QAT_CHECKPOINT} ]; then
            echo "Downloading QAT checkpoint: scripts/download_model.sh pyt v1_1 ${MODEL_VARIANT}"
            scripts/download_model.sh pyt v1_1 "${MODEL_VARIANT}"
        fi;
        PRECISION="int8"
        BUILDER_ARGS="-x ${QAT_CHECKPOINT}"
    else
        BUILDER_ARGS="-m ${CHECKPOINTS_DIR}/model.ckpt"
    fi;
    BUILDER_ARGS="${BUILDER_ARGS} -tcf ${TIMING_CACHE_FILE} -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -c ${CHECKPOINTS_DIR} -v ${CHECKPOINTS_DIR}/vocab.txt --${PRECISION}"
    if [ "${PRECISION}" == "int8" ]; then
        BUILDER_ARGS="${BUILDER_ARGS} --fp16 --strict --calib-num 1"
        if [ "${GPU_ARCH}" == "Ampere" ] || [ "${GPU_ARCH}" == "Turing" ]; then
            BUILDER_ARGS="${BUILDER_ARGS} -iln -imh"
        elif [ "${GPU_ARCH}" == "Xavier" ]; then
            BUILDER_ARGS="${BUILDER_ARGS} -iln"
        fi;
    fi;

    echo "Building engine: python3 builder.py ${BUILDER_ARGS}"
    python3 builder.py ${BUILDER_ARGS}
fi;


if [ "${GPU_ARCH}" == "Ampere" ]; then
    # Use more iterations for faster GPUs
    NUM_ITERATIONS=2000
else
    NUM_ITERATIONS=1000
fi;
if [ -f ${CUDAGRAPH_PERFBIN} ]; then
    echo "Running benchmark with CUDA graph acceleration: perf ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME} -w 100 -i ${NUM_ITERATIONS} --enable_graph"
    ${CUDAGRAPH_PERFBIN} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME} -w 100 -i ${NUM_ITERATIONS} --enable_graph
else
    echo "Running benchmark: perf.py ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME} -w 100 -i ${NUM_ITERATIONS}"
    python3 perf.py ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME} -w 100 -i ${NUM_ITERATIONS}
fi;
echo
}

arg_gpu="Volta"
arg_help=0
while [[ "$#" -gt 0 ]]; do case $1 in
  --gpu) arg_gpu="$2"; shift;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done
if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help or -h  : Print this help menu."
    echo " --gpu <arch>  : GPU arch. Options: 'Volta', 'Xavier', 'Turing', 'Ampere'"
    exit;
fi

mkdir -p /workspace/TensorRT/demo/BERT/engines
nvidia-smi -q

# BERT BASE

## INT8
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "int8" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "int8" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "int8" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "int8" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "int8" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "int8" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "int8" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "int8" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "int8" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "int8" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "int8" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "int8" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "int8" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "int8" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "int8" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "int8" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "int8" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "int8" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "int8" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "int8" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "int8" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "int8" "384" "128" "${arg_gpu}"

## FP16
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "fp16" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "fp16" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "fp16" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "fp16" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "fp16" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "fp16" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "fp16" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "fp16" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "fp16" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "fp16" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "fp16" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "fp16" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "fp16" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "fp16" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "fp16" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "fp16" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "fp16" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "fp16" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "fp16" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "fp16" "384" "128" "${arg_gpu}"

## FP32
#run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "fp32" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "fp32" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "fp32" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "fp32" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "fp32" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "fp32" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "fp32" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "fp32" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "fp32" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "fp32" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "base" "fp32" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "base" "fp32" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "base" "fp32" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "base" "fp32" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "base" "fp32" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "base" "fp32" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "base" "fp32" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "base" "fp32" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "base" "fp32" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "base" "fp32" "384" "128" "${arg_gpu}"

# BERT LARGE

## INT8-QAT
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "int8-qat" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "int8-qat" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "int8-qat" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "int8-qat" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "int8-qat" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "int8-qat" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "int8-qat" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "int8-qat" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "int8-qat" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "int8-qat" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "int8-qat" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "int8-qat" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "int8-qat" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "int8-qat" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "int8-qat" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "int8-qat" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "int8-qat" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "int8-qat" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "int8-qat" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "int8-qat" "384" "128" "${arg_gpu}"

## INT8
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "int8" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "int8" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "int8" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "int8" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "int8" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "int8" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "int8" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "int8" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "int8" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "int8" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "int8" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "int8" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "int8" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "int8" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "int8" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "int8" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "int8" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "int8" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "int8" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "int8" "384" "128" "${arg_gpu}"

## FP16
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "fp16" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "fp16" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "fp16" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "fp16" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "fp16" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "fp16" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "fp16" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "fp16" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "fp16" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "fp16" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "fp16" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "fp16" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "fp16" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "fp16" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "fp16" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "fp16" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "fp16" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "fp16" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "fp16" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "fp16" "384" "128" "${arg_gpu}"

## FP32
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "128" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "fp32" "128" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "fp32" "128" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "fp32" "128" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "fp32" "128" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "fp32" "128" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "fp32" "128" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "fp32" "128" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "fp32" "128" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "fp32" "128" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "fp32" "128" "128" "${arg_gpu}"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "384" "32" "${arg_gpu}"
run_benchmark "-b 1" "large" "fp32" "384" "1" "${arg_gpu}"
run_benchmark "-b 2" "large" "fp32" "384" "2" "${arg_gpu}"
run_benchmark "-b 4" "large" "fp32" "384" "4" "${arg_gpu}"
run_benchmark "-b 8" "large" "fp32" "384" "8" "${arg_gpu}"
run_benchmark "-b 12" "large" "fp32" "384" "12" "${arg_gpu}"
run_benchmark "-b 16" "large" "fp32" "384" "16" "${arg_gpu}"
run_benchmark "-b 24" "large" "fp32" "384" "24" "${arg_gpu}"
run_benchmark "-b 32" "large" "fp32" "384" "32" "${arg_gpu}"
run_benchmark "-b 64" "large" "fp32" "384" "64" "${arg_gpu}"
run_benchmark "-b 128" "large" "fp32" "384" "128" "${arg_gpu}"
