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

# Usage: run_benchmark(batch_sizes, model_variant: (large), precision: (int8-qat), sequence_length, max_batch_size, gpu_arch: Ampere, weights: (dense/sparse))
run_benchmark() {
BATCH_SIZES="${1}"
MODEL_VARIANT="${2}"
PRECISION="${3}"
SEQUENCE_LENGTH="${4}"
MAX_BATCH="${5}"
GPU_ARCH="${6}"
WEIGHTS="${7}"

CHECKPOINTS_DIR="models/fine-tuned/bert_tf_ckpt_${MODEL_VARIANT}_qa_squad2_amp_${SEQUENCE_LENGTH}_v19.03.1"
SQUAD_DIR="BERT/squad"
ENGINE_NAME="engines/bert_mt_${MODEL_VARIANT}_${PRECISION}_bs${MAX_BATCH}_seqlen${SEQUENCE_LENGTH}_weight${WEIGHTS}_benchmark.engine"
QAT_CHECKPOINT="models/fine-tuned/bert_pyt_statedict_megatron_sparse_int8qat_v21.03.0/bert_pyt_statedict_megatron_sparse_int8_qat"
CUDAGRAPH_PERFBIN="build/perf"
TIMING_CACHE_FILE="build_megatron.tcf"

if [ "${PRECISION}" != "int8-qat" ]; then
    echo "Skipping: Megatron-BERT only supported for int8 (QAT)"
    return
elif [ "${GPU_ARCH}" != "Ampere" ]; then
    echo "Skipping: Sparsity only supported on Ampere GPUs"
    return
fi

echo "==== Benchmarking BERT ${MODEL_VARIANT} ${PRECISION} SEQLEN ${SEQUENCE_LENGTH} on ${GPU_ARCH} ===="
if [ ! -f ${ENGINE_NAME} ]; then
    if [ ! -d ${CHECKPOINTS_DIR} ]; then
        echo "Downloading checkpoints: scripts/download_model.sh ${MODEL_VARIANT} ${SEQUENCE_LENGTH}"
        scripts/download_model.sh "${MODEL_VARIANT}" "${SEQUENCE_LENGTH}"
    fi;
    if [ "${PRECISION}" == "int8-qat" ]; then
        if [ ${MODEL_VARIANT} != "large" ]; then
            echo "Skipping: Megatron-BERT-base not supported for int8 (QAT)"
            return
        fi;
        if [ ! -f ${QAT_CHECKPOINT} ]; then
            echo "Downloading QAT checkpoint: scripts/download_model.sh pyt megatron-${MODEL_VARIANT} ${PRECISION} sparse"
            scripts/download_model.sh pyt megatron-${MODEL_VARIANT} ${PRECISION} sparse
        fi;
        PRECISION="int8"
        BUILDER_ARGS="--pickle ${QAT_CHECKPOINT}"
    fi;
    BUILDER_ARGS="${BUILDER_ARGS} -tcf ${TIMING_CACHE_FILE} -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -c ${CHECKPOINTS_DIR} -v ${CHECKPOINTS_DIR}/vocab.txt --megatron"
    if [ "${WEIGHTS}" == "sparse" ]; then
        BUILDER_ARGS="${BUILDER_ARGS} --sp"
    fi;
    if [ "${PRECISION}" == "int8" ]; then
        BUILDER_ARGS="${BUILDER_ARGS} --fp16 --int8 --strict"
        if [ "${GPU_ARCH}" == "Ampere" ]; then
            BUILDER_ARGS="${BUILDER_ARGS} -il"
        fi;
    fi;

    echo "Building engine: python3 builder_varseqlen.py ${BUILDER_ARGS}"
    python3 builder_varseqlen.py ${BUILDER_ARGS}
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
    python3 perf_varseqlen.py ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME} -w 100 -i ${NUM_ITERATIONS}
fi;
echo
}

arg_gpu="Ampere"
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
fi;

mkdir -p /workspace/TensorRT/demo/BERT/engines
nvidia-smi -q

# BERT LARGE

## INT8-QAT (dense)
# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "128" "32" "${arg_gpu}" "dense"
run_benchmark "-b 1" "large" "int8-qat" "128" "1" "${arg_gpu}" "dense"
run_benchmark "-b 2" "large" "int8-qat" "128" "2" "${arg_gpu}" "dense"
run_benchmark "-b 4" "large" "int8-qat" "128" "4" "${arg_gpu}" "dense"
run_benchmark "-b 8" "large" "int8-qat" "128" "8" "${arg_gpu}" "dense"
run_benchmark "-b 12" "large" "int8-qat" "128" "12" "${arg_gpu}" "dense"
run_benchmark "-b 16" "large" "int8-qat" "128" "16" "${arg_gpu}" "dense"
run_benchmark "-b 24" "large" "int8-qat" "128" "24" "${arg_gpu}" "dense"
run_benchmark "-b 32" "large" "int8-qat" "128" "32" "${arg_gpu}" "dense"
run_benchmark "-b 64" "large" "int8-qat" "128" "64" "${arg_gpu}" "dense"
run_benchmark "-b 128" "large" "int8-qat" "128" "128" "${arg_gpu}" "dense"

# run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "384" "32" "${arg_gpu}" "dense"
run_benchmark "-b 1" "large" "int8-qat" "384" "1" "${arg_gpu}" "dense"
run_benchmark "-b 2" "large" "int8-qat" "384" "2" "${arg_gpu}" "dense"
run_benchmark "-b 4" "large" "int8-qat" "384" "4" "${arg_gpu}" "dense"
run_benchmark "-b 8" "large" "int8-qat" "384" "8" "${arg_gpu}" "dense"
run_benchmark "-b 12" "large" "int8-qat" "384" "12" "${arg_gpu}" "dense"
run_benchmark "-b 16" "large" "int8-qat" "384" "16" "${arg_gpu}" "dense"
run_benchmark "-b 24" "large" "int8-qat" "384" "24" "${arg_gpu}" "dense"
run_benchmark "-b 32" "large" "int8-qat" "384" "32" "${arg_gpu}" "dense"
run_benchmark "-b 64" "large" "int8-qat" "384" "64" "${arg_gpu}" "dense"
run_benchmark "-b 128" "large" "int8-qat" "384" "128" "${arg_gpu}" "dense"

## INT8-QAT (sparse)
if [ "${arg_gpu}" == "Ampere" ]; then
    # run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "128" "32" "${arg_gpu}" "sparse"
    run_benchmark "-b 1" "large" "int8-qat" "128" "1" "${arg_gpu}" "sparse"
    run_benchmark "-b 2" "large" "int8-qat" "128" "2" "${arg_gpu}" "sparse"
    run_benchmark "-b 4" "large" "int8-qat" "128" "4" "${arg_gpu}" "sparse"
    run_benchmark "-b 8" "large" "int8-qat" "128" "8" "${arg_gpu}" "sparse"
    run_benchmark "-b 12" "large" "int8-qat" "128" "12" "${arg_gpu}" "sparse"
    run_benchmark "-b 16" "large" "int8-qat" "128" "16" "${arg_gpu}" "sparse"
    run_benchmark "-b 24" "large" "int8-qat" "128" "24" "${arg_gpu}" "sparse"
    run_benchmark "-b 32" "large" "int8-qat" "128" "32" "${arg_gpu}" "sparse"
    run_benchmark "-b 64" "large" "int8-qat" "128" "64" "${arg_gpu}" "sparse"
    run_benchmark "-b 128" "large" "int8-qat" "128" "128" "${arg_gpu}" "sparse"

    # run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "int8-qat" "384" "32" "${arg_gpu}" "sparse"
    run_benchmark "-b 1" "large" "int8-qat" "384" "1" "${arg_gpu}" "sparse"
    run_benchmark "-b 2" "large" "int8-qat" "384" "2" "${arg_gpu}" "sparse"
    run_benchmark "-b 4" "large" "int8-qat" "384" "4" "${arg_gpu}" "sparse"
    run_benchmark "-b 8" "large" "int8-qat" "384" "8" "${arg_gpu}" "sparse"
    run_benchmark "-b 12" "large" "int8-qat" "384" "12" "${arg_gpu}" "sparse"
    run_benchmark "-b 16" "large" "int8-qat" "384" "16" "${arg_gpu}" "sparse"
    run_benchmark "-b 24" "large" "int8-qat" "384" "24" "${arg_gpu}" "sparse"
    run_benchmark "-b 32" "large" "int8-qat" "384" "32" "${arg_gpu}" "sparse"
    run_benchmark "-b 64" "large" "int8-qat" "384" "64" "${arg_gpu}" "sparse"
    run_benchmark "-b 128" "large" "int8-qat" "384" "128" "${arg_gpu}" "sparse"
else
    echo "Sparsity only supported on Ampere GPUs. Skip benchmark."
fi;
