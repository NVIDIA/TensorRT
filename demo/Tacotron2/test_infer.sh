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

BATCH_SIZE=1
INPUT_LENGTH=128
NUM_ITERS=1003 # extra 3 iterations for warmup
TACOTRON2_CKPT="nvidia_tacotron2pyt_fp16_20190427"
WAVEGLOW_CKPT="nvidia_waveglow256pyt_fp16"
RUN_MODE="" # = fp32
LOG_RUN_MODE="gpu_fp32"
TEST_PROGRAM="test_infer.py"
WN_CHANNELS=512
LOG_SUFFIX_ADD="" #additional info, e.g., GPU type

while [ -n "$1" ]
do
    case "$1" in
	-bs|--batch-size)
	    BATCH_SIZE="$2"
	    shift
	    ;;
	-il|--input-length)
	    INPUT_LENGTH="$2"
	    shift
	    ;;
	--num-iters)
	    NUM_ITERS="$2"
	    shift
	    ;;
	--test)
	    TEST_PROGRAM="$2"
	    shift
	    ;;
	--tacotron2)
	    TACOTRON2_CKPT="$2"
	    shift
	    ;;
	--encoder)
	    ENCODER_CKPT="$2"
	    shift
	    ;;
	--decoder)
	    DECODER_CKPT="$2"
	    shift
	    ;;
	--postnet)
	    POSTNET_CKPT="$2"
	    shift
	    ;;
	--waveglow)
	    WAVEGLOW_CKPT="$2"
	    shift
	    ;;
	--wn-channels)
	    WN_CHANNELS="$2"
	    shift
	    ;;
	--cpu)
	    RUN_MODE="--cpu"
	    LOG_RUN_MODE="cpu_fp32"
	    ;;
	--fp16)
	    RUN_MODE="--fp16"
	    LOG_RUN_MODE="gpu_fp16"
	    ;;
	--log-suffix)
	    LOG_SUFFIX_ADD="$2"
	    shift
	    ;;
	*)
	    echo "Option $1 not recognized"
    esac
    shift
done

LOG_SUFFIX=bs${BATCH_SIZE}_il${INPUT_LENGTH}_${LOG_RUN_MODE}_wn${WN_CHANNELS}_${LOG_SUFFIX_ADD}
NVLOG_FILE=nvlog_${LOG_SUFFIX}.json
TMP_LOGFILE=tmp_log_${LOG_SUFFIX}.log
LOGFILE=log_${LOG_SUFFIX}.log


if [ "$TEST_PROGRAM" = "tensorrt/test_infer_trt.py" ]
then
    TACOTRON2_PARAMS="--encoder $ENCODER_CKPT --decoder $DECODER_CKPT --postnet $POSTNET_CKPT"
else
    TACOTRON2_PARAMS="--tacotron2 $TACOTRON2_CKPT"
fi

set -x
python3 $TEST_PROGRAM \
       $TACOTRON2_PARAMS \
       --waveglow $WAVEGLOW_CKPT \
       --batch-size $BATCH_SIZE \
       --input-length $INPUT_LENGTH \
       --log-file $NVLOG_FILE \
       --num-iters $NUM_ITERS \
       --wn-channels $WN_CHANNELS \
       $RUN_MODE \
       |& tee $TMP_LOGFILE
set +x


PERF=$(cat $TMP_LOGFILE | grep -F 'Throughput average (samples/sec)' | awk -F'= ' '{print $2}')
NUM_MELS=$(cat $TMP_LOGFILE | grep -F 'Number of mels per audio average' | awk -F'= ' '{print $2}')
LATENCY=$(cat $TMP_LOGFILE | grep -F 'Latency average (seconds)' | awk -F'= ' '{print $2}')
LATENCYSTD=$(cat $TMP_LOGFILE | grep -F 'Latency std (seconds)' | awk -F'= ' '{print $2}')
LATENCY50=$(cat $TMP_LOGFILE | grep -F 'Latency cl 50 (seconds)' | awk -F'= ' '{print $2}')
LATENCY90=$(cat $TMP_LOGFILE | grep -F 'Latency cl 90 (seconds)' | awk -F'= ' '{print $2}')
LATENCY95=$(cat $TMP_LOGFILE | grep -F 'Latency cl 95 (seconds)' | awk -F'= ' '{print $2}')
LATENCY99=$(cat $TMP_LOGFILE | grep -F 'Latency cl 99 (seconds)' | awk -F'= ' '{print $2}')

echo "$BATCH_SIZE,$INPUT_LENGTH,$LOG_RUN_MODE,$NUM_ITERS,$LATENCY,$LATENCYSTD,$LATENCY50,$LATENCY90,$LATENCY95,$LATENCY99,$PERF,$NUM_MELS" | tee $LOGFILE
