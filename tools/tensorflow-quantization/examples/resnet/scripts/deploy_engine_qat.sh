#!/usr/bin/env bash

# Single run:
# ../../engine_builder/build_engine_single.py --root_dir=/home/nvidia/PycharmProjects/tensorrt_qat_sagar/examples/resnet/weights/resnet50v1 --onnx=model_baseline_dynamic.onnx --engine=model_baseline_dynamic.engine --input=224,224,3 --min_bs=1 --max_bs=1 --opt_bs=1 --precision=fp32
#

RESNET_DEPTH=50
RESNET_VERSION=v1
QAT_SUBDIR=qat_tfrecord_ep10_steps500_l2False_baselr0.0001_piecewise_sgd_bs128
ROOT_DIR=../weights/resnet${RESNET_DEPTH}${RESNET_VERSION}/${QAT_SUBDIR}
LOGS_SUBDIR=engines_trtSource
LOGS_DIR=${ROOT_DIR}/${LOGS_SUBDIR}
mkdir $LOGS_DIR

echo "1/3. Building engine"
# bs=32 OOM in workstation
ONNX=model_dynamic.onnx
ENGINE=${LOGS_SUBDIR}/model_baseline_bs{min1,opt8,max128}.engine
python ../../../engine_builder/build_engine_single.py --root_dir=$ROOT_DIR \
                                                      --onnx=$ONNX \
                                                      --engine=$ENGINE \
                                                      --input=224,224,3 \
                                                      --min_bs=1 --opt_bs=8 --max_bs=128 \
                                                      --precision=int8
wait

for BS in 1 8 128;  do # 8 32 128; do
  echo "Model evaluation..."
  echo "############### bs=${BS} ###############"

  # Latency calculation from built engine
  echo "2/3. Latency evaluation"
  trtexec --device=0 \
          --loadEngine=${ROOT_DIR}/${ENGINE} \
          --shapes=input_1:0:${BS}x224x224x3 \
          --workspace=1024 \
          --separateProfileRun \
          --dumpProfile \
          --explicitBatch \
          --int8 &> ${LOGS_DIR}/trtexec_latency_bs${BS}.log
  wait

  echo "3/3. Accuracy evaluation"
  python ../infer_engine.py --engine=${ROOT_DIR}/${ENGINE} \
                            --log_file=engine_accuracy_bs${BS}.log \
                            --model_name=resnet_$RESNET_VERSION \
                            -b=$BS
  wait
done
