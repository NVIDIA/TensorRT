ROOT_DIR="/home/nvidia/PycharmProjects/tensorrt_qat/examples/resnet/"

RESNET_DEPTH="50"
RESNET_VERSION="v1"
MODEL_TYPE="baseline"  # "qat"
PRECISION="fp32"  # "int8"

ENGINES_DIR="engines_gtc_trt8.4_gittrt/${MODEL_TYPE}"
LOGS_DIR="logs_gtc_trt8.4_gittrt/${MODEL_TYPE}"

for BS in 1; do
  SUBDIR="resnet${RESNET_DEPTH}${RESNET_VERSION}_${PRECISION}_${BS}_sparsity_disable_DLA_disabled"

  python ../infer_engine.py --engine=${ROOT_DIR}/${ENGINES_DIR}/${SUBDIR}.plan \
                            --log_file=${ROOT_DIR}/${LOGS_DIR}/${SUBDIR}_accuracy.log \
                            --model_name=resnet_$RESNET_VERSION -b=1
done
