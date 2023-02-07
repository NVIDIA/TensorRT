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

set -e

PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION:-3}
PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION:-8}
TARGET=${TARGET_ARCHITECTURE:-x86_64}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
ROOT_PATH=${TRT_OSSPATH:-/workspace/TensorRT}
EXT_PATH=${EXT_PATH:-/tmp/external}
WHEEL_OUTPUT_DIR=${ROOT_PATH}/python/build

mkdir -p ${WHEEL_OUTPUT_DIR}
pushd ${WHEEL_OUTPUT_DIR}

# Generate tensorrt.so
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DTARGET=${TARGET} \
         -DPYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
         -DPYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
         -DEXT_PATH=${EXT_PATH} \
         -DCUDA_INCLUDE_DIRS=${CUDA_ROOT}/include \
         -DTENSORRT_ROOT=${ROOT_PATH} \
         -DTENSORRT_BUILD=${ROOT_PATH}/build/
make -j12

# Generate wheel
TRT_MAJOR=$(awk '/^\#define NV_TENSORRT_MAJOR/ {print $3}' ${ROOT_PATH}/include/NvInferVersion.h)
TRT_MINOR=$(awk '/^\#define NV_TENSORRT_MINOR/ {print $3}' ${ROOT_PATH}/include/NvInferVersion.h)
TRT_PATCH=$(awk '/^\#define NV_TENSORRT_PATCH/ {print $3}' ${ROOT_PATH}/include/NvInferVersion.h)
TRT_BUILD=$(awk '/^\#define NV_TENSORRT_BUILD/ {print $3}' ${ROOT_PATH}/include/NvInferVersion.h)
TRT_VERSION=${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}
TRT_MAJMINPATCH=${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}

echo "Generating python ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} bindings for TensorRT ${TRT_VERSION}"

expand_vars_cp () {
    test -f ${1} || (echo "ERROR: File: ${1} does not exist!" && exit 1); \
    sed -e "s|\#\#TENSORRT_VERSION\#\#|${TRT_VERSION}|g" \
        -e "s|\#\#TENSORRT_MAJMINPATCH\#\#|${TRT_MAJMINPATCH}|g" \
        ${1} > ${2}
}

pushd ${ROOT_PATH}/python/packaging
for dir in $(find . -type d); do mkdir -p ${WHEEL_OUTPUT_DIR}/$dir; done
for file in $(find . -type f); do expand_vars_cp $file ${WHEEL_OUTPUT_DIR}/${file}; done
popd
python3 setup.py -q bdist_wheel --python-tag=cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION} --plat-name=linux_${TARGET}

popd
