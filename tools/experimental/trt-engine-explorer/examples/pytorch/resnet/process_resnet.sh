#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Use this script to process the four example ResNet engines from resnet_example.py
# and generate the JSON files from exploration with TREx.


SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
PROCESS_ENGINE=$SCRIPT_DIR/../../../utils/process_engine.py
ONNX_DIR=$SCRIPT_DIR/generated

# Batch size = 32
SHAPES='shapes=input.1:32x3x224x224'

python3 $PROCESS_ENGINE $ONNX_DIR/resnet.onnx $ONNX_DIR/fp32 $SHAPES
python3 $PROCESS_ENGINE $ONNX_DIR/resnet.onnx $ONNX_DIR/fp16 $SHAPES fp16
python3 $PROCESS_ENGINE $ONNX_DIR/resnet-qat.onnx $ONNX_DIR/qat $SHAPES best
python3 $PROCESS_ENGINE $ONNX_DIR/resnet-qat-residual.onnx $ONNX_DIR/qat-residual $SHAPES best
python3 $PROCESS_ENGINE $ONNX_DIR/resnet-qat-residual-qgap.onnx $ONNX_DIR/qat-residual-qgap $SHAPES best
