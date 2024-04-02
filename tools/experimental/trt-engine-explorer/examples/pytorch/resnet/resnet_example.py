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


# This script generates 4 different ResNet models:
# 1. FP32 Resnet.
# 2. QAT Resnet, with only the Convolutions quantized.
# 3. QAT Resnet, with Convolutions and residual-connections quantized.
# 4. QAT Resnet, with only the Convolutions, residual-connections, and GAP quantized.
#
# GAP is Global Average Pooling.


import os
import torch
from qat_model import resnet18

# For QAT
from pytorch_quantization import nn as quant_nn
quant_nn.TensorQuantizer.use_fb_fake_quant = True

# Create the directory which will store the generated models.
output_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(output_dir, "generated")
try:
    os.mkdir(output_dir)
except OSError as error:
    pass


input = torch.randn(1, 3, 224, 224)


resnet = resnet18(
    pretrained=True,
    quantize=False).eval()

with torch.no_grad():
    torch.onnx.export(
        resnet, input,
        os.path.join(output_dir, "resnet.onnx"),
        input_names=["input.1"],
        opset_version=13,
        dynamic_axes={"input.1": {0: "batch_size"}})


resnet = resnet18(
    pretrained=True,
    quantize=True).eval()

with torch.no_grad():
    torch.onnx.export(
        resnet, input,
        os.path.join(output_dir, "resnet-qat.onnx"),
        input_names=["input.1"],
        opset_version=13,
        dynamic_axes={"input.1": {0: "batch_size"}})


resnet = resnet18(
    pretrained=True,
    quantize=True,
    quantize_residual=True).eval()

with torch.no_grad():
    input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        resnet, input,
        os.path.join(output_dir, "resnet-qat-residual.onnx"),
        input_names=["input.1"],
        opset_version=13,
        dynamic_axes={"input.1": {0: "batch_size"}})


resnet = resnet18(
    pretrained=True,
    quantize=True,
    quantize_residual=True,
    quantize_gap=True).eval()

with torch.no_grad():
    input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        resnet, input,
        os.path.join(output_dir, "resnet-qat-residual-qgap.onnx"),
        input_names=["input.1"],
        opset_version=13,
        dynamic_axes={"input.1": {0: "batch_size"}})


