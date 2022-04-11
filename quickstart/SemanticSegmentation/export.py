#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from PIL import Image
from io import BytesIO
import requests

output_image="input.ppm"

# Read sample image input and save it in ppm format
print("Exporting ppm image {}".format(output_image))
response = requests.get("https://pytorch.org/assets/images/deeplab1.png")
with Image.open(BytesIO(response.content)) as img:
    ppm = Image.new("RGB", img.size, (255, 255, 255))
    ppm.paste(img, mask=img.split()[3])
    ppm.save(output_image)


import torch
import torch.nn as nn

output_onnx="fcn-resnet101.onnx"

# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class FCN_ResNet101(nn.Module):
    def __init__(self):
        super(FCN_ResNet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)

    def forward(self, inputs):
        x = self.model(inputs)['out']
        x = x.argmax(1, keepdims=True)
        return x

model = FCN_ResNet101()
model.eval()

# Generate input tensor with random values
input_tensor = torch.rand(4, 3, 224, 224)

# Export torch model to ONNX
print("Exporting ONNX model {}".format(output_onnx))
torch.onnx.export(model, input_tensor, output_onnx,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}},
    verbose=False)

