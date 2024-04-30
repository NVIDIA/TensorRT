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

import torch
import torchvision.models as models

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# load model; We are going to use a pretrained resnet model
model = models.resnet50(pretrained=True).eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)

# Export the model
torch.onnx.export(model,                        # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'],    # the model's output names
                  )
