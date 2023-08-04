#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests of calibrators"""
import inspect
import pytest
import numpy as np

import torch

from pytorch_quantization import utils as quant_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
import tests.utils as test_utils
from examples.torchvision.models.classification import *
from tests.fixtures import verbose
from tests.fixtures.models import QuantLeNet

np.random.seed(12345)
torch.manual_seed(12345)

# pylint:disable=missing-docstring, no-self-use


class TestExampleModels():

    def test_resnet50(self):
        model = resnet50(pretrained=True, quantize=True)
        model.eval()
        model.cuda()
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
        if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
            torch.onnx.export(model,
                              dummy_input,
                              "/tmp/resnet50.onnx",
                              verbose=False,
                              opset_version=13,
                              enable_onnx_checker=False,
                              do_constant_folding=True)
        else:
            torch.onnx.export(model,
                              dummy_input,
                              "/tmp/resnet50.onnx",
                              verbose=False,
                              opset_version=13,
                              do_constant_folding=True)
        quant_nn.TensorQuantizer.use_fb_fake_quant = False

    def test_resnet50_cpu(self):
        model = resnet50(pretrained=True, quantize=True)
        model.eval()

        for name, module in model.named_modules():
            if name.endswith('_quantizer'):
                module.amax = 2.50

        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        dummy_input = torch.randn(1, 3, 224, 224)
        if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
            torch.onnx.export(model,
                              dummy_input,
                              "/tmp/resnet50_cpu.onnx",
                              verbose=False,
                              opset_version=13,
                              enable_onnx_checker=False,
                              do_constant_folding=True)
        else:
            torch.onnx.export(model,
                              dummy_input,
                              "/tmp/resnet50.onnx",
                              verbose=False,
                              opset_version=13,
                              do_constant_folding=True)
        quant_nn.TensorQuantizer.use_fb_fake_quant = False
