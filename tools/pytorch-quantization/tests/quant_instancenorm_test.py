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


"""tests of QuantInstanceNorm module.
Mose tests check the functionality of all the combinations in Quant instancenorm against the corresponding functionalities in
tensor_quant. There are tests for all the three QuantInstaceNorm1D, QuantInstanceNorm2D, and QuantInstanceNorm3D
"""
import pytest
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import utils as quant_utils
from pytorch_quantization.nn.modules import quant_instancenorm
#import tests.utils as test_utils

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.backends.cudnn.deterministic = True

np.random.seed(1234)

# pylint:disable=missing-docstring, no-self-use
NUM_CHANNELS = 15

class TestQuantInstanceNorm1D():

    def test_no_quant(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm1d(NUM_CHANNELS, affine=True)
        quant_instancenorm_object.input_quantizer.disable()

        test_input = torch.randn(8, NUM_CHANNELS, 128)

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(test_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm1d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor())

        test_input = torch.randn(8, NUM_CHANNELS, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm1d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor(axis=(1)))

        test_input = torch.randn(8, NUM_CHANNELS, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input,
                                                     torch.abs(test_input).max(0, keepdim=True)[0].max(2, keepdim=True)[0])

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())


class TestQuantInstanceNorm2D():

    def test_no_quant(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm2d(NUM_CHANNELS, affine=True)
        quant_instancenorm_object.input_quantizer.disable()

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128)

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(test_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm2d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor())

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm2d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor(axis=(1)))

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input,
                                                     torch.abs(test_input).max(0, keepdim=True)[0].max(2, keepdim=True)[0].max(3, keepdim=True)[0])

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())



class TestQuantInstanceNorm3D():

    def test_no_quant(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm3d(NUM_CHANNELS, affine=True)
        quant_instancenorm_object.input_quantizer.disable()

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128, 128)

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(test_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm3d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor())

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):

        quant_instancenorm_object = quant_instancenorm.QuantInstanceNorm3d(NUM_CHANNELS, affine=True,
                                                                           quant_desc_input=QuantDescriptor(axis=(1)))

        test_input = torch.randn(8, NUM_CHANNELS, 128, 128, 128)
        quant_input = tensor_quant.fake_tensor_quant(test_input,
                                                     torch.abs(test_input).max(0, keepdim=True)[0].max(2, keepdim=True)[0]
                                                          .max(3, keepdim=True)[0].max(4, keepdim=True)[0])

        out1 = quant_instancenorm_object(test_input)
        out2 = F.instance_norm(quant_input,
                               quant_instancenorm_object.running_mean,
                               quant_instancenorm_object.running_var,
                               quant_instancenorm_object.weight,
                               quant_instancenorm_object.bias)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())
