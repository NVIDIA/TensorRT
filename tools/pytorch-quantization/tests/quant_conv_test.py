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


"""tests of QuantConv module.
Mose tests check the functionality of all the combinations in Quant conv against the corresponding functionalities in
tensor_quant. There are tests for all the three QuantConv1D, QuantConv2D, and QuantConv3D
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
from pytorch_quantization.nn.modules import quant_conv
import tests.utils as test_utils

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.backends.cudnn.deterministic = True

np.random.seed(1234)

# pylint:disable=missing-docstring, no-self-use

_NUM_IN_CHANNELS = 13
_NUM_OUT_CHANNELS = 17

class TestQuantConv2D():
    #Quantizing weight

    def test_no_quant(self):

        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256, 256)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_tensor(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor())
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256, 256)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256, 256)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1, 1))

        out1 = F.conv2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_in_feature_fake_quant(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256, 256)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.conv2d(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=False, quant_desc_weight=QuantDescriptor())
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=False,
                                                   quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1, 1))

        out1 = F.conv2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3)

        quant_conv_object = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = F.conv2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv2d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True,
                                                   quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1, 1))

        out1 = F.conv2d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 24, 24).cuda()

        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        fake_quant_conv2d = quant_conv.QuantConv2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(0)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        conv2d = nn.Conv2d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv2d(test_input)
        output = conv2d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-6, atol=1.5e-4)


    def test_set_default_quant_desc(self):
        quant_conv_layer = quant_conv.Conv2d(32, 257, 3)
        assert quant_conv_layer.input_quantizer._axis == None
        assert quant_conv_layer.weight_quantizer._axis == (0)

        # set default to a different one
        quant_desc_input = QuantDescriptor(num_bits=11)
        quant_desc_weight = QuantDescriptor(num_bits=13, axis=(1))
        quant_conv.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_conv.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)

        # Create one with default descriptor
        quant_conv_layer = quant_conv.Conv2d(32, 257, 3)
        # Check quant_desc in quantizer created with default descriptor
        assert quant_conv_layer.input_quantizer._num_bits == quant_desc_input.num_bits
        assert quant_conv_layer.weight_quantizer._axis == quant_desc_weight.axis

        # Test default is per class
        quant_conv_layer = quant_conv.Conv3d(31, 255, 5)
        assert quant_conv_layer.input_quantizer._num_bits != quant_desc_input.num_bits
        assert quant_conv_layer.weight_quantizer._axis != quant_desc_weight.axis

        # Reset default
        quant_conv.QuantConv2d.set_default_quant_desc_input(QuantDescriptor())
        quant_conv.QuantConv2d.set_default_quant_desc_weight(QuantDescriptor(axis=(0)))

    def test_unused_kwargs(self):
        with pytest.raises(TypeError, match="Unused keys"):
            quant_conv.Conv2d(32, 257, 3, descriptor='oops')

class TestQuantConv1D():

    def test_no_quant(self):
        kernel_size = 8

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_tensor(self):
        kernel_size = 8

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor())
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor(axis=(0)))
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256)

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(1, 2))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_input(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(20, _NUM_IN_CHANNELS, 50)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.conv1d(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=False, quant_desc_weight=QuantDescriptor())
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor(axis=(0)))
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1))

        out1 = F.conv1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3, axis=(0))

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = F.conv1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantDescriptor(axis=(0)))
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1))

        out1 = F.conv1d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 24).cuda()

        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        fake_quant_conv1d = quant_conv.QuantConv1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(0)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        conv1d = nn.Conv1d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv1d(test_input)
        output = conv1d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-5, atol=1e-4)


class TestQuantConv3D():
    #Quantizing weight

    def test_no_quant(self):
        kernel_size = 8

        quant_conv_object = quant_conv.QuantConv3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 8, 8, 8)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv3d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3, axis=(0))

        quant_conv_object = quant_conv.QuantConv3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 8, 8, 8)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = F.conv3d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConv3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantDescriptor(axis=(0)))
        test_input = torch.randn(8, _NUM_IN_CHANNELS, 8, 8, 8)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy,
            torch.max(torch.abs(weight_copy).view(_NUM_OUT_CHANNELS, -1), dim=1, keepdim=True)[0].view(
                _NUM_OUT_CHANNELS, 1, 1, 1, 1))

        out1 = F.conv3d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 24, 24, 24).cuda()

        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        fake_quant_conv3d = quant_conv.QuantConv3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(0)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        conv3d = nn.Conv3d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv3d(test_input)
        output = conv3d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-6, atol=2e-4)
