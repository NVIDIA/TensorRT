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
Test for QuantConvTransposed
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


class TestQuantConvTranspose2D():

    def test_no_quant(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 32, 32)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv_transpose2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_tensor(self):
        kernel_size = 8

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor())
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(256, _NUM_IN_CHANNELS, 32, 32)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv_transpose2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL)
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256, 256)

        weight_copy = quant_conv_object.weight.clone()

        amax = quant_utils.reduce_amax(weight_copy, axis=(1, 2, 3))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose2d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_input(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(20, _NUM_IN_CHANNELS, 50, 50)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.conv_transpose2d(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=False, quant_desc_weight=QuantDescriptor())
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv_transpose2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor(axis=(1)))
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(0, 2, 3))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3, axis=(1))

        quant_conv_object = quant_conv.QuantConvTranspose2d(
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

        out1 = F.conv_transpose2d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantDescriptor(axis=(1)))
        test_input = torch.randn(2, _NUM_IN_CHANNELS, 2, 2)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(0, 2, 3))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose2d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 32, 32).cuda()

        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        fake_quant_conv2d = quant_conv.QuantConvTranspose2d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(1)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        conv2d = nn.ConvTranspose2d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv2d(test_input)
        output = conv2d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-5, atol=2e-4)


class TestQuantConvTranspose3D():

    def test_no_quant(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 32, 32, 32)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv_transpose3d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3, axis=(1))

        quant_conv_object = quant_conv.QuantConvTranspose3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight)
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16, 16, 16)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = F.conv_transpose3d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL)
        test_input = torch.randn(2, _NUM_IN_CHANNELS, 2, 2, 2)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(1, 2, 3, 4))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose3d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 32, 32, 32).cuda()

        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        fake_quant_conv3d = quant_conv.QuantConvTranspose3d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(1)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        conv3d = nn.ConvTranspose3d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv3d(test_input)
        output = conv3d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-5, atol=2e-4)


class TestQuantConvTranspose1D():

    def test_no_quant(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 32)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = F.conv_transpose1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_tensor(self):
        kernel_size = 8

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor())
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(256, _NUM_IN_CHANNELS, 32)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv_transpose1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL)
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 256)

        weight_copy = quant_conv_object.weight.clone()

        amax = quant_utils.reduce_amax(weight_copy, axis=(1, 2))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose1d(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_input(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(20, _NUM_IN_CHANNELS, 50)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.conv_transpose1d(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=False, quant_desc_weight=QuantDescriptor())
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.conv_transpose1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantDescriptor(axis=(1)))
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 16)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(0, 2))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_prec(self):
        kernel_size = 3

        quant_desc_input = QuantDescriptor(num_bits=4)
        quant_desc_weight = QuantDescriptor(num_bits=3, axis=(1))

        quant_conv_object = quant_conv.QuantConvTranspose1d(
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

        out1 = F.conv_transpose1d(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_bias(self):
        kernel_size = 3

        quant_conv_object = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantDescriptor(axis=(1)))
        test_input = torch.randn(2, _NUM_IN_CHANNELS, 2)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=(0, 2))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.conv_transpose1d(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_against_unquantized(self):
        kernel_size = 3
        test_input = torch.randn(16, _NUM_IN_CHANNELS, 24).cuda()

        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        fake_quant_conv1d = quant_conv.QuantConvTranspose1d(
            _NUM_IN_CHANNELS,
            _NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantDescriptor(num_bits=16),
            quant_desc_weight=QuantDescriptor(num_bits=16, axis=(1)))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        conv1d = nn.ConvTranspose1d(_NUM_IN_CHANNELS, _NUM_OUT_CHANNELS, kernel_size, bias=True)

        fake_quant_output = fake_quant_conv1d(test_input)
        output = conv1d(test_input)

        test_utils.compare(fake_quant_output, output, rtol=1e-5, atol=1e-4)
