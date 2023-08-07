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


"""tests of QuantLinear module.
Most tests check the functionality of all the combinations in Quant Linear against the corresponding functionalities
in tensor_quant.
"""
import pytest
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import utils as quant_utils
from pytorch_quantization.nn.modules import quant_linear
import tests.utils as test_utils

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

np.random.seed(1234)
torch.manual_seed(1234)

# pylint:disable=missing-docstring, no-self-use


class TestQuantLinear():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_linear_object = quant_linear.QuantLinear(
                7, 9, bias=False, quant_desc_weight=tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    #Quantizing weight
    def test_weight_fake_per_tensor(self):
        with torch.cuda.device(0):
            size = 256
            quant_linear_object = quant_linear.QuantLinear(
                size,
                size,
                bias=False,
                quant_desc_weight=tensor_quant.QuantDescriptor(axis=None))
            quant_linear_object.input_quantizer.disable()
            test_input = torch.randn(size, size)

            weight_copy = quant_linear_object.weight.clone()
            quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

            out1 = F.linear(test_input, quant_weight)
            out2 = quant_linear_object(test_input)
            np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_weight_fake_per_channel(self):
        size_in = 255
        size_out = 257
        quant_linear_object = quant_linear.QuantLinear(
            size_in, size_out, bias=False,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW)
        quant_linear_object.input_quantizer.disable()
        test_input = torch.randn(32, size_in)

        weight_copy = quant_linear_object.weight.clone()
        amax = quant_utils.reduce_amax(weight_copy, axis=1, keepdims=True)
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = F.linear(test_input, quant_weight)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    # Quantizing activations
    def test_test_input_fake_per_tensor(self):
        size_in = 255
        size_out = 257
        quant_linear_object = quant_linear.QuantLinear(
            size_in, size_out, bias=False)
        quant_linear_object.weight_quantizer.disable()
        test_input = torch.randn(32, size_in)

        weight_copy = quant_linear_object.weight.clone()
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.linear(quant_input, weight_copy)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor(self):
        """quantize everything, activations will scaled per tensor in ALL cases"""
        size_in = 255
        size_out = 257
        quant_linear_object = quant_linear.QuantLinear(
            size_in, size_out, bias=False, quant_desc_weight=tensor_quant.QuantDescriptor())
        test_input = torch.randn(32, size_in)

        weight_copy = quant_linear_object.weight.clone()
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.linear(quant_input, quant_weight)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_tensor_with_bias(self):
        """quantize everything, activations will scaled per tensor in ALL cases"""
        size_in = 255
        size_out = 257
        quant_linear_object = quant_linear.QuantLinear(
            size_in, size_out, bias=False, quant_desc_weight=tensor_quant.QuantDescriptor())
        test_input = torch.randn(32, 17, 93, size_in)  # Test input other than 2 dimensional

        weight_copy = quant_linear_object.weight.clone()
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, torch.max(torch.abs(weight_copy)))

        out1 = F.linear(quant_input, quant_weight, bias=quant_linear_object.bias)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel(self):
        """quantize everything, activations will scaled per tensor in ALL cases"""
        size_in = 255
        size_out = 257
        quant_linear_object = quant_linear.QuantLinear(size_in, size_out, bias=False,
                                                       quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW)
        test_input = torch.randn(32, size_in)

        weight_copy = quant_linear_object.weight.clone()
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy,
                                                      torch.max(torch.abs(weight_copy), dim=1, keepdim=True)[0])

        out1 = F.linear(quant_input, quant_weight)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_per_channel_other_precs(self):
        """Test some precisions other than 8bit."""
        size_in = 255
        size_out = 257
        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=4)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=3)
        quant_linear_object = quant_linear.QuantLinear(
            size_in,
            size_out,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight)
        weight_quantizer = TensorQuantizer(quant_desc_weight)
        test_input_quantizer = TensorQuantizer(quant_desc_input)

        test_input = torch.randn(32, size_in)

        weight_copy = quant_linear_object.weight.clone()
        quant_input = test_input_quantizer(test_input)
        quant_weight = weight_quantizer(weight_copy)

        out1 = F.linear(quant_input, quant_weight)
        out2 = quant_linear_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_fake_quant_against_unquantized(self):
        """
        Quantized Linear should introduce bounded error compare to Linear
        """
        size_in = 255
        size_out = 257
        test_input = torch.randn(32, size_in).cuda()

        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        quant_linear_layer = quant_linear.QuantLinear(
            size_in,
            size_out,
            bias=True,
            quant_desc_input=tensor_quant.QuantDescriptor(num_bits=16),
            quant_desc_weight=tensor_quant.QuantDescriptor(num_bits=16, axis=0))

        # Reset seed. Make sure weight and bias are the same
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        linear_layer = nn.Linear(size_in, size_out, bias=True)

        quant_out_features = quant_linear_layer(test_input)
        out_features = linear_layer(test_input)

        # The difference between Linear and QuantLinear should be bounded in a range
        # Small values which become 0 after quantization lead to large relative errors. rtol and atol could be
        # much smaller without those values
        np.testing.assert_allclose(
            quant_out_features.detach().cpu().numpy(), out_features.detach().cpu().numpy(), rtol=0.01, atol=1e-4)

    def test_set_default_quant_desc(self):
        quant_linear_layer = quant_linear.QuantLinear(32, 257)
        assert quant_linear_layer.input_quantizer.axis == None
        assert quant_linear_layer.weight_quantizer.axis == (0)

        # set default to a different one
        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=11)
        quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=13, axis=1)
        quant_linear.Linear.set_default_quant_desc_input(quant_desc_input)
        quant_linear.Linear.set_default_quant_desc_weight(quant_desc_weight)

        # Create one with default descriptor
        quant_linear_layer = quant_linear.QuantLinear(32, 257)
        # Check quant_desc in quantizer created with default descriptor
        assert quant_linear_layer.input_quantizer.num_bits == quant_desc_input.num_bits
        assert quant_linear_layer.weight_quantizer.axis == quant_desc_weight.axis

    def test_unused_kwargs(self):
        with pytest.raises(TypeError, match="Unused keys"):
            quant_linear_layer = quant_linear.QuantLinear(32, 257, descriptor='oops')
