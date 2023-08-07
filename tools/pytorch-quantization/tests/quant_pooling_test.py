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


"""tests of QuantPooling module.
Most tests check the functionality of all the combinations in Quant Pooling against the corresponding functionalities
in tensor_quant.
"""
import pytest
import numpy as np

import torch
import torch.nn.functional as F

from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import quant_pooling

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

np.random.seed(1234)
torch.manual_seed(1234)

# pylint:disable=missing-docstring, no-self-use
class TestQuantMaxPool1d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantMaxPool1d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantMaxPool1d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.max_pool1d(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantMaxPool2d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.max_pool2d(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_variable_bits(self):
        # Repeat checking the output for variable number of bits to QuantDescriptor
        for bits in [2, 4, 6]:
            quant_desc_input = tensor_quant.QuantDescriptor(num_bits=bits)

            quant_pooling.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
            quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

            test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

            quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)), bits)

            out1 = F.max_pool2d(quant_input, 3, 1, 0, 1, False, False)
            out2 = quant_pooling_object(test_input)
            np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.max_pool2d(test_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_multi_axis(self):
        quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8, axis=(0, 1))

        quant_pooling.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(16, 7, 5, 5, dtype=torch.double)
        input_amax = torch.amax(torch.abs(test_input), dim=(2, 3), keepdim=True)
        quant_input = tensor_quant.fake_tensor_quant(test_input, input_amax)

        out1 = F.max_pool2d(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantMaxPool3d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantMaxPool3d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantMaxPool3d(kernel_size=3, stride=1)

        test_input = torch.randn(5, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.max_pool3d(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAvgPool1d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAvgPool1d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAvgPool1d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.avg_pool1d(quant_input, 3, 1, 0, False, True)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAvgPool2d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.avg_pool2d(quant_input, 3, 1, 0, False, True, None)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_variable_bits(self):
        # Repeat checking the output for variable number of bits to QuantDescriptor
        for bits in [2, 4, 6]:
            quant_desc_input = tensor_quant.QuantDescriptor(num_bits=bits)

            quant_pooling.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
            quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1)

            test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

            quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)), bits)

            out1 = F.avg_pool2d(quant_input, 3, 1, 0, False, True, None)
            out2 = quant_pooling_object(test_input)
            np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.avg_pool2d(test_input, 3, 1, 0, False, True, None)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAvgPool3d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAvgPool3d(kernel_size=3, stride=1,
                                                                quant_desc_input=
                                                                tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAvgPool3d(kernel_size=3, stride=1)

        test_input = torch.randn(5, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.avg_pool3d(quant_input, 3, 1, 0, False, True, None)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAdaptiveAvgPool1d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool1d(output_size=3,
                                                                        quant_desc_input=
                                                                        tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool1d(output_size=3)

        test_input = torch.randn(1, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.adaptive_avg_pool1d(quant_input, 3)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAdaptiveAvgPool2d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3,
                                                                        quant_desc_input=
                                                                        tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.adaptive_avg_pool2d(quant_input, 3)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_variable_bits(self):
        # Repeat checking the output for variable number of bits to QuantDescriptor
        for bits in [2, 4, 6]:
            quant_desc_input = tensor_quant.QuantDescriptor(num_bits=bits)

            quant_pooling.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(quant_desc_input)
            quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3)

            test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

            quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)), bits)

            out1 = F.adaptive_avg_pool2d(quant_input, 3)
            out2 = quant_pooling_object(test_input)
            np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.adaptive_avg_pool2d(test_input, 3)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())

class TestQuantAdaptiveAvgPool3d():

    def test_raise(self):
        with pytest.raises(ValueError) as excinfo:
            quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool3d(output_size=3,
                                                                        quant_desc_input=
                                                                        tensor_quant.QuantDescriptor(fake_quant=False))
        assert "Only fake quantization is supported" in str(excinfo.value)

    # Quantizing activations
    def test_input_fake_quant(self):
        quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool3d(output_size=3)

        test_input = torch.randn(5, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = F.adaptive_avg_pool3d(quant_input, 3)
        out2 = quant_pooling_object(test_input)
        np.testing.assert_array_equal(out1.detach().cpu().numpy(), out2.detach().cpu().numpy())
