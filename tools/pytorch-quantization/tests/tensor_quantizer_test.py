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


"""tests of tensor quantizer"""
import yaml
import pytest
import numpy as np

import torch

from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.nn.modules import tensor_quantizer
from pytorch_quantization import utils as quant_utils
import tests.utils as test_utils
from tests.fixtures import verbose

np.random.seed(12345)

# pylint:disable=missing-docstring, no-self-use

class TestTensorQuantizer():

    def test_simple_run(self):
        """Quantizer calls fake_tensor_quant by default"""
        x = torch.randn(3, 7).cuda()
        amax_x = torch.max(torch.abs(x))
        fn_quant_x = tensor_quant.fake_tensor_quant(x, amax_x)
        quantizer = tensor_quantizer.TensorQuantizer()
        module_quant_x = quantizer(x)
        np.testing.assert_array_equal(fn_quant_x.cpu().numpy(), module_quant_x.cpu().numpy())

    def test_simple_run_no_fake(self):
        """Quantizer fake_quant=False calls tensor_quant and sets the scale property"""
        x = torch.randn(3, 7).cuda()
        amax_x = torch.max(torch.abs(x))
        fn_quant_x, fn_scale = tensor_quant.tensor_quant(x, amax_x)
        quantizer = tensor_quantizer.TensorQuantizer(tensor_quant.QuantDescriptor(num_bits=8, fake_quant=False))
        module_quant_x = quantizer(x)
        module_scale = quantizer.scale
        np.testing.assert_array_equal(fn_quant_x.cpu().numpy(), module_quant_x.cpu().numpy())
        np.testing.assert_array_equal(fn_scale.cpu().numpy(), module_scale.cpu().numpy())

    def test_per_tensor_scale(self):
        """Quantizer performs expected quantization"""
        x_np = np.random.rand(1023)
        x_torch = torch.Tensor(x_np)
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)))
        quantizer = tensor_quantizer.TensorQuantizer(tensor_quant.QuantDescriptor(num_bits=8, fake_quant=False))
        module_quant_x = quantizer(x_torch)
        np.testing.assert_array_equal(module_quant_x.cpu().numpy(), quant_x_np)

    def test_per_channel_scale(self, verbose):
        """Quantizer performs per channel scaling"""
        x_np = np.random.rand(15, 15, 64, 128).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()

        # Pytorch filter layout seems to be KCRS, reduce max to shape [K, 1, 1, 1] to test per channel scale
        # Shrink max a little, so that clip behavior is tested
        amax_x_np = 0.7 * np.max(np.abs(x_np), axis=(1, 2, 3), keepdims=True)

        quant_x_np = test_utils.quant_np(x_np, amax_x_np)
        quantizer = tensor_quantizer.TensorQuantizer(
            tensor_quant.QuantDescriptor(num_bits=8, axis=(0), fake_quant=False, scale_amax=0.7))
        quantizer.cuda()
        module_quant_x = quantizer(x_torch)

        # np.testing.assert_array_equal(quant_x_torch.cpu().numpy(), quant_x_np)
        # Pytorch numerics is not the same as numpy, it will be off by 1
        error = np.abs(module_quant_x.cpu().numpy() - quant_x_np)
        np.testing.assert_array_less(error, 2)
        if verbose:
            mismatches = np.where(error >= 1)
            print("Mismatches:")
            print(" Original: ", x_np[mismatches])
            print(" numpy: ", quant_x_np[mismatches])
            print(" TensorQuantizer: ", module_quant_x.cpu().numpy()[mismatches])

    def test_learn_amax(self):
        """Test the clip implied by learn_amax"""
        x_np = np.random.rand(1023).astype(np.float32)
        x_torch = torch.Tensor(x_np).cuda()
        amax = 0.5
        quant_x_np = test_utils.quant_np(x_np, 0.5, fake=True)
        quantizer = tensor_quantizer.TensorQuantizer(
            tensor_quant.QuantDescriptor(num_bits=8, amax=amax, learn_amax=True)).cuda()
        assert hasattr(quantizer, 'clip')
        module_quant_x = quantizer(x_torch)
        np.testing.assert_array_equal(module_quant_x.cpu().detach().numpy(), quant_x_np)

    def test_clip_mode(self):
        """Test the clip stage only"""
        x_np = np.random.rand(1023).astype(np.float32)
        x_torch = torch.Tensor(x_np).cuda()
        amax = 0.5
        clip_x_np = np.clip(x_np, -amax, amax)
        quantizer = tensor_quantizer.TensorQuantizer(
            tensor_quant.QuantDescriptor(amax=amax, learn_amax=True), if_quant=False, if_clip=True).cuda()
        assert hasattr(quantizer, 'clip')
        module_clip_x = quantizer(x_torch)
        np.testing.assert_array_equal(module_clip_x.cpu().detach().numpy(), clip_x_np)

    def test_scale_amax(self):
        x_np = np.random.rand(1023).astype(np.float32)
        x_torch = torch.Tensor(x_np).cuda()
        amax = 0.5
        scale_amax = 0.9
        quant_x_np = test_utils.quant_np(x_np, amax * scale_amax, fake=True)
        quantizer = tensor_quantizer.TensorQuantizer(
            tensor_quant.QuantDescriptor(num_bits=8, amax=amax, scale_amax=scale_amax)).cuda()
        module_quant_x = quantizer(x_torch)
        np.testing.assert_array_equal(module_quant_x.cpu().detach().numpy(), quant_x_np)

        # Test twice. There was a but in scale amax logic that modify the amax every time
        module_quant_x = quantizer(x_torch)
        np.testing.assert_array_equal(module_quant_x.cpu().detach().numpy(), quant_x_np)

    def test_disable(self):
        x = torch.randn(3, 7).cuda()
        amax_x = torch.max(torch.abs(x))
        quantizer = tensor_quantizer.TensorQuantizer(disabled=True).cuda()
        module_quant_x = quantizer(x)
        np.testing.assert_array_equal(x.cpu().numpy(), module_quant_x.cpu().numpy())

    def test_state_loading(self):
        """Test quant_desc loading via state_dict"""
        amax = [3.142, 2.718]
        quant_desc1 = tensor_quant.QuantDescriptor(amax=amax)
        quantizer1 = tensor_quantizer.TensorQuantizer(quant_desc1)

        # copy state
        quantizer1.load_state_dict(quantizer1.state_dict())
        np.testing.assert_array_equal(quantizer1.amax.detach().cpu().numpy(), quant_desc1.amax)

    def test_properties(self):
        quant_desc1 = tensor_quant.QuantDescriptor(amax=3.14)
        quantizer1 = tensor_quantizer.TensorQuantizer(quant_desc1)
        quantizer1.amax = 0.577

        assert quantizer1.amax.detach().cpu().numpy() == np.float32(0.577)
        np.testing.assert_array_equal(quantizer1.amax.detach().cpu().numpy(), quantizer1.amax)
        assert quantizer1.step_size == 0.577 / 127.

        quant_desc2 = tensor_quant.QuantDescriptor()
        quantizer2 = tensor_quantizer.TensorQuantizer(quant_desc2)
        amax_np = np.array([3.142, 2.718], dtype=np.float32)
        quantizer2.amax = amax_np
        np.testing.assert_array_equal(quantizer2.amax.detach().cpu().numpy(), amax_np)

        quant_desc3 = tensor_quant.QuantDescriptor()
        quantizer3 = tensor_quantizer.TensorQuantizer(quant_desc3)
        assert quantizer3.amax is None

    def test_init_calib(self):
        quant_desc2 = tensor_quant.QuantDescriptor(axis=(0, 1))
        quantizer2 = tensor_quantizer.TensorQuantizer(quant_desc2, if_calib=True, if_quant=False).cuda()

        x_2 = torch.rand(127, 63, 7, 7).cuda()
        quantizer2(x_2)
        quantizer2.load_calib_amax()

        assert quantizer2.amax.numel() == 127 * 63

    def test_max_calib(self):
        axis = 0
        reduce_axis = (1, 2, 3)
        quant_desc1 = tensor_quant.QuantDescriptor(axis=axis)
        quantizer1 = tensor_quantizer.TensorQuantizer(quant_desc1).cuda()
        quantizer1.enable_calib()

        quant_desc1 = tensor_quant.QuantDescriptor(axis=axis)
        quantizer1 = tensor_quantizer.TensorQuantizer(quant_desc1).cuda()
        quantizer1.enable_calib()

        with pytest.raises(RuntimeError, match="Calibrator returned None"):
            quantizer1.load_calib_amax()

        x_1 = torch.rand(127, 63, 7, 7).cuda()
        x_2 = torch.rand(127, 63, 7, 7).cuda()
        quantizer1(x_1)
        quantizer1(x_2)
        quantizer1.disable_calib()

        global_amax = torch.max(
            quant_utils.reduce_amax(x_1, axis=reduce_axis, keepdims=True),
            quant_utils.reduce_amax(x_2, axis=reduce_axis, keepdims=True))
        test_utils.compare(quantizer1._calibrator.compute_amax(), global_amax, atol=0, rtol=0, ctol=0)

        quantizer1.load_calib_amax()
        test_utils.compare(quantizer1.amax, global_amax, atol=0, rtol=0, ctol=0)

        quant_desc2 = tensor_quant.QuantDescriptor(learn_amax=True)
        quantizer2 = tensor_quantizer.TensorQuantizer(quant_desc2).cuda()
        quantizer2.enable_calib()
        quantizer2(x_1)
        quantizer2(x_2)

        quantizer2.load_calib_amax()
        quantizer2.init_learn_amax()
        test_utils.compare(quantizer2.clip.clip_value_min, -torch.max(global_amax), atol=0, rtol=0, ctol=0)
        test_utils.compare(quantizer2.clip.clip_value_max, torch.max(global_amax), atol=0, rtol=0, ctol=0)

    def test_entropy_and_percentile_calib(self):
        """Don't really have a good way to test it."""
        quant_desc1 = tensor_quant.QuantDescriptor(calib_method='histogram')
        quantizer1 = tensor_quantizer.TensorQuantizer(quant_desc1, if_calib=True, if_quant=False).cuda()

        x_1 = torch.rand(3, 63, 7, 7).cuda()
        x_2 = torch.rand(3, 63, 7, 7).cuda()
        quantizer1(x_1)
        quantizer1(x_2)

        quantizer1.load_calib_amax("entropy")
        test_utils.compare(quantizer1._calibrator.compute_amax("entropy"), quantizer1.amax, atol=0, rtol=0, ctol=0)
        quantizer1._calibrator.reset()

        quantizer1(x_1)
        quantizer1(x_2)

        quantizer1.load_calib_amax("percentile", percentile=99.99)
        test_utils.compare(quantizer1._calibrator.compute_amax(
            "percentile", percentile=99.99), quantizer1.amax, atol=0, rtol=0, ctol=0)

    def test_setters(self):
        quantizer = tensor_quantizer.TensorQuantizer()
        quantizer.num_bits = 7
        quantizer.unsigned = True

        assert quantizer.num_bits == 7
        assert quantizer.unsigned
