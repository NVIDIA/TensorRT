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


"""tests of tensor quantization function and module"""
import pytest
import numpy as np

import torch
from torch.nn.parameter import Parameter

from pytorch_quantization import calib
from pytorch_quantization import cuda_ext
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

import tests.utils as test_utils
from tests.fixtures import verbose

np.random.seed(123456)  # seed 1234 causes 1 number mismatch at 6th decimal in one of the tests

# pylint:disable=missing-docstring, no-self-use


class TestTensorQuant():

    def test_simple_run(self):
        """ quantizer passes gradcheck
        """
        x = Parameter(torch.randn(2, 3, dtype=torch.float64).cuda()) * 100
        tensor_quant.tensor_quant(x, torch.max(torch.abs(x)), 7)

    def test_per_tensor_scale(self):
        """ tensor_quant matches numpy quantization
        """
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Test on GPU
        x_np = np.random.rand(1023)
        x_torch = torch.Tensor(x_np)
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)))
        quant_x_torch, _ = tensor_quant.tensor_quant(x_torch, torch.max(torch.abs(x_torch)))
        np.testing.assert_array_equal(quant_x_torch.cpu().numpy(), quant_x_np)
        torch.set_default_tensor_type('torch.FloatTensor')

    def test_per_channel_scale(self):
        """ fake_tensor_quant performs per channel quantization
        """
        x_np = np.random.rand(15, 15, 64, 128).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()

        # Pytorch filter layout seems to be KCRS, reduce max to shape [K, 1, 1, 1] to test per channel scale
        # Shrink max a little, so that clip behavior is tested
        amax_x_np = 0.7 * np.max(np.abs(x_np), axis=(1, 2, 3), keepdims=True)
        # Pytorch's max function doesn't support reduces multiple axis, and returns (max, argmax) tuple,
        # so it has to be reduced by multiple torch.max
        amax_x_torch = 0.7 * torch.max(
            torch.max(torch.max(x_torch, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]

        quant_x_np = test_utils.quant_np(x_np, amax_x_np)
        quant_x_torch, _ = tensor_quant.tensor_quant(x_torch, amax_x_torch)

        # np.testing.assert_array_equal(quant_x_torch.cpu().numpy(), quant_x_np)
        # Pytorch numerics is not the same as numpy, it will be off by 1
        np.testing.assert_array_less(np.abs(quant_x_torch.cpu().numpy() - quant_x_np), 2)
        if verbose:
            mismatches = np.where(np.abs(quant_x_torch.cpu().numpy() - quant_x_np) >= 1)
            print("Mismatches:")
            print(" Original: ", x_np[mismatches])
            print(" numpy: ", quant_x_np[mismatches])
            print(" Pytorch: ", quant_x_torch.cpu().numpy()[mismatches])

    def test_backward(self):
        """ tensor_quant implements straight through estimator on the backward pass
            Note: this does not work for integer output_dtype
        """
        x = torch.randn(3, 7, requires_grad=True).cuda()
        labels = torch.randint(6, (3,)).type(torch.LongTensor).cuda()
        quant_x, _ = tensor_quant.tensor_quant(x, x.abs().max(), 7)
        float_quant_x = quant_x.type(torch.FloatTensor).cuda()
        x.retain_grad()
        float_quant_x.retain_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = criterion(float_quant_x, labels)
        loss.backward()
        np.testing.assert_array_equal(float_quant_x.grad.cpu().numpy(), x.grad.cpu().numpy())

    def test_unsigned(self):
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np)
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)), num_bits=9, fake=False)
        quant_x_torch, _ = tensor_quant.tensor_quant(x_torch, torch.max(torch.abs(x_torch)), 8, True)
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np)

        x_torch = torch.randn(3, 7)
        with pytest.raises(TypeError, match="Negative values encountered"):
            tensor_quant.tensor_quant(x_torch, torch.max(torch.abs(x_torch)), 8, True)

    def test_overflow_fp16(self):
        x_torch = torch.randn(1023).cuda().half()
        with pytest.raises(ValueError, match="scale is too large for FP16"):
            quant_x_torch, scale = tensor_quant.tensor_quant(x_torch, torch.tensor(1e-4).cuda().half(), 8, False)

    def test_clip_gradient(self):
        x = torch.randn(3, 7, requires_grad=True).cuda()
        x.retain_grad()
        amax = x.abs().max() / 2
        x_in_range = (-amax <= x) * (x <= amax)
        quant_x, _ = tensor_quant.tensor_quant(x, amax, 8)
        loss = torch.sum((quant_x - 0.5)**2)
        loss.backward()
        np.testing.assert_array_equal(x.grad.cpu().numpy() != 0, x_in_range.cpu().numpy())

    def test_full_range(self):
        """ fake_tensor_quant uses the full integer range when narrow=False
        """
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()
        amax = np.max(np.abs(x_np))
        quant_x_np = test_utils.quant_np(x_np, amax, num_bits=9, fake=False, narrow_range=False)
        quant_x_torch, _ = tensor_quant.tensor_quant(x_torch, torch.max(torch.abs(x_torch)), 8, True, False)
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np)

class TestFakeTensorQuant():

    def test_simple_run(self):
        x = Parameter(torch.randn(3, 7).cuda())
        tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)))

    def test_per_tensor_scale(self):
        """ fake_tensor_quant matches numpy quantization
        """
        x_np = np.random.rand(13).astype('float32')
        print(x_np)
        x_torch = torch.Tensor(x_np).cuda()
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)), fake=True)
        quant_x_torch = tensor_quant.fake_tensor_quant(x_torch, torch.max(torch.abs(x_torch)))
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np)

    def test_per_channel_scale(self):
        """ fake_tensor_quant performs per channel quantization
        """
        x_np = np.random.rand(15, 15, 64, 128).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()

        # Pytorch filter layout seems to be KCRS, reduce max to shape [K, 1, 1, 1] to test per channel scale
        # Shrink max a little, so that clip behavior is tested
        amax_x_np = 0.9 * np.max(np.abs(x_np), axis=(1, 2, 3), keepdims=True)
        # Pytorch's max function doesn't support reduces multiple axis, and returns (max, argmax) tuple,
        # so it has to be reduced by multiple torch.max
        amax_x_torch = 0.9 * torch.max(
            torch.max(torch.max(x_torch, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]

        quant_x_np = test_utils.quant_np(x_np, amax_x_np, fake=True)
        quant_x_torch = tensor_quant.fake_tensor_quant(x_torch, amax_x_torch)

        # Pytorch numerics is not the same as numpy, results will be off a little
        # np.testing.assert_array_equal(quant_x_torch.cpu().numpy(), quant_x_np)
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np, decimal=2)
        if verbose:
            mismatches = np.where(np.abs(quant_x_torch.cpu().numpy() - quant_x_np) >= 1e-5)
            print("Mismatches:")
            print(" Original: ", x_np[mismatches])
            print(" numpy: ", quant_x_np[mismatches])
            print(" Pytorch: ", quant_x_torch.cpu().numpy()[mismatches])

    def test_backward(self):
        """ fake_tensor_quant implements straight through estimator on the backward pass
        """
        x = torch.randn(3, 7, requires_grad=True).cuda()
        labels = torch.randint(6, (3,)).type(torch.LongTensor).cuda()
        quant_x = tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)), 7)
        x.retain_grad()
        quant_x.retain_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = criterion(quant_x, labels)
        loss.backward()
        np.testing.assert_array_equal(quant_x.grad.cpu().numpy(), x.grad.cpu().numpy())

    def test_unsigned(self):
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)), num_bits=9, fake=True)
        quant_x_torch = tensor_quant.fake_tensor_quant(x_torch, torch.max(torch.abs(x_torch)), 8, True)
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np)

    def test_cuda_ext(self):
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()

        for num_bits in [3, 4, 5, 7, 8, 11]:
            for unsigned in [True, False]:
                test_utils.compare(
                    cuda_ext.fake_tensor_quant(x_torch, torch.max(torch.abs(x_torch)), num_bits, unsigned),
                    tensor_quant.fake_tensor_quant(x_torch, torch.max(torch.abs(x_torch)), num_bits, unsigned),
                    rtol=0, atol=0)

        # Test fp16
        x_np_fp16 = np.random.rand(1023).astype('float16')
        x_torch_fp16 = torch.Tensor(x_np_fp16).cuda().half()
        test_utils.compare(
            cuda_ext.fake_tensor_quant(x_torch_fp16, torch.max(torch.abs(x_torch_fp16))),
            tensor_quant.fake_tensor_quant(x_torch_fp16, torch.max(torch.abs(x_torch_fp16))),
            rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_cuda_ext_with_axis(self, dtype):
        x_np = np.random.rand(3, 4, 5, 6).astype(dtype)
        x_torch = torch.Tensor(x_np).cuda()

        # amax along axis 1
        amax_torch = torch.tensor([0.8, 0.9, 0.7, 0.6], device="cuda")

        for num_bits in [3, 4, 5, 7, 8, 11]:
            for unsigned in [True, False]:
                cuda_ext_out = cuda_ext.fake_tensor_quant_with_axis(x_torch, amax_torch, 1, num_bits, unsigned)
                pytorch_out = tensor_quant.fake_tensor_quant(x_torch, amax_torch.view(1, -1, 1, 1), num_bits, unsigned)
                test_utils.compare(cuda_ext_out, pytorch_out, rtol=0, atol=0)

    def test_cuda_ext_inplace(self):
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()
        quant_x_np = test_utils.quant_np(x_np, np.max(np.abs(x_np)), fake=True)
        cuda_ext.fake_tensor_quant_(x_torch, torch.max(torch.abs(x_torch)))
        np.testing.assert_array_equal(x_torch.cpu().numpy(), quant_x_np)

        # Test fp16
        x_np_fp16 = np.random.rand(1023).astype('float16')
        x_torch_fp16 = torch.Tensor(x_np_fp16).cuda().half()
        quant_x_np_fp16 = test_utils.quant_np(x_np_fp16, np.max(np.abs(x_np_fp16)), fake=True)
        cuda_ext.fake_tensor_quant_(x_torch_fp16, torch.max(torch.abs(x_torch_fp16)))
        np.testing.assert_array_almost_equal(x_torch_fp16.cpu().numpy(), quant_x_np_fp16, decimal=2)

    def test_cuda_ext_tiny_amax(self):
        x_torch = torch.rand(2, 3, 4, device="cuda")
        amax = torch.tensor([1., 1.e-26, 1.], device="cuda").unsqueeze(-1).unsqueeze(1)
        quant_x = cuda_ext.fake_tensor_quant_with_axis(x_torch, amax, axis=1)
        assert quant_x[:, 1, :].sum() == 0

    def test_overflow_fp16(self):
        x_torch = torch.randn(1023).cuda().half()
        quant_x_torch = tensor_quant.fake_tensor_quant(x_torch, torch.tensor(1e-4).cuda().half(), 8, False)
        assert not (torch.isinf(quant_x_torch).any() or torch.isnan(quant_x_torch).any())

    def test_clip_gradient(self):
        x = torch.randn(3, 7, requires_grad=True).cuda()
        x.retain_grad()
        amax = x.abs().max() / 2
        x_in_range = (-amax <= x) * (x <= amax)
        quant_x = tensor_quant.fake_tensor_quant(x, amax, 8)
        loss = torch.sum((quant_x - 0.5)**2)
        loss.backward()
        np.testing.assert_array_equal(x.grad.cpu().numpy() != 0, x_in_range.cpu().numpy())

    def test_full_range(self):
        """ fake_tensor_quant uses the full integer range when narrow=False
        """
        x_np = np.random.rand(1023).astype('float32')
        x_torch = torch.Tensor(x_np).cuda()
        amax = np.max(np.abs(x_np))
        quant_x_np = test_utils.quant_np(x_np, amax, num_bits=9, fake=True, narrow_range=False)
        quant_x_torch = tensor_quant.fake_tensor_quant(x_torch, torch.max(torch.abs(x_torch)), 8, True, False)
        np.testing.assert_array_almost_equal(quant_x_torch.cpu().numpy(), quant_x_np)

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_against_legacy(self, dtype):
        x_np = np.random.rand(3, 4, 5, 6).astype(dtype)
        x_torch = torch.Tensor(x_np).cuda()

        amax_torch = torch.tensor(0.7, device="cuda")

        for num_bits in [3, 4, 5, 7, 8, 11]:
            for unsigned in [True, False]:
                legacy_out = tensor_quant.legacy_fake_tensor_quant(x_torch, amax_torch, num_bits, unsigned)
                test_out = tensor_quant.fake_tensor_quant(x_torch, amax_torch, num_bits, unsigned)
                test_utils.compare(legacy_out, test_out, rtol=0, atol=0)

    def test_against_legacy_noncontiguous(self):
        x_np = np.random.rand(3, 4, 5, 6)
        x_torch = torch.Tensor(x_np).cuda()

        amax_torch = torch.tensor(0.7, device="cuda")

        x_torch_noncontiguous = x_torch[:, 2, :, 3]
        assert not x_torch_noncontiguous.is_contiguous()

        legacy_out = tensor_quant.legacy_fake_tensor_quant(x_torch_noncontiguous, amax_torch)
        test_out = tensor_quant.fake_tensor_quant(x_torch_noncontiguous, amax_torch)
        test_utils.compare(legacy_out, test_out, rtol=0, atol=0)


    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_against_legacy_with_axis(self, dtype):
        x_np = np.random.rand(3, 4, 5, 6).astype(dtype)
        x_torch = torch.Tensor(x_np).cuda()

        # amax along axis 1
        amax_torch = torch.tensor([0.8, 0.9, 0.7, 0.6], device="cuda").view(1, -1, 1, 1)

        for num_bits in [3, 4, 5, 7, 8, 11]:
            for unsigned in [True, False]:
                legacy_out = tensor_quant.legacy_fake_tensor_quant(x_torch, amax_torch, num_bits, unsigned)
                test_out = tensor_quant.fake_tensor_quant(x_torch, amax_torch, num_bits, unsigned)
                test_utils.compare(legacy_out, test_out, rtol=0, atol=0)

class TestQuantDescriptor():

    def test_scaled_mode(self):
        num_bits = np.random.randint(0, 16)

        test_quant_desc = tensor_quant.QuantDescriptor(num_bits=num_bits)
        assert test_quant_desc.num_bits == num_bits
        assert test_quant_desc.axis is None
        assert test_quant_desc.amax is None
        assert not test_quant_desc.learn_amax

        axis = (0, 1, 3)
        test_quant_desc = tensor_quant.QuantDescriptor(axis=axis)
        assert test_quant_desc.num_bits == 8  # default value
        assert test_quant_desc.axis == axis
        assert test_quant_desc.amax is None

        amax = 0.7
        test_quant_desc = tensor_quant.QuantDescriptor(amax=amax, unsigned=True)
        assert test_quant_desc.axis is None
        assert test_quant_desc.amax == np.float32(amax)
        assert test_quant_desc.unsigned

        amax = 0.7
        test_quant_desc = tensor_quant.QuantDescriptor(amax=amax, learn_amax=True)
        assert test_quant_desc.amax == np.float32(amax)
        assert test_quant_desc.learn_amax

        # Test the print string once if verbose is set.
        if verbose:
            print(test_quant_desc)

        with pytest.raises(TypeError, match="must be float, list or ndarray"):
            tensor_quant.QuantDescriptor(amax='oops')

        with pytest.raises(TypeError, match="amax must be float, list or ndarray"):
            tensor_quant.QuantDescriptor(amax='oops', learn_amax=True)

        with pytest.raises(TypeError, match="axis is ignored and must be None"):
            tensor_quant.QuantDescriptor(axis=(1, 2), amax=0.7, learn_amax=True)

    def test_amax(self):
        test_quant_desc = tensor_quant.QuantDescriptor()
        assert test_quant_desc.amax is None

        test_quant_desc = tensor_quant.QuantDescriptor(amax=1.2)
        assert isinstance(test_quant_desc.amax, np.ndarray)
        np.testing.assert_array_equal(test_quant_desc.amax, np.float32(1.2))

        test_quant_desc = tensor_quant.QuantDescriptor(amax=[1.3, 1.4])
        assert isinstance(test_quant_desc.amax, np.ndarray)
        np.testing.assert_array_equal(test_quant_desc.amax, np.float32([1.3, 1.4]))

        with pytest.raises(TypeError, match="must be float, list or ndarray"):
            tensor_quant.QuantDescriptor(amax='oops')

    def test_from_to_dict(self):
        quant_desc_1 = tensor_quant.QuantDescriptor(
            num_bits=2, name='a', fake_quant=True, axis=(1, 2),
            amax=3.1415926536)
        quant_desc_2 = tensor_quant.QuantDescriptor(**quant_desc_1.dict())
        if verbose:
            print(quant_desc_1.dict())
        assert quant_desc_1 == quant_desc_2

        quant_desc_1 = tensor_quant.QuantDescriptor(num_bits=2, amax=0.1, unsigned=True)
        quant_desc_2 = tensor_quant.QuantDescriptor(**quant_desc_1.dict())
        assert quant_desc_1 == quant_desc_2

    def test_from_to_yaml(self):
        quant_desc_1 = tensor_quant.QuantDescriptor(
            num_bits=2, name='a', fake_quant=True, axis=(1, 2),
            amax=3.1415926536)
        quant_desc_2 = tensor_quant.QuantDescriptor.from_yaml(quant_desc_1.to_yaml())
        if verbose:
            print(quant_desc_1.to_yaml())
        assert quant_desc_1 == quant_desc_2

        quant_desc_1 = tensor_quant.QuantDescriptor(num_bits=2, amax=0.1)
        quant_desc_2 = tensor_quant.QuantDescriptor.from_yaml(quant_desc_1.to_yaml())
        assert quant_desc_1 == quant_desc_2


class TestFakeAffineTensorQuant():

    def test_simple_run(self, verbose):
        x = np.array([-1., -13., -101., -128., 0., 2., 5., 13., 93., 111., 127.], dtype=np.float32)
        torch_x = torch.tensor(x).cuda()
        quant_x = tensor_quant.fake_affine_tensor_quant(torch_x, torch.min(torch_x), torch.max(torch_x))

        if verbose:
            print(quant_x)

        np.testing.assert_array_almost_equal(quant_x.cpu().numpy(), x)

    def test_clip_gradient(self):
        x = torch.randn(3, 7, requires_grad=True).cuda()
        x.retain_grad()
        xmin = x.min() / 2
        xmax = x.max() / 2
        x_in_range = (xmin <= x) * (x <= xmax)
        quant_x = tensor_quant.fake_affine_tensor_quant(x, xmin, xmax, 8)
        loss = torch.sum((quant_x - 0.5)**2)
        loss.backward()
        np.testing.assert_array_equal(x.grad.cpu().numpy() != 0, x_in_range.cpu().numpy())
