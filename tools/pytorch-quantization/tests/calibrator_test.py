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
import pytest
import numpy as np

import torch

from pytorch_quantization import utils as quant_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
import tests.utils as test_utils
from tests.fixtures import verbose
from tests.fixtures.models import QuantLeNet

np.random.seed(12345)
torch.manual_seed(12345)

# pylint:disable=missing-docstring, no-self-use

class TestMaxCalibrator():

    def test_simple_run(self):
        max_calibrator = calib.MaxCalibrator(8, None, False)

        x_1 = torch.rand(129).cuda()
        x_2 = torch.rand(127).cuda()
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        test_utils.compare(max_calibrator.compute_amax(), torch.max(x_1.max(), x_2.max()), atol=0, rtol=0, ctol=0)

        # Nothing to test other than creation
        max_calibrator = calib.MaxCalibrator(8, None, True)

    def test_fine_grain(self):
        axis = 0
        reducs_axis = (1, 2, 3)
        max_calibrator = calib.MaxCalibrator(8, axis, False)

        x_1 = torch.rand(31, 63, 7, 7).cuda()
        x_2 = torch.rand(31, 63, 7, 7).cuda()
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        assert max_calibrator.compute_amax().shape[0] == 31

        test_utils.compare(max_calibrator.compute_amax(),
                           quant_utils.reduce_amax(torch.max(x_1, x_2), axis=reducs_axis),
                           atol=0, rtol=0, ctol=0)

        max_calibrator.reset()
        assert max_calibrator.compute_amax() is None

    def test_raises(self):
        axis = 0
        max_calibrator = calib.MaxCalibrator(8, axis, False)

        x_2 = torch.rand(32, 63, 7, 7).cuda()
        x_3 = torch.rand(33, 63, 7, 7).cuda()
        max_calibrator.collect(x_2)
        with pytest.raises(RuntimeError, match="shape changed"):
            max_calibrator.collect(x_3)

    def test_track_amax(self):
        max_calibrator = calib.MaxCalibrator(8, None, False, track_amax=True)

        x_1 = torch.rand(129).cuda()
        x_2 = torch.rand(127).cuda()
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        test_utils.compare(max_calibrator.compute_amax(), torch.max(x_1.max(), x_2.max()), atol=0, rtol=0, ctol=0)
        np.testing.assert_array_equal(max_calibrator.amaxs[0], x_1.max().cpu().numpy())
        np.testing.assert_array_equal(max_calibrator.amaxs[1], x_2.max().cpu().numpy())

    def test_repr(self):
        max_calibrator = calib.MaxCalibrator(8, None, False, track_amax=True)
        repr(max_calibrator)

class TestHistogramCalibrator():

    def test_grow(self, verbose):
        x_1 = torch.tensor([0, 255, 255, 255, 255, 255]).cuda()
        x_2 = torch.tensor([0, 255, 255, 255, 255, 256]).cuda()

        hist_calibrator = calib.HistogramCalibrator(8, None, False, grow_method='stretch')
        hist_calibrator.collect(x_1)
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax(method='entropy')

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be closer to 256 because the last bin gets stretched to (~255, 257)
        assert (amax - 255.).abs() < (amax - 256.).abs()

        hist_calibrator = calib.HistogramCalibrator(8, None, False, grow_method='append')
        hist_calibrator.collect(x_1)
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax(method='mse')

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be closer to 255
        assert (amax - 255.).abs() < 0.5

    def test_skip_zeros(self, verbose):
        x_1 = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        x_2 = torch.tensor([0, 0, 0, 0, 0, 6, 7, 8, 9, 10])

        calibrator = calib.HistogramCalibrator(8, None, False, skip_zeros=True)
        calibrator.collect(x_1)
        calibrator.collect(x_2)

        amax = calibrator.compute_amax("percentile", percentile=50)

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be close to 5
        assert (amax - 5.).abs() < 10/2048

    def test_torch_hist(self):
        x_1 = torch.rand(1023, device="cuda")
        x_1[0] = 0
        x_2 = torch.rand(1023, device="cuda") + 1  # Make sure histogram bins need to be grown
        x_2[1] = 0

        calibrator_np = calib.HistogramCalibrator(8, None, False, num_bins=19, torch_hist=False)
        calibrator_torch = calib.HistogramCalibrator(8, None, False, num_bins=19, torch_hist=True)

        calibrator_np.collect(x_1)
        calibrator_torch.collect(x_1)
        assert calibrator_torch._calib_hist.numel() == calibrator_torch._calib_bin_edges.numel() - 1
        np.testing.assert_array_equal(calibrator_np._calib_hist, calibrator_torch._calib_hist.cpu().numpy())
        np.testing.assert_array_almost_equal(
            calibrator_np._calib_bin_edges, calibrator_torch._calib_bin_edges.cpu().numpy())

        # Test multiple collections with some of them needs to expand range
        for _ in range(3):
            calibrator_np.collect(x_2)
            calibrator_torch.collect(x_2)
            calibrator_np.collect(x_1)
            calibrator_torch.collect(x_1)

            # Test compute_amax function doesn't convert _calib_hist and _calib_bin_edges unnecessarily
            calibrator_np.compute_amax("percentile", percentile=99.99)
            calibrator_torch.compute_amax("percentile", percentile=99.99)

            np.testing.assert_array_equal(calibrator_np._calib_hist, calibrator_torch._calib_hist.cpu().numpy())
            np.testing.assert_array_almost_equal(
                calibrator_np._calib_bin_edges, calibrator_torch._calib_bin_edges.cpu().numpy())
            assert calibrator_torch._calib_hist.numel() == calibrator_torch._calib_bin_edges.numel() - 1


class TestEntropyCalibrator():

    def test_one_tensor(self, verbose):
        hist_calibrator = calib.HistogramCalibrator(8, None, False, grow_method='stretch')

        x_2 = torch.rand(11, 7, 3, 3).cuda() # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10. # create outlier
        hist_calibrator.collect(x_2)

        # Don't have a better test metric. One outlier 10 should be discared by KL-divergence
        amax = hist_calibrator.compute_amax("entropy")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        assert amax < 1.1

    def test_unsigned(self, verbose):
        hist_calibrator = calib.HistogramCalibrator(8, None, True, grow_method='stretch')

        x_2 = torch.rand(11, 7, 3, 3).cuda() # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10. # create outlier
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax("entropy")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        assert amax < 1.1

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        hist_calibrator = calib.HistogramCalibrator(8, None, False, torch_hist=torch_hist)

        x_2 = torch.rand(11, 7, 3, 3).cuda() # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10. # create outlier


        x_2 = torch.rand(11, 7, 3, 3).cuda() # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10. # create outlier
        hist_calibrator.collect(x_2)
        x_3 = torch.rand(11, 7, 3, 3).cuda()
        hist_calibrator.collect(x_3)

        # Don't have a better test metric. One outlier 10 should be discared by KL-divergence
        amax = hist_calibrator.compute_amax("entropy")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        assert amax < 1.1

    def test_repr(self):
        hist_calibrator = calib.HistogramCalibrator(8, None, True)
        repr(hist_calibrator)

class TestMSECalibrator():

    def test_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False)

        x_1 = torch.ones(11, 7, 3, 3).cuda() * 255.
        x_1[1, 1, 1, 1] = 256. # create an outlier
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("mse")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be closer to 255
        assert (amax - 255.).abs() < (amax - 256.).abs()

    def test_unsigned_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, True)

        x_1 = torch.ones(11, 7, 3, 3).cuda() * 512.
        x_1[1, 1, 1, 1] = 513. # create an outlier
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("mse")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be closer to 512
        assert (amax - 512.).abs() < (amax - 513.).abs()

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False, torch_hist=torch_hist)

        x_1 = torch.ones(11, 7, 3, 3).cuda() * 255.
        x_1[1, 1, 1, 1] = 256. # create an outlier
        calibrator.collect(x_1)
        x_2 = torch.ones(11, 7, 3, 3).cuda() * 255.
        calibrator.collect(x_2)

        amax = calibrator.compute_amax("mse")

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be closer to 255
        assert (amax - 255.).abs() < (amax - 256.).abs()

    def test_repr(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        repr(calibrator)

class TestPercentileCalibrator():

    def test_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("percentile", percentile=90)

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be approximately 89
        assert (amax - 89.).abs() < 100/1024

    def test_unsigned_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator( 8, None, True)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("percentile", percentile=80)

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be approximately 79
        assert (amax - 79.).abs() < 100/2048

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False, torch_hist=torch_hist)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)
        x_2 = torch.arange(0, 50, 0.5)
        calibrator.collect(x_2)
        amax = calibrator.compute_amax("percentile", percentile=99)

        if verbose:
            print('amax={:.4f}'.format(amax.item()), end=' ')

        # amax should be approximately 97
        assert (amax - 97.).abs() < 100/1024

    def test_repr(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        repr(calibrator)

    def test_range(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        x_1 = torch.arange(100)
        calibrator.collect(x_1)
        with pytest.raises(ValueError, match="range"):
            calibrator.compute_amax("percentile", percentile=-10)
        with pytest.raises(ValueError, match="range"):
            calibrator.compute_amax("percentile", percentile=200)

class TestCalibrateWeights():

    def test_max(self):
        torch.manual_seed(12345)
        ref_lenet = QuantLeNet()
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()

        for module in ref_lenet.modules():
            if isinstance(module, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
                module.weight_quantizer.enable_calib()
                module.weight_quantizer.disable_quant()
                module.weight_quantizer(module.weight)
                module.weight_quantizer.load_calib_amax()

        calib.calibrate_weights(test_lenet, method="max")

        for ref_module, test_module in zip(ref_lenet.modules(), test_lenet.modules()):
            if isinstance(ref_module, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
                test_utils.compare(
                    ref_module.weight_quantizer.amax, test_module.weight_quantizer.amax, rtol=0, atol=0, ctol=0)
                assert ref_module.weight_quantizer.amax.shape == test_module.weight_quantizer.amax.shape

    def test_shape_with_axis(self):
        """Check calibrate_weight function returns same shape as TensorQuantizer"""
        torch.manual_seed(12345)
        ref_lenet = QuantLeNet()
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()

        for module in ref_lenet.modules():
            if isinstance(module, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
                module.weight_quantizer.enable_calib()
                module.weight_quantizer.disable_quant()
                module.weight_quantizer(module.weight)
                module.weight_quantizer.load_calib_amax()

        calib.calibrate_weights(test_lenet, method="percentile")

        for ref_module, test_module in zip(ref_lenet.modules(), test_lenet.modules()):
            if isinstance(ref_module, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
                assert ref_module.weight_quantizer.amax.shape == test_module.weight_quantizer.amax.shape

    def test_percentile(self):
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()
        test_percentile = 99.99

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method="percentile", perchannel=False, percentile=test_percentile)
        ref_calibrator.collect(test_lenet.conv1.weight)
        ref_amax = ref_calibrator.compute_amax("percentile", percentile=test_percentile)
        test_utils.compare(ref_amax, test_lenet.conv1.weight_quantizer.amax, rtol=0, atol=0, ctol=0)

    def test_percentile_with_axis(self):
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()
        test_percentile = 99.99

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method="percentile", perchannel=True, percentile=test_percentile)
        ref_calibrator.collect(test_lenet.conv2.weight[1])
        ref_amax = ref_calibrator.compute_amax("percentile", percentile=test_percentile)
        test_utils.compare(ref_amax, test_lenet.conv2.weight_quantizer.amax[1], rtol=0, atol=0, ctol=0)

    def test_mse(self):
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method="mse", perchannel=False)
        ref_calibrator.collect(test_lenet.conv1.weight)
        ref_amax = ref_calibrator.compute_amax("mse")
        test_utils.compare(ref_amax, test_lenet.conv1.weight_quantizer.amax, rtol=0, atol=0, ctol=0)

    def test_mse_with_axis(self):
        torch.manual_seed(12345)
        test_lenet = QuantLeNet()

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method="mse", perchannel=True)
        ref_calibrator.collect(test_lenet.conv2.weight[1])
        ref_amax = ref_calibrator.compute_amax("mse")
        test_utils.compare(ref_amax, test_lenet.conv2.weight_quantizer.amax[1], rtol=0, atol=0, ctol=0)
