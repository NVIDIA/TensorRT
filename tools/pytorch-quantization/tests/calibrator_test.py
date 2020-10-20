#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import tests.utils as test_utils
from tests.fixtures import verbose

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

    def test_two_tensor(self, verbose):
        hist_calibrator = calib.HistogramCalibrator(8, None, False, grow_method='stretch')

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

    def test_two_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False)

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

    def test_two_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False)

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
