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


"""Histogram based calibrators"""
from collections import Counter
import numpy as np
from scipy.stats import entropy

from absl import logging

import torch

from pytorch_quantization.calib.calibrator import _Calibrator
from pytorch_quantization.tensor_quant import fake_tensor_quant

class HistogramCalibrator(_Calibrator):
    """Unified histogram calibrator

    Histogram will be only collected once. compute_amax() performs entropy, percentile, or mse
        calibration based on arguments

    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.
        num_bins: An integer. Number of histograms bins. Default 2048.
        grow_method: A string. DEPRECATED. default None.
        skip_zeros: A boolean. If True, skips zeros when collecting data for histogram. Default False.
    """
    def __init__(self, num_bits, axis, unsigned, num_bins=2048, grow_method=None, skip_zeros=False):
        super(HistogramCalibrator, self).__init__(num_bits, axis, unsigned)
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros

        self._calib_bin_edges = None
        self._calib_hist = None

        if axis is not None:
            raise NotImplementedError("Calibrator histogram collection only supports per tensor scaling")

        if grow_method is not None:
            logging.warning("grow_method is deprecated. Got %s, ingored!", grow_method)

    def collect(self, x):
        """Collect histogram"""
        if torch.min(x) < 0.:
            logging.log_first_n(
                logging.INFO,
                ("Calibrator encountered negative values. It shouldn't happen after ReLU. "
                 "Make sure this is the right tensor to calibrate."),
                1)
            x = x.abs()
        x_np = x.cpu().detach().numpy()

        if self._skip_zeros:
            x_np = x_np[np.where(x_np != 0)]

        if self._calib_bin_edges is None and self._calib_hist is None:
            # first time it uses num_bins to compute histogram.
            self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)
        else:
            temp_amax = np.max(x_np)
            if temp_amax > self._calib_bin_edges[-1]:
                # increase the number of bins
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                # NOTE: np.arange may create an extra bin after the one containing temp_amax
                new_bin_edges = np.arange(self._calib_bin_edges[-1] + width, temp_amax + width, width)
                self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_bin_edges))
            hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
            hist[:len(self._calib_hist)] += self._calib_hist
            self._calib_hist = hist

    def reset(self):
        """Reset the collected histogram"""
        self._calib_bin_edges = None
        self._calib_hist = None

    def _compute_amax_entropy(self, stride, start_bin):
        """Returns amax that minimizes KL-Divergence of the collected histogram"""

        # If calibrator hasn't collected any data, return none
        if self._calib_bin_edges is None and self._calib_hist is None:
            return None

        def _normalize_distr(distr):
            summ = np.sum(distr)
            if summ != 0:
                distr = distr / summ

        bins = self._calib_hist[:]
        bins[0] = bins[1]

        total_data = np.sum(bins)

        divergences = []
        arguments = []

        # we are quantizing to 128 values + sign if num_bits=8
        nbins = 1 << (self._num_bits - 1 + int(self._unsigned))

        starting = start_bin
        stop = len(bins)

        new_density_counts = np.zeros(nbins, dtype=np.float64)

        for i in range(starting, stop + 1, stride):
            new_density_counts.fill(0)
            space = np.linspace(0, i, num=nbins + 1)
            digitized_space = np.digitize(range(i), space) - 1

            digitized_space[bins[:i] == 0] = -1

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bins[idx]

            counter = Counter(digitized_space)
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            new_density = np.zeros(i, dtype=np.float64)
            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = np.sum(new_density) + np.sum(bins[i:])
            _normalize_distr(new_density)

            reference_density = np.array(bins[:len(digitized_space)])
            reference_density[-1] += np.sum(bins[i:])

            total_counts_old = np.sum(reference_density)
            if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
                raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                    total_counts_new, total_counts_old, total_data))

            _normalize_distr(reference_density)

            ent = entropy(reference_density, new_density)
            divergences.append(ent)
            arguments.append(i)

        divergences = np.array(divergences)
        logging.debug("divergences={}".format(divergences))
        last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
        calib_amax = self._calib_bin_edges[last_argmin * stride + starting]
        calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

        return calib_amax

    def _compute_amax_mse(self, stride, start_bin):
        """Returns amax that minimizes MSE of the collected histogram"""

        # If calibrator hasn't collected any data, return none
        if self._calib_bin_edges is None and self._calib_hist is None:
            return None

        counts = torch.from_numpy(self._calib_hist[:]).float()
        edges = torch.from_numpy(self._calib_bin_edges[:]).float()
        centers = (edges[1:] + edges[:-1])/2

        mses = []
        arguments = []

        for i in range(start_bin, len(centers), stride):

            amax = centers[i]
            quant_centers = fake_tensor_quant(centers, amax, self._num_bits, self._unsigned)

            mse = ((quant_centers - centers)**2 * counts).mean()

            mses.append(mse)
            arguments.append(i)

        logging.debug("mses={}".format(mses))
        argmin = np.argmin(mses)
        calib_amax = centers[arguments[argmin]]

        return calib_amax

    def _compute_amax_percentile(self, percentile):
        """Returns amax that clips the percentile fraction of collected data"""

        if percentile < 0 or percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        # If calibrator hasn't collected any data, return none
        if self._calib_bin_edges is None and self._calib_hist is None:
            return None

        total = self._calib_hist.sum()
        cdf = np.cumsum(self._calib_hist / total)
        idx = np.searchsorted(cdf, percentile/100)
        calib_amax = self._calib_bin_edges[idx]
        calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

        return calib_amax

    def compute_amax(
            self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
        """Compute the amax from the collected histogram

        Args:
            method: A string. One of ['entropy', 'mse', 'percentile']

        Keyword Arguments:
            stride: An integer. Default 1
            start_bin: An integer. Default 128
            percentils: A float number between [0, 100]. Default 99.99.

        Returns:
            amax: a tensor
        """
        if method == 'entropy':
            calib_amax = self._compute_amax_entropy(stride, start_bin)
        elif method == 'mse':
            calib_amax = self._compute_amax_mse(stride, start_bin)
        elif method == 'percentile':
            calib_amax = self._compute_amax_percentile(percentile)
        else:
            raise TypeError("Unknown calibration method {}".format(method))

        return calib_amax

    # pylint:disable=missing-docstring
    def __str__(self):
        s = "HistogramCalibrator("
        if self._calib_bin_edges is None:
            bin_edge_str = "None"
        else:
            bin_edge_str = "[{:.3f}, ..., {:.3f}]({})".format(
                self._calib_bin_edges[0], self._calib_bin_edges[-1], len(self._calib_bin_edges))
        s += "calib_bin_edges={})".format(bin_edge_str)
        return s

    def __repr__(self):
        s = "HistogramCalibrator("
        s += super(HistogramCalibrator, self).__repr__()
        s += " calib_bin_edges={_calib_bin_edges}"
        s += " calib_hist={_calib_hist})"
        return s.format(**self.__dict__)
    # pylint:enable=missing-docstring
