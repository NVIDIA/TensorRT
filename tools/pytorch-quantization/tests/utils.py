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


"""Utils for testing quantization."""
import numpy as np
from scipy.spatial import distance

import torch

from pytorch_quantization import tensor_quant

def quantize_by_range(x, num_bits):
    """Quantize torch tensor by range to num_bits with symmetric zero-mean quantizer."""
    amax = x.abs().max()
    x_q = tensor_quant.fake_tensor_quant(x, amax, num_bits)
    return x_q

def quantize_by_range_fused(x_tuple, num_bits):
    """Quantize multiple torch tensors by combined range to num_bits with symmetric zero-mean quantizer."""
    # compute aggregate amax across all tensors
    amax = max([x.abs().max() for x in x_tuple])
    # quantize each tensor with the aggregate amax
    x_q_tuple = tuple(tensor_quant.fake_tensor_quant(x, amax, num_bits) for x in x_tuple)
    return x_q_tuple

def copy_state_and_quantize(dst, src, num_bits):
    """Copy src to dst, quantize all 'weight' entries to num_bits."""
    src_state_dict = src.state_dict()
    dst_state_dict = dict()
    for key in src_state_dict:
        if 'weight' in key:
            dst_state_dict[key] = quantize_by_range(src_state_dict[key], num_bits)
        else:
            dst_state_dict[key] = src_state_dict[key].clone()

    dst.load_state_dict(dst_state_dict)

def copy_state_and_quantize_fused(dst, src, num_bits):
    """Copy src to dst, quantize all 'weight' entries to num_bits using the aggregate amax."""
    src_state_dict = src.state_dict()
    dst_state_dict = dict()

    # compute aggregate amax across all weight tensors
    amax = 0
    for key in src_state_dict:
        if 'weight' in key:
            amax = max(amax, src_state_dict[key].abs().max())

    # quantize each weight tensor with the aggregate amax
    for key in src_state_dict:
        if 'weight' in key:
            dst_state_dict[key] = tensor_quant.fake_tensor_quant(src_state_dict[key], amax, num_bits)
        else:
            dst_state_dict[key] = src_state_dict[key].clone()

    dst.load_state_dict(dst_state_dict)

def compare(a, b, rtol=1e-7, atol=1e-6, ctol=1e-6):
    """Compare two tensors and raise AssertionError if their difference is outside of tolerance."""
    if torch.isinf(a).any():
        raise ValueError("a contains infs")
    if torch.isinf(b).any():
        raise ValueError("b contains infs")

    a = a.detach().cpu().numpy().flatten()
    b = b.detach().cpu().numpy().flatten()

    # compare elements of a and b relative to the max value in b
    # large fp32 values may cause quantization errors that propagate to small values
    rel_diff = np.abs(a-b)/np.linalg.norm(b)
    abs_diff = np.abs(a-b)
    cos_diff = distance.cosine(a, b)
    try:
        if rel_diff.max() > rtol:
            raise AssertionError("Tensor relative error > %.2e (%.2e)" % (rtol, rel_diff.max()))
        if abs_diff.max() > atol:
            raise AssertionError("Tensor absolute error > %.2e (%.2e)" % (atol, abs_diff.max()))
        if cos_diff > ctol:
            raise AssertionError("Tensor cosine distance > %.2e (%.2e)" % (ctol, cos_diff))
        # np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        # np.testing.assert_array_almost_equal_nulp(a, b)
    except AssertionError as e:
        print('norm(a) =', np.linalg.norm(a))
        print('norm(b) =', np.linalg.norm(b))
        print('Largest relative difference = %.2e' % rel_diff.max())
        idx = np.argmax(rel_diff)
        print('a[%d] = %.10f' % (idx, a[idx]))
        print('b[%d] = %.10f' % (idx, b[idx]))
        print('Largest absolute difference = %.2e' % abs_diff.max())
        idx = np.argmax(abs_diff)
        print('a[%d] = %.10f' % (idx, a[idx]))
        print('b[%d] = %.10f' % (idx, b[idx]))
        print('Cosine distance = %.2e' % cos_diff)
        raise e

def assert_min_mse(a, b, tol=1e-20):
    """Assert that the mean squared error between a and b is at least tol."""
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    mse = ((a-b)**2).mean()
    if mse < tol:
        raise AssertionError("MSE = %.2e < %.2e" % (mse, tol))

def quant_np(x, amax, num_bits=8, fake=False, narrow_range=True):
    """Quantize x using numpy."""
    intmax = 2.0**(num_bits - 1) - 1
    intmin = -intmax if narrow_range else -intmax - 1
    scale = intmax / amax
    x_q = np.round(np.clip(x * scale, intmin, intmax))

    if fake:
        x_q /= scale

    return x_q
