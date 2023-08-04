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


"""Test pytorch_quantization.utils"""
import pytest
import numpy as np

import torch
from pytorch_quantization import utils as quant_utils
from tests.fixtures import verbose

np.random.seed(12345)

# pylint:disable=missing-docstring, no-self-use

class TestQuantUtils():

    def test_reduce_amax(self):
        x_np = (np.random.rand(3, 7, 11, 13, 17) - 0.1).astype(np.float32)
        x_torch = torch.tensor(x_np)

        # Test reduce to one value
        amax_np = np.max(np.abs(x_np))
        amax_torch = quant_utils.reduce_amax(x_torch)
        np.testing.assert_array_equal(amax_np, amax_torch.cpu().numpy())

        # Test different axis
        axes = [(1, 2, 3), (0, 2, 3), (0, 3), (0, 1, 3, 4)]
        for axis in axes:
            keepdims = np.random.rand() > 0.5
            amax_np = np.max(np.abs(x_np), axis=axis, keepdims=keepdims)
            amax_torch = quant_utils.reduce_amax(x_torch, axis=axis, keepdims=keepdims)
            np.testing.assert_array_almost_equal(amax_np, amax_torch.cpu().numpy())

        with pytest.raises(ValueError) as excinfo:
            quant_utils.reduce_amax(x_torch, axis=(0, 1, 2, 3, 4, 5))
            assert "Cannot reduce more axes" in str(excinfo.value)
