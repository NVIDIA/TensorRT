#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from tensorflow_quantization.quantize_wrapper_base import BaseQuantizeWrapper
import copy
import pytest


EXPECTED_WRAPPERS = [
    "WeightedBaseQuantizeWrapper",
    "Conv2DQuantizeWrapper",
    "DenseQuantizeWrapper",
    "DepthwiseConv2DQuantizeWrapper",
    "NonWeightedBaseQuantizeWrapper",
    "AveragePooling2DQuantizeWrapper",
    "GlobalAveragePooling2DQuantizeWrapper",
    "MaxPooling2DQuantizeWrapper",
    "BatchNormalizationQuantizeWrapper",
    "NonWeightedBaseQuantizeWrapperForMultipleInputs",
    "MultiplyQuantizeWrapper",
    "ConcatenateQuantizeWrapper",
    "AddQuantizeWrapper",
]


def test_old_wrappers_registration():
    all_wrappers = BaseQuantizeWrapper.CHILD_WRAPPERS
    assert EXPECTED_WRAPPERS == list(all_wrappers.keys())


def test_new_wrapper_registration():
    class TestWrapper(BaseQuantizeWrapper):
        def __init__(self, layer, **kwargs):
            super().__init__(layer, **kwargs)

    all_wrappers = BaseQuantizeWrapper.CHILD_WRAPPERS
    expected = copy.deepcopy(EXPECTED_WRAPPERS)
    expected.append("TestWrapper")

    assert expected == list(all_wrappers.keys())
