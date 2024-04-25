#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import pytest
from polygraphy import util
from polygraphy.comparator import PostprocessFunc, IterationResult

build_torch = lambda a, **kwargs: util.array.to_torch(np.array(a, **kwargs))


@pytest.mark.parametrize("array_type", [np.array, build_torch])
class TestTopK:
    def test_basic(self, array_type):
        arr = array_type([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k=3)
        top_k = func(IterationResult({"x": arr}))
        assert util.array.equal(top_k["x"], array_type([4, 3, 2]))

    def test_k_can_exceed_array_len(self, array_type):
        arr = array_type([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k=10)
        top_k = func(IterationResult({"x": arr}))
        assert util.array.equal(top_k["x"], array_type([4, 3, 2, 1, 0]))

    def test_per_output_top_k(self, array_type):
        arr = array_type([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k={"": 10, "y": 2})
        top_k = func(IterationResult({"x": arr, "y": arr}))
        assert util.array.equal(top_k["x"], array_type([4, 3, 2, 1, 0]))
        assert util.array.equal(top_k["y"], array_type([4, 3]))

    def test_per_output_top_k_axis(self, array_type):
        arr = array_type([[5, 6, 5], [6, 5, 6]], dtype=np.float32)
        func = PostprocessFunc.top_k(k={"": (1, 0), "y": (1, 1)})
        top_k = func(IterationResult({"x": arr, "y": arr}))
        assert util.array.equal(top_k["x"], array_type([[1, 0, 1]]))
        assert util.array.equal(top_k["y"], array_type([[1], [0]]))

    def test_top_k_half(self, array_type):
        arr = array_type([1, 2, 3, 4, 5], dtype=np.float16)
        func = PostprocessFunc.top_k(k=3)
        top_k = func(IterationResult({"x": arr}))
        assert util.array.equal(top_k["x"], array_type([4, 3, 2]))
