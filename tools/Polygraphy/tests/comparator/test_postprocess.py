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
import numpy as np
from polygraphy.comparator import PostprocessFunc, IterationResult


class TestTopK:
    def test_basic(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k=3)
        top_k = func(IterationResult({"x": arr}))
        assert np.all(top_k["x"] == [4, 3, 2])

    def test_k_can_exceed_array_len(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k=10)
        top_k = func(IterationResult({"x": arr}))
        assert np.all(top_k["x"] == [4, 3, 2, 1, 0])

    def test_per_output_top_k(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.top_k(k={"": 10, "y": 2})
        top_k = func(IterationResult({"x": arr, "y": arr}))
        assert np.all(top_k["x"] == [4, 3, 2, 1, 0])
        assert np.all(top_k["y"] == [4, 3])

    def test_per_output_top_k_axis(self):
        arr = np.array([[5, 6, 5], [6, 5, 6]], dtype=np.float32)
        func = PostprocessFunc.top_k(k={"": (1, 0), "y": (1, 1)})
        top_k = func(IterationResult({"x": arr, "y": arr}))
        assert np.all(top_k["x"] == [[1, 0, 1]])
        assert np.all(top_k["y"] == [[1], [0]])
