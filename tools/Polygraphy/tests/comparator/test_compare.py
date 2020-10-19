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
import numpy as np
from polygraphy.comparator import CompareFunc, IterationResult


class TestCompareFunc(object):
    def test_basic_compare_func_can_compare_bool(self):
        iter_result0 = IterationResult(outputs={"output": np.zeros((4, 4), dtype=np.bool)})
        iter_result1 = IterationResult(outputs={"output": np.ones((4, 4), dtype=np.bool)})

        compare_func = CompareFunc.basic_compare_func()
        acc = compare_func(iter_result0, iter_result1)

        assert not acc["output"]
