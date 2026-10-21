#!/usr/bin/env python3
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

"""
Defines a `load_data` generator that yields one feed_dict at a time.

Because the data is produced lazily (with `yield`), the entire dataset is never held in memory at
once. Combined with `polygraphy run` (which streams by default), this lets you run accuracy
comparison over a dataset that is far too large to fit in memory.
"""
import numpy as np

INPUT_SHAPE = (1, 1, 2, 2)
NUM_ITERATIONS = 50


def load_data():
    # In a real workload, this might read images/tokens from disk one at a time.
    for index in range(NUM_ITERATIONS):
        yield {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32) * index}
