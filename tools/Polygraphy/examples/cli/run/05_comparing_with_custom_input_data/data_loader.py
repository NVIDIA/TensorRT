#!/usr/bin/env python3
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

"""
Demonstrates two methods of loading custom input data in Polygraphy:

Option 1: Defines a `load_data` function that returns a generator yielding
    feed_dicts so that this script can be used as the argument for
    the --data-loader-script command-line parameter.

Option 2: Writes input data to a JSON file that can be used as the argument for
    the --load-inputs command-line parameter.
"""
import numpy as np
from polygraphy.json import save_json

INPUT_SHAPE = (1, 2, 28, 28)

# Option 1: Define a function that will yield feed_dicts (i.e. Dict[str, np.ndarray])
def load_data():
    for _ in range(5):
        yield {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}  # Still totally real data


# Option 2: Create a JSON file containing the input data using the `save_json()` helper.
#   The input to `save_json()` should have type: List[Dict[str, np.ndarray]].
#   For convenience, we'll reuse our `load_data()` implementation to generate the list.
input_data = list(load_data())
save_json(input_data, "custom_inputs.json", description="custom input data")
