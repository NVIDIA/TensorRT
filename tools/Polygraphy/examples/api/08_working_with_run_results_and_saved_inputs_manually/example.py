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
This script demonstrates how to use the `load_json` and `RunResults` APIs to load
and manipulate inference inputs and outputs respectively.
"""

from polygraphy.comparator import RunResults
from polygraphy.json import load_json


def main():
    # Use the `load_json` API to load inputs from file.
    #
    # NOTE: The `save_json` and `load_json` standalone helpers should be used only with non-Polygraphy objects.
    # Polygraphy objects that support serialization include `save` and `load` methods.
    inputs = load_json("inputs.json")

    # Inputs are stored as a `List[Dict[str, np.ndarray]]`, i.e. a list of feed_dicts,
    # where each feed_dict maps input names to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we'll only look at the first item.
    # If you need to access inputs from multiple iterations, you can do something like this instead:
    #
    #    for feed_dict in inputs:
    #        for name, array in feed_dict.items():
    #            ... # Do something with the inputs here
    #
    [feed_dict] = inputs
    for name, array in feed_dict.items():
        print(f"Input: '{name}' | Values:\n{array}")

    # Use the `RunResults.load` API to load results from file.
    #
    # TIP: You can provide either a file path or a file-like object here.
    results = RunResults.load("outputs.json")

    # The `RunResults` object is structured like a `Dict[str, List[IterationResult]]``,
    # mapping runner names to inference outputs from one or more iterations.
    # An `IterationResult` behaves just like a `Dict[str, np.ndarray]` mapping output names
    # to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we can unpack it
    # directly in the loop. If you need to access outputs from multiple iterations,
    # you can do something like this instead:
    #
    #    for runner_name, iters in results.items():
    #        for outputs in iters:
    #             ... # Do something with the outputs here
    #
    for runner_name, [outputs] in results.items():
        print(f"\nProcessing outputs for runner: {runner_name}")
        # Now you can read or modify the outputs for each runner.
        # For the sake of this example, we'll just print them:
        for name, array in outputs.items():
            print(f"Output: '{name}' | Values:\n{array}")


if __name__ == "__main__":
    main()
