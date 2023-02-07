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
Generates input and output data for an identity model and saves it to disk.
"""
import numpy as np
from polygraphy.comparator import RunResults
from polygraphy.json import save_json

INPUT_SHAPE = (1, 1, 2, 2)


# We'll generate arbitrary input data and then "compute" the expected output data before saving both to disk.
# In order for Polygraphy to load the input and output data, they must be in the following format:
#   - Input Data: List[Dict[str, np.ndarray]] (A list of feed_dicts)
#   - Output Data: RunResults


# Generate arbitrary input data compatible with the model.
#
# TIP: We could have alternatively used a generator as in `run` example 05 (05_comparing_with_custom_input_data).
#   In that case, we would simply provide this script to `--data-loader-script` instead of saving the inputs here
#   and then using `--load-inputs`.
input_data = {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}

# NOTE: Input data must be in a list (to support multiple sets of inputs), so we create one before saving it.
#   The `description` argument is optional:
save_json([input_data], "custom_inputs.json", description="custom input data")


# "Compute" the outputs based on the input data. Since this is an identity model, we can just copy the inputs.
output_data = {"y": input_data["x"]}

# To save output data, we can create a RunResults object:
custom_outputs = RunResults()

# The `add()` helper function allows us to easily add entries.
#
# NOTE: As with input data, output data must be in a list, so we create one before saving it.
#
# TIP: Alternatively, we can manually add entries using an approach like:
#   runner_name = "custom_runner"
#   custom_outputs[runner_name] = [IterationResult(output_data, runner_name=runner_name), ...]
#
# TIP: To store outputs from multiple different implementations, you can specify different `runner_name`s to `add()`.
#   If `runner_name` is omitted, a default is used.
custom_outputs.add([output_data], runner_name="custom_runner")
custom_outputs.save("custom_outputs.json")
