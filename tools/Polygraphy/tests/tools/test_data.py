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
from polygraphy import util
from polygraphy.json import load_json
from tests.models.meta import ONNX_MODELS


class TestToInput:
    def test_merge_inputs_outputs(self, poly_run, poly_data):
        with util.NamedTemporaryFile() as inps, util.NamedTemporaryFile() as outs, util.NamedTemporaryFile() as merged:
            poly_run(
                [ONNX_MODELS["identity"].path, "--onnxrt", "--save-inputs", inps.name, "--save-outputs", outs.name],
            )

            poly_data(["to-input", inps.name, outs.name, "-o", merged.name])

            merged_data = load_json(merged.name)
            assert len(merged_data) == 1
            assert list(merged_data[0].keys()) == ["x", "y"]
            assert all(isinstance(val, np.ndarray) for val in merged_data[0].values())
