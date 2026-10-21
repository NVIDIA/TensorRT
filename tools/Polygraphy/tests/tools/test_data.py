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
from collections import OrderedDict

import numpy as np
import pytest
from polygraphy import util
from polygraphy.comparator import RunResults
from polygraphy.json import load_json, save_json
from tests.models.meta import ONNX_MODELS


class TestMerge:
    def test_merge_inputs_outputs(self, poly_run, poly_data, subtool="merge"):
        with util.NamedTemporaryFile(suffix=".json") as inps, util.NamedTemporaryFile(
            suffix=".json"
        ) as outs, util.NamedTemporaryFile() as merged:
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--save-inputs",
                    inps.name,
                    "--save-outputs",
                    outs.name,
                ],
            )

            poly_data([subtool, inps.name, outs.name, "-o", merged.name])

            merged_data = load_json(merged.name)
            assert len(merged_data) == 1
            assert list(merged_data[0].keys()) == ["x", "y"]
            assert all(isinstance(val, np.ndarray) for val in merged_data[0].values())


class TestConcat:
    def test_concat_inputs(self, poly_data):
        inputs_a = [
            OrderedDict(
                [
                    ("x", np.array([1], dtype=np.float32)),
                    ("y", np.array([2], dtype=np.float32)),
                ]
            ),
            OrderedDict(
                [
                    ("x", np.array([3], dtype=np.float32)),
                    ("y", np.array([4], dtype=np.float32)),
                ]
            ),
        ]
        inputs_b = [
            OrderedDict(
                [
                    ("x", np.array([5], dtype=np.float32)),
                    ("y", np.array([6], dtype=np.float32)),
                ]
            ),
            OrderedDict(
                [
                    ("x", np.array([7], dtype=np.float32)),
                    ("y", np.array([8], dtype=np.float32)),
                ]
            ),
            OrderedDict(
                [
                    ("x", np.array([9], dtype=np.float32)),
                    ("y", np.array([10], dtype=np.float32)),
                ]
            ),
        ]

        with util.NamedTemporaryFile() as first, util.NamedTemporaryFile() as second, util.NamedTemporaryFile() as merged:
            save_json(inputs_a, first.name, "input file containing 2 iteration(s)")
            save_json(inputs_b, second.name, "input file containing 3 iteration(s)")

            poly_data(["concat", first.name, second.name, "-o", merged.name])

            merged_data = load_json(merged.name)
            assert len(merged_data) == 5
            assert list(merged_data[0].keys()) == ["x", "y"]

    def test_concat_outputs(self, poly_data):
        results_a = RunResults()
        results_a.add(
            [
                OrderedDict([("out", np.array([1], dtype=np.float32))]),
                OrderedDict([("out", np.array([2], dtype=np.float32))]),
            ],
            runner_name="runner",
        )
        results_b = RunResults()
        results_b.add(
            [
                OrderedDict([("out", np.array([3], dtype=np.float32))]),
                OrderedDict([("out", np.array([4], dtype=np.float32))]),
                OrderedDict([("out", np.array([5], dtype=np.float32))]),
            ],
            runner_name="runner",
        )

        with util.NamedTemporaryFile() as first, util.NamedTemporaryFile() as second, util.NamedTemporaryFile() as merged:
            save_json(results_a, first.name, "output file containing 2 iteration(s)")
            save_json(results_b, second.name, "output file containing 3 iteration(s)")

            poly_data(["concat", first.name, second.name, "-o", merged.name])

            merged_data = load_json(merged.name)
            assert isinstance(merged_data, RunResults)
            assert len(merged_data["runner"]) == 5
