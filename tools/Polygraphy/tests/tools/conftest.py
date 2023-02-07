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

import os
import tempfile
from collections import namedtuple
from textwrap import dedent
from typing import List

import pytest
import tensorrt as trt
from polygraphy.backend.trt import Algorithm, TacticReplayData
from polygraphy.json import save_json


def make_poly_fixture(subtool: List[str]):
    @pytest.fixture()
    def poly_fixture(script_runner):
        def poly_fixture_impl(additional_opts: List[str] = [], expect_error: bool = False, *args, **kwargs):
            cmd = ["polygraphy"] + subtool + ["-v"] + additional_opts
            # NOTE: script_runner does not work very well in `in-process`` mode if you need to inspect stdout/stderr.
            # Occasionally, the output comes out empty - not clear why. Cave emptor!
            # Decorate your tests with `@pytest.mark.script_launch_mode("subprocess")` to use `subprocess` to avoid this issue.
            status = script_runner.run(*cmd, *args, **kwargs)
            assert status.success == (not expect_error)
            return status

        return poly_fixture_impl

    return poly_fixture


poly = make_poly_fixture([])
poly_run = make_poly_fixture(["run"])
poly_convert = make_poly_fixture(["convert"])
poly_inspect = make_poly_fixture(["inspect"])
poly_surgeon = make_poly_fixture(["surgeon"])
poly_surgeon_extract = make_poly_fixture(["surgeon", "extract"])
poly_template = make_poly_fixture(["template"])
poly_debug = make_poly_fixture(["debug"])
poly_data = make_poly_fixture(["data"])


FakeAlgorithmContext = namedtuple("FakeAlgorithmContext", ["name", "num_inputs", "num_outputs"])
FakeAlgorithm = namedtuple("FakeAlgorithm", ["algorithm_variant", "io_info"])
FakeAlgorithm.get_algorithm_io_info = lambda this, index: this.io_info[index]

FakeAlgorithmVariant = namedtuple("FakeAlgorithmVariant", ["implementation", "tactic"])
FakeAlgorithmIOInfo = namedtuple("FakeAlgorithmIOInfo", ["tensor_format", "dtype", "strides"])


@pytest.fixture(scope="session", params=["", "subdir"])
def replay_dir(request):
    def fake_context(name, num_inputs=1, num_outputs=1):
        return FakeAlgorithmContext(name=name, num_inputs=num_inputs, num_outputs=num_outputs)

    def fake_algo(
        implementation=6, tactic=0, num_io=2, tensor_format=trt.TensorFormat.LINEAR, dtype=trt.float32, strides=(1, 2)
    ):
        io_info = [FakeAlgorithmIOInfo(tensor_format=tensor_format, dtype=dtype, strides=strides)] * num_io
        return FakeAlgorithm(algorithm_variant=FakeAlgorithmVariant(implementation, tactic), io_info=io_info)

    def make_replay(tactic):
        return TacticReplayData().add("layer0", Algorithm.from_trt(fake_context("layer0"), fake_algo(0, tactic)))

    with tempfile.TemporaryDirectory() as dir:

        def make_path(prefix, *args):
            path = os.path.join(dir, prefix)
            if request.param:
                path = os.path.join(path, request.param)
            path = os.path.join(path, *args)
            return path

        # Good tactics
        save_json(make_replay(0), make_path("good", "0.json"))
        save_json(make_replay(1), make_path("good", "1.json"))

        # Bad tactics
        save_json(make_replay(1), make_path("bad", "0.json"))
        save_json(make_replay(2), make_path("bad", "1.json"))

        EXPECTED_OUTPUT = dedent(
            """
            [I] Loaded {num} good tactic replays.
            [I] Loaded {num} bad tactic replays.
            [I] Found potentially bad tactics:
            [I] Layer: layer0
                    Algorithms: ["(Implementation: 0, Tactic: 2) | Inputs: (('TensorFormat.LINEAR', 'DataType.FLOAT', '(1, 2)'),) | Outputs: (('TensorFormat.LINEAR', 'DataType.FLOAT', '(1, 2)'),)"]
            """
        )
        yield dir, EXPECTED_OUTPUT
