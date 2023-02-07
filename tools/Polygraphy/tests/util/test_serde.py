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
import pytest
import tensorrt as trt
from polygraphy import constants, util
from polygraphy.backend.trt import Algorithm, TacticReplayData
from polygraphy.comparator import IterationResult, RunResults
from polygraphy.exception import PolygraphyException
from polygraphy.json import Decoder, Encoder, from_json, load_json, to_json


class Dummy:
    def __init__(self, x):
        self.x = x


@Encoder.register(Dummy)
def encode(dummy):
    return {"x": dummy.x}


@Decoder.register(Dummy)
def decode(dct):
    assert len(dct) == 1  # Custom type markers should be removed at this point
    return Dummy(x=dct["x"])


class TestEncoder:
    def test_registered(self):
        d = Dummy(x=-1)
        d_json = to_json(d)
        assert encode(d) == {"x": d.x, constants.TYPE_MARKER: "Dummy"}
        expected = f'{{\n    "x": {d.x},\n    "{constants.TYPE_MARKER}": "Dummy"\n}}'
        assert d_json == expected


class TestDecoder:
    def test_object_pairs_hook(self):
        d = Dummy(x=-1)
        d_json = to_json(d)

        new_d = from_json(d_json)
        assert new_d.x == d.x


def make_algo():
    return Algorithm(
        implementation=4,
        tactic=5,
        # Should work even if strides are not set
        inputs=[(trt.TensorFormat.LINEAR, trt.float32, (1, 2)), (trt.TensorFormat.LINEAR, trt.float32)],
        outputs=[(trt.TensorFormat.LINEAR, trt.float32, (2, 3))],
    )


def make_iter_result():
    return IterationResult(
        runtime=4.5,
        runner_name="test",
        outputs={
            "out0": np.random.random_sample((1, 2, 1)),
            "out1": np.ones((1, 2), dtype=np.float32),
        },
    )


JSONABLE_CASES = [
    RunResults([("runner0", [make_iter_result()]), ("runner0", [make_iter_result()])]),
    TacticReplayData().add("hi", algorithm=make_algo()),
]


class TestImplementations:
    @pytest.mark.parametrize(
        "obj",
        [
            Algorithm(
                implementation=4,
                tactic=5,
                inputs=[(trt.TensorFormat.LINEAR, trt.float32)],
                outputs=[(trt.TensorFormat.LINEAR, trt.float32)],
            ),
            Algorithm(
                implementation=4,
                tactic=5,
                inputs=[(trt.TensorFormat.LINEAR, trt.float32), (trt.TensorFormat.CHW32, trt.int8)],
                outputs=[(trt.TensorFormat.CHW32, trt.float16)],
            ),
            np.ones((3, 4, 5), dtype=np.int64),
            np.ones(5, dtype=np.int64),
            np.zeros((4, 5), dtype=np.float32),
            np.random.random_sample((3, 5)),
            make_iter_result(),
            RunResults([("runner0", [make_iter_result()]), ("runner0", [make_iter_result()])]),
        ],
        ids=lambda x: type(x),
    )
    def test_serde(self, obj):
        encoded = to_json(obj)
        decoded = from_json(encoded)
        if isinstance(obj, np.ndarray):
            assert np.array_equal(decoded, obj)
        else:
            assert decoded == obj

    @pytest.mark.parametrize("obj", JSONABLE_CASES)
    def test_to_from_json(self, obj):
        encoded = obj.to_json()
        decoded = type(obj).from_json(encoded)
        assert decoded == obj

    @pytest.mark.parametrize("obj", JSONABLE_CASES)
    def test_save_load(self, obj):
        with util.NamedTemporaryFile("w+") as f:
            obj.save(f)
            decoded = type(obj).load(f)
            assert decoded == obj

    def test_cannot_save_load_to_different_types(self):
        run_result = JSONABLE_CASES[0]
        encoded = run_result.to_json()

        with pytest.raises(PolygraphyException, match="JSON cannot be decoded into"):
            TacticReplayData.from_json(encoded)


def test_load_json_errors_if_file_nonexistent():
    with pytest.raises(FileNotFoundError, match="No such file"):
        load_json("polygraphy-nonexistent-path")
