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

from collections import namedtuple

import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import Algorithm, TacticRecorder, TacticReplayData, TacticReplayer
from polygraphy.exception import PolygraphyException

ALGO_EQ_CASES = [
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        True,
    ),  # Same
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            7, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        False,
    ),  # Different implementation
    (
        Algorithm(
            6, 2, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        False,
    ),  # Different tactic
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.CHW32, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        False,
    ),  # Different input format
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.int8)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]),
        False,
    ),  # Different input data type
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.CHW32, trt.float32)]
        ),
        False,
    ),  # Different output format
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.int8)]),
        False,
    ),  # Different output data type
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)] * 2, outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        False,
    ),  # Different number of inputs
    (
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)] * 2
        ),
        Algorithm(
            6, 1, inputs=[(trt.TensorFormat.LINEAR, trt.float32)], outputs=[(trt.TensorFormat.LINEAR, trt.float32)]
        ),
        False,
    ),  # Different number of outputs
]


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestAlgorithm:
    @pytest.mark.parametrize("left, right, expected", ALGO_EQ_CASES)
    def test_equality(self, left, right, expected):
        assert (left == right) == expected


FakeAlgorithmContext = namedtuple("FakeAlgorithmContext", ["name", "num_inputs", "num_outputs"])
FakeAlgorithm = namedtuple("FakeAlgorithm", ["algorithm_variant", "io_info"])
FakeAlgorithm.get_algorithm_io_info = lambda this, index: this.io_info[index]

FakeAlgorithmVariant = namedtuple("FakeAlgorithmVariant", ["implementation", "tactic"])
FakeAlgorithmIOInfo = namedtuple("FakeAlgorithmIOInfo", ["tensor_format", "dtype", "strides"])


def fake_context(name):
    return FakeAlgorithmContext(name=name, num_inputs=1, num_outputs=1)


def fake_algo(implementation=6, tactic=0, io=None):
    io_info = [FakeAlgorithmIOInfo(tensor_format=trt.TensorFormat.LINEAR, dtype=trt.float32, strides=(4, 5, 6))] * 2
    if io:
        io_info = []
        for fmt, dtype, strides in io:
            io_info.append(FakeAlgorithmIOInfo(tensor_format=fmt, dtype=dtype, strides=strides))

    trt_algo = FakeAlgorithm(algorithm_variant=FakeAlgorithmVariant(implementation, tactic), io_info=io_info)
    return trt_algo


@pytest.fixture(params=[True, False], ids=["path", "object"])
def replay(request):
    """
    Returns:
        Tuple[FakeAlgorithmContext, Algorithm, FakeAlgorithm,
              Union[str, TacticReplayData], Union[str, TacticReplayData]]:
                This fixture returns 5 things:
                1. A fake TensorRT algorithm context
                2. A Polygraphy Algorithm instance
                3. A fake TensorRT algorithm (with the same information as (2))
                4. An input tactic replay data, populated with the Polygraphy Algorithm (2), either
                    as a ``TacticReplayData`` instance, or a path.
                5. An output tactic replay data, empty, either as a ``TacticReplayData`` instance, or
                    a path.
    """
    jsonify = request.param

    name = "node_of_y"
    context = fake_context(name)

    trt_algo = fake_algo()
    poly_algo = Algorithm.from_trt(context, trt_algo)

    in_replay_data = TacticReplayData().add(name, poly_algo)
    out_replay_data = TacticReplayData()
    if jsonify:
        inpath = util.NamedTemporaryFile("w")
        in_replay_data.save(inpath.name)
        in_replay_data = inpath.name

        outpath = util.NamedTemporaryFile("r")
        out_replay_data = outpath.name

    yield context, poly_algo, trt_algo, in_replay_data, out_replay_data


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestReplayer:
    def test_basic(self, replay):
        context, _, algo, replay_data, _ = replay
        replayer = TacticReplayer(replay_data)
        selected = replayer.select_algorithms(context, [fake_algo(implementation=2), algo, fake_algo(tactic=1)])
        assert selected == [1]

    def test_new_layer_falls_back(self, replay):
        _, _, _, replay_data, _ = replay
        replayer = TacticReplayer(replay_data)
        selected = replayer.select_algorithms(
            fake_context(name="new_layer"), [fake_algo(2, 1), fake_algo(3, 4), fake_algo(5, 6)]
        )
        assert selected == [0, 1, 2]

    def test_missing_algo_fails(self, replay):
        context, _, _, replay_data, _ = replay
        replayer = TacticReplayer(replay_data)
        with pytest.raises(PolygraphyException, match="was not provided by TensorRT as a choice"):
            assert replayer.select_algorithms(context, [fake_algo(2, 1)]) == [0]

    @pytest.mark.parametrize(
        "algo",
        [
            fake_algo(2),
            fake_algo(tactic=2),
            fake_algo(
                io=[(trt.TensorFormat.CHW32, trt.float32, (1, 2)), (trt.TensorFormat.LINEAR, trt.float32, (1, 2))]
            ),
            fake_algo(io=[(trt.TensorFormat.LINEAR, trt.int8, (1, 2)), (trt.TensorFormat.LINEAR, trt.float32, (1, 2))]),
            fake_algo(
                io=[(trt.TensorFormat.LINEAR, trt.float32, (1, 2)), (trt.TensorFormat.CHW32, trt.float32, (1, 2))]
            ),
            fake_algo(
                io=[(trt.TensorFormat.LINEAR, trt.float32, (1, 2)), (trt.TensorFormat.LINEAR, trt.int32, (1, 2))]
            ),
        ],
    )
    def test_different_algo_fails(self, replay, algo):
        context, _, _, replay_data, _ = replay
        replayer = TacticReplayer(replay_data)
        with pytest.raises(PolygraphyException, match="was not provided by TensorRT as a choice"):
            assert replayer.select_algorithms(context, [algo]) == [0]

    def test_fails_if_wrong_selected(self, replay):
        context, _, _, replay_data, _ = replay
        replayer = TacticReplayer(replay_data)
        # We should be able to check tactics even if we're not recording them.
        with pytest.raises(PolygraphyException, match="TensorRT selected a tactic different"):
            replayer.report_algorithms([context], [fake_algo(implementation=9)])


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestRecorder:
    def test_basic(self, replay):
        context, poly_algo, algo, _, replay_data = replay
        assert isinstance(replay_data, str) or not replay_data
        replayer = TacticRecorder(replay_data)

        replayer.report_algorithms([context], [algo])
        assert len(replayer.data) == 1
        assert replayer.data[context.name] == poly_algo
