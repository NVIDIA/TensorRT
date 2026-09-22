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

import os

import numpy as np
import pytest
import tensorrt as trt
import torch

from polygraphy import constants, util
from polygraphy.backend.trt import Algorithm, TacticReplayData, TensorInfo
from polygraphy.comparator import IterationResult, RunResults
from polygraphy.exception import PolygraphyException
from polygraphy.json import Decoder, Encoder, from_json, load_json, save_json, to_json
from polygraphy.json.serde import (
    ITERATION_FILE_INDEX_WIDTH,
    IterationWriter,
    list_iteration_files,
)


class Dummy:
    def __init__(self, x):
        self.x = x


@Encoder.register(Dummy)
def encode_dummy(dummy):
    return {"x": dummy.x}


@Decoder.register(Dummy)
def decode_dummy(dct):
    assert len(dct) == 1  # Custom type markers should be removed at this point
    return Dummy(x=dct["x"])


class NoDecoder:
    def __init__(self, x):
        self.x = x


@Encoder.register(NoDecoder)
def encode_nodecoder(no_decoder):
    return {"x": no_decoder.x}


class TestEncoder:
    def test_registered(self):
        d = Dummy(x=-1)
        d_json = to_json(d)
        assert encode_dummy(d) == {"x": d.x, constants.TYPE_MARKER: "Dummy"}
        expected = f'{{\n    "x": {d.x},\n    "{constants.TYPE_MARKER}": "Dummy"\n}}'
        assert d_json == expected


class TestDecoder:
    def test_object_pairs_hook(self):
        d = Dummy(x=-1)
        d_json = to_json(d)

        new_d = from_json(d_json)
        assert new_d.x == d.x

    def test_error_on_no_decoder(self):
        d = NoDecoder(x=1)
        d_json = to_json(d)

        with pytest.raises(
            PolygraphyException,
            match="Could not decode serialized type: NoDecoder. This could be because a required module is missing.",
        ):
            from_json(d_json)

    def test_names_correct(self):
        # Trigger `try_register_common_json`
        d = Dummy(x=-1)
        to_json(d)

        # If the name of a class changes, then we need to specify an `alias` when registering
        # to retain backwards compatibility.
        assert set(Decoder.polygraphy_registered.keys()) == {
            "__polygraphy_encoded_Algorithm",
            "__polygraphy_encoded_AttentionLayerHint",
            "__polygraphy_encoded_DistCollective",
            "__polygraphy_encoded_Dummy",
            "__polygraphy_encoded_FormattedArray",
            "__polygraphy_encoded_FusedAttention",
            "__polygraphy_encoded_IterationContext",
            "__polygraphy_encoded_IterationResult",
            "__polygraphy_encoded_LazyArray",
            "__polygraphy_encoded_ndarray",
            "__polygraphy_encoded_RunResults",
            "__polygraphy_encoded_ShardHints",
            "__polygraphy_encoded_ShardTensor",
            "__polygraphy_encoded_TacticReplayData",
            "__polygraphy_encoded_Tensor",
            "__polygraphy_encoded_TensorInfo",
            "__polygraphy_encoded_AccuracyResult",
            "__polygraphy_encoded_AccuracyResults",
            "__polygraphy_encoded_OutputCompareResult",
            "__polygraphy_encoded_PerceptualMetricsResult",
            "__polygraphy_encoded_L2Result",
            "__polygraphy_encoded_CosineSimilarityResult",
            "__polygraphy_encoded_PsnrResult",
            "__polygraphy_encoded_SnrResult",
            "__polygraphy_encoded_SimpleThreshold",
            "__polygraphy_encoded_L2Threshold",
            "__polygraphy_encoded_CosineSimilarityThreshold",
            "__polygraphy_encoded_PsnrThreshold",
            "__polygraphy_encoded_SnrThreshold",
            "__polygraphy_encoded_LpipsThreshold",
            "Algorithm",
            "AttentionLayerHint",
            "DistCollective",
            "Dummy",
            "FormattedArray",
            "FusedAttention",
            "IterationContext",
            "IterationResult",
            "LazyArray",
            "LazyNumpyArray",
            "ndarray",
            "RunResults",
            "ShardHints",
            "ShardTensor",
            "TacticReplayData",
            "Tensor",
            "TensorInfo",
            "AccuracyResult",
            "AccuracyResults",
            "OutputCompareResult",
            "PerceptualMetricsResult",
            "L2Result",
            "CosineSimilarityResult",
            "PsnrResult",
            "SnrResult",
            "SimpleThreshold",
            "L2Threshold",
            "CosineSimilarityThreshold",
            "PsnrThreshold",
            "SnrThreshold",
            "LpipsThreshold",
        }


def make_algo():
    return Algorithm(
        implementation=4,
        tactic=5,
        # Should work even if strides are not set
        inputs=[
            TensorInfo(trt.float32, (1, 2), -1, 1),
            TensorInfo(trt.float32, (1, 2), -1, 1),
        ],
        outputs=[TensorInfo(trt.float32, (2, 3), -1, 1)],
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
            TensorInfo(trt.float32, (1, 2, 3), -1, 1),
            Algorithm(
                implementation=4,
                tactic=5,
                inputs=[TensorInfo(trt.float32, (1, 2, 3), -1, 1)],
                outputs=[TensorInfo(trt.float32, (1, 2, 3), -1, 1)],
            ),
            Algorithm(
                implementation=4,
                tactic=5,
                inputs=[
                    TensorInfo(trt.float32, (1, 2, 3), -1, 1),
                    TensorInfo(trt.int8, (1, 2, 3), -1, 1),
                ],
                outputs=[TensorInfo(trt.float16, (1, 2, 3), -1, 1)],
            ),
            np.ones((3, 4, 5), dtype=np.int64),
            np.ones(5, dtype=np.int64),
            np.zeros((4, 5), dtype=np.float32),
            np.random.random_sample((3, 5)),
            torch.ones((3, 4, 5), dtype=torch.int64),
            make_iter_result(),
            RunResults(
                [("runner0", [make_iter_result()]), ("runner0", [make_iter_result()])]
            ),
        ],
        ids=lambda x: type(x),
    )
    def test_serde(self, obj):
        encoded = to_json(obj)
        decoded = from_json(encoded)
        if isinstance(obj, np.ndarray):
            assert np.array_equal(decoded, obj)
        elif isinstance(obj, torch.Tensor):
            assert torch.equal(decoded, obj)
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


class TestIterationWriter:
    def test_single_file_snapshots_on_accumulate(self, tmp_path):
        # With copy_on_accumulate, a single-file writer snapshots each appended object so later
        # mutation of a reused buffer does not corrupt what was already accumulated.
        path = str(tmp_path / "inputs.json")
        buffer = {"x": np.zeros((2,), dtype=np.float32)}
        with IterationWriter(path, copy_on_accumulate=True) as writer:
            writer.append(buffer)
            buffer["x"][:] = 99.0  # mutate the reused buffer before the next append
            writer.append(buffer)
        loaded = load_json(path)
        assert [float(fd["x"][0]) for fd in loaded] == [0.0, 99.0]

    def test_disabled_when_path_none(self, tmp_path):
        # A None path disables writing entirely (used when --save-* is not provided).
        writer = IterationWriter(None)
        writer.append({"x": np.zeros((1,), dtype=np.float32)})
        writer.flush()
        assert os.listdir(tmp_path) == []

    def test_rejects_nonempty_directory(self, tmp_path):
        # A per-iteration (extensionless) directory must be empty or not yet exist, so existing
        # files are never overwritten or deleted.
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        (out_dir / "leftover.json").write_text("{}")
        with pytest.raises(PolygraphyException, match="not empty"):
            IterationWriter(str(out_dir))

    def test_rejects_file_at_extensionless_path(self, tmp_path):
        # An extensionless path that already exists as a regular file is an error (it is treated as
        # a directory of per-iteration files).
        path = tmp_path / "outputs"
        path.write_text("not a directory")
        with pytest.raises(PolygraphyException, match="a file already exists there"):
            IterationWriter(str(path))


class TestListIterationFiles:
    def test_ordered_lexically(self, tmp_path):
        # Zero-padded index names (as written by IterationWriter) sort correctly under a pure
        # lexical sort, so 10 follows 2 rather than 1.
        for index in [0, 1, 2, 10, 11]:
            save_json(
                {"x": np.zeros((1,), dtype=np.float32)},
                str(tmp_path / f"{index:0{ITERATION_FILE_INDEX_WIDTH}d}.json"),
            )
        (tmp_path / "manifest.json").write_text("{}")
        files = list_iteration_files(str(tmp_path))
        assert [os.path.basename(f) for f in files] == [
            "0000000000.json",
            "0000000001.json",
            "0000000002.json",
            "0000000010.json",
            "0000000011.json",
            "manifest.json",
        ]
