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
Tests for ``trt_util.check_onnx_slice_input_lengths``.

These tests only require ``onnx`` and can run without a TensorRT installation.
"""

from __future__ import annotations

import pytest

from polygraphy import mod
from polygraphy.backend.trt import util as trt_util
from polygraphy.exception import PolygraphyException

onnx = mod.lazy_import("onnx")


def make_slice_model(start_len, end_len, axes_len=None, as_inputs=False):
    """Build an opset-13 model with a single Slice node."""
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, None])

    node_inputs = ["x", "starts", "ends"]
    initializers = [
        onnx.helper.make_tensor(
            "starts", onnx.TensorProto.INT64, [start_len], [0] * start_len
        ),
        onnx.helper.make_tensor(
            "ends", onnx.TensorProto.INT64, [end_len], [2] * end_len
        ),
    ]
    graph_inputs = [x]
    if axes_len is not None:
        node_inputs.append("axes")
        initializers.append(
            onnx.helper.make_tensor(
                "axes", onnx.TensorProto.INT64, [axes_len], list(range(axes_len))
            )
        )

    if as_inputs:
        # Keep the Slice parameters as graph inputs with static shapes instead of initializers.
        graph_inputs += [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT64, [n])
            for name, n in (("starts", start_len), ("ends", end_len))
        ]
        if axes_len is not None:
            graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "axes", onnx.TensorProto.INT64, [axes_len]
                )
            )
        initializers = []

    node = onnx.helper.make_node("Slice", node_inputs, ["y"], name="slicer")
    graph = onnx.helper.make_graph(
        [node], "g", graph_inputs, [y], initializer=initializers
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    return model


def test_slice_mismatched_lengths_raises():
    model = make_slice_model(start_len=1, end_len=2, axes_len=2)
    with pytest.raises(PolygraphyException, match="slicer"):
        trt_util.check_onnx_slice_input_lengths(model.SerializeToString())


def test_slice_mismatched_lengths_from_path(tmp_path):
    model = make_slice_model(start_len=1, end_len=2, axes_len=2)
    path = tmp_path / "slice_bad.onnx"
    onnx.save(model, path)
    with pytest.raises(PolygraphyException, match="starts=1"):
        trt_util.check_onnx_slice_input_lengths(str(path))


def test_slice_dynamic_inputs_with_static_shapes_raises():
    model = make_slice_model(start_len=1, end_len=2, axes_len=2, as_inputs=True)
    with pytest.raises(PolygraphyException, match="slicer"):
        trt_util.check_onnx_slice_input_lengths(model.SerializeToString())


def test_slice_matching_lengths_passes():
    model = make_slice_model(start_len=2, end_len=2, axes_len=2)
    assert trt_util.check_onnx_slice_input_lengths(model.SerializeToString()) is None


def test_slice_no_axes_passes():
    model = make_slice_model(start_len=2, end_len=2)
    assert trt_util.check_onnx_slice_input_lengths(model.SerializeToString()) is None


def test_slice_steps_omitted_passes():
    model = make_slice_model(start_len=1, end_len=1)
    assert trt_util.check_onnx_slice_input_lengths(model.SerializeToString()) is None


def test_model_without_slice_passes():
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])
    node = onnx.helper.make_node("Relu", ["x"], ["y"])
    graph = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    assert trt_util.check_onnx_slice_input_lengths(model.SerializeToString()) is None
