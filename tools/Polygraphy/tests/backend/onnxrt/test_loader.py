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
from polygraphy.backend.onnxrt import SessionFromOnnx
from polygraphy.exception import PolygraphyException
from tests.models.meta import ONNX_MODELS

import onnxruntime as onnxrt
import pytest


class TestSessionFromOnnx:
    def test_defaults(self):
        model = ONNX_MODELS["identity"]
        loader = SessionFromOnnx(model.loader)
        sess = loader()

        assert sess
        assert isinstance(sess, onnxrt.InferenceSession)
        assert sess.get_providers() == ["CPUExecutionProvider"]

    @pytest.mark.parametrize(
        "providers,expected",
        [
            (["cpu"], ["CPUExecutionProvider"]),
            (["CPU"], ["CPUExecutionProvider"]),
        ],
    )
    def test_provider_matching(self, providers, expected):
        model = ONNX_MODELS["identity"]
        loader = SessionFromOnnx(model.loader, providers=providers)
        sess = loader()

        assert sess
        assert isinstance(sess, onnxrt.InferenceSession)
        assert sess.get_providers() == expected

    @pytest.mark.skipif(
        "TensorrtExecutionProvider" not in onnxrt.get_available_providers(),
        reason="Skip test if TensorrtExecutionProvider is not available",
    )
    @pytest.mark.parametrize(
        "providers,expected_dict",
        [
            # Searches for 'tensorrt' as the execution provider's name
            (["tensorrt", "cpu"], {"TensorrtExecutionProvider": {}, "CPUExecutionProvider": {}}),
            # Searches for the execution provider's name if the item is a tuple in the format (EP name, EP options)
            (
                    [("TensorrtExecutionProvider", {"trt_op_types_to_exclude": "Add"}), "CPUExecutionProvider"],
                    {"TensorrtExecutionProvider": {"trt_op_types_to_exclude": "Add"}, "CPUExecutionProvider": {}}
            ),
        ],
    )
    def test_provider_with_options(self, providers, expected_dict):
        model = ONNX_MODELS["identity"]
        loader = SessionFromOnnx(model.loader, providers=providers)
        sess = loader()

        assert sess
        assert isinstance(sess, onnxrt.InferenceSession)
        assert sess.get_providers() == list(expected_dict.keys())

        provider_options = sess.get_provider_options()
        for k, v in provider_options.items():
            if expected_dict.get(k, None):
                assert set(expected_dict[k].items()).issubset(v.items())

    def test_invalid_providers_raise_errors(self):
        model = ONNX_MODELS["identity"]
        loader = SessionFromOnnx(model.loader, providers=["cpu", "not_a_real_provider"])
        with pytest.raises(
            PolygraphyException,
            match="Could not find specified ONNX-Runtime execution provider",
        ):
            loader()
