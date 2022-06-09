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
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.exception import PolygraphyException
from tests.models.meta import ONNX_MODELS


def test_infer_raises_if_runner_inactive():
    runner = OnnxrtRunner(SessionFromOnnx(ONNX_MODELS["identity"].loader))
    feed_dict = {"x": np.ones((1, 1, 2, 2), dtype=np.float32)}

    with pytest.raises(PolygraphyException, match="Must be activated"):
        runner.infer(feed_dict)
