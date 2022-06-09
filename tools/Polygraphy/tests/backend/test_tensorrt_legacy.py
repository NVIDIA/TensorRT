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
from polygraphy.backend.trt import Calibrator
from polygraphy.backend.trt_legacy import ConvertToUff, LoadNetworkFromUff, ParseNetworkFromOnnxLegacy, TrtLegacyRunner
from polygraphy.comparator import DataLoader
from tests.models.meta import ONNX_MODELS, TF_MODELS


def test_uff_identity():
    pytest.importorskip("tensorflow")

    model = TF_MODELS["identity"]
    loader = model.loader
    with TrtLegacyRunner(
        LoadNetworkFromUff(ConvertToUff(loader)), int8=True, calibrator=Calibrator(DataLoader())
    ) as runner:
        assert runner.is_active
        feed_dict = {"Input": np.random.random_sample(size=(1, 15, 25, 30)).astype(np.float32)}
        outputs = runner.infer(feed_dict)
        assert np.all(outputs["Identity_2"] == feed_dict["Input"])
    assert not runner.is_active


def test_can_construct_onnx_loader():
    model = ONNX_MODELS["identity"].path
    loader = ParseNetworkFromOnnxLegacy(model)
