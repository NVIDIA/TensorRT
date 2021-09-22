#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from polygraphy.backend.trt_legacy import TrtLegacyRunner, LoadNetworkFromUff, ConvertToUff, ParseNetworkFromOnnxLegacy

from tests.models.meta import TF_MODELS, ONNX_MODELS

import numpy as np


def test_uff_identity():
    model = TF_MODELS["identity"]
    loader = model.loader
    with TrtLegacyRunner(LoadNetworkFromUff(ConvertToUff(loader))) as runner:
        assert runner.is_active
        feed_dict = {"Input": np.random.random_sample(size=(1, 15, 25, 30)).astype(np.float32)}
        outputs = runner.infer(feed_dict)
        assert np.all(outputs["Identity_2"] == feed_dict["Input"])
    assert not runner.is_active


def test_can_construct_onnx_loader():
    model = ONNX_MODELS["identity"].path
    loader = ParseNetworkFromOnnxLegacy(model)
