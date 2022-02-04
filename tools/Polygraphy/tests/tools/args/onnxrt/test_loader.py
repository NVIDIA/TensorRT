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

import onnxruntime as onnxrt
from polygraphy.tools.args import ModelArgs, OnnxLoaderArgs, OnnxrtSessionArgs
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper


class TestOnnxrtSessionArgs(object):
    def test_execution_providers(self):
        arg_group = ArgGroupTestHelper(OnnxrtSessionArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        arg_group.parse_args([ONNX_MODELS["identity_identity"].path, "--providers", "cpu"])
        sess = arg_group.load_onnxrt_session()

        assert sess
        assert isinstance(sess, onnxrt.InferenceSession)
        assert sess.get_providers() == ["CPUExecutionProvider"]
