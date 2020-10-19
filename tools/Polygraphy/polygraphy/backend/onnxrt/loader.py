#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import onnxruntime
from polygraphy.backend.base import BaseLoadModel
from polygraphy.util import misc

misc.log_module_info(onnxruntime)

class SessionFromOnnxBytes(BaseLoadModel):
    def __init__(self, model_bytes):
        """
        Functor that builds an ONNX-Runtime inference session.

        Args:
            model_bytes (Callable() -> bytes): A loader that can supply a serialized ONNX model.
        """
        self._model_bytes = model_bytes


    def __call__(self):
        """
        Builds an ONNX-Runtime inference session.

        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = misc.try_call(self._model_bytes)
        return onnxruntime.InferenceSession(model_bytes)
