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
from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader

onnxruntime = mod.lazy_import("onnxruntime")


@mod.export(funcify=True)
class SessionFromOnnx(BaseLoader):
    """
    Functor that builds an ONNX-Runtime inference session.
    """

    def __init__(self, model_bytes):
        """
        Builds an ONNX-Runtime inference session.

        Args:
            model_bytes (Union[Union[bytes, str], Callable() -> Union[bytes, str]]):
                    A serialized ONNX model or a path to a model or a callable that returns one of those.
        """
        self._model_bytes_or_path = model_bytes

    def call_impl(self):
        """
        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = util.invoke_if_callable(self._model_bytes_or_path)
        return onnxruntime.InferenceSession(model_bytes)
