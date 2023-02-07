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
from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.logger import G_LOGGER

onnxrt = mod.lazy_import("onnxruntime")


@mod.export(funcify=True)
class SessionFromOnnx(BaseLoader):
    """
    Functor that builds an ONNX-Runtime inference session.
    """

    def __init__(self, model_bytes, providers=None):
        """
        Builds an ONNX-Runtime inference session.

        Args:
            model_bytes (Union[Union[bytes, str], Callable() -> Union[bytes, str]]):
                    A serialized ONNX model or a path to a model or a callable that returns one of those.

            providers (Sequence[str]):
                    A sequence of execution providers to use in order of priority.
                    Each element of the sequence may be either an exact match or a case-insensitive partial match
                    for the execution providers available in ONNX-Runtime. For example, a value of "cpu" would
                    match the "CPUExecutionProvider".
                    Defaults to ``["cpu"]``.

        """
        self._model_bytes_or_path = model_bytes
        self.providers = util.default(providers, ["cpu"])

    def call_impl(self):
        """
        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = util.invoke_if_callable(self._model_bytes_or_path)

        available_providers = onnxrt.get_available_providers()
        providers = []
        for prov in self.providers:
            matched_prov = util.find_str_in_iterable(prov, available_providers)
            if matched_prov is None:
                G_LOGGER.critical(
                    f"Could not find specified ONNX-Runtime execution provider.\nNote: Requested provider was: {prov}, but available providers are: {available_providers}"
                )
            providers.append(matched_prov)

        G_LOGGER.start(f"Creating ONNX-Runtime Inference Session with providers: {providers}")
        return onnxrt.InferenceSession(model_bytes, providers=providers)
