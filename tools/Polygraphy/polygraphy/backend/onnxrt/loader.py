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
from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.logger import G_LOGGER
import os

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

    @util.check_called_by("__call__")
    def call_impl(self):
        """
        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = util.invoke_if_callable(self._model_bytes_or_path)

        available_providers = onnxrt.get_available_providers()
        providers = []
        for prov in self.providers:
            matched_prov_name = util.find_str_in_iterable(prov[0] if isinstance(prov, tuple) else prov, available_providers)
            matched_prov = (matched_prov_name, prov[1]) if isinstance(prov, tuple) else matched_prov_name
            if matched_prov is None:
                G_LOGGER.critical(
                    f"Could not find specified ONNX-Runtime execution provider.\nNote: Requested provider was: {prov}, but available providers are: {available_providers}"
                )
            providers.append(matched_prov)

        G_LOGGER.start(
            f"Creating ONNX-Runtime Inference Session with providers: {providers}"
        )
        # ONNX Runtime tried to bind each thread to a logical CPU, but not all assigned cpu cores are available on some platforms.
        # Set the number of threads within each operator and between operators the number of usable CPUs to avoid crash in onnxruntime on those platforms.
        options = onnxrt.SessionOptions()
        try:
            # sched_getaffinity is only available on UNIX platforms
            process_cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            process_cpu_count = 1

        options.intra_op_num_threads = process_cpu_count
        options.inter_op_num_threads = process_cpu_count
        return onnxrt.InferenceSession(
            model_bytes, providers=providers, sess_options=options
        )
