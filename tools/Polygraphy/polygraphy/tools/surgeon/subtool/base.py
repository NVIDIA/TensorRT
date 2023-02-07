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

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import LoggerArgs, OnnxLoadArgs, OnnxSaveArgs
from polygraphy.tools.base import Tool

gs = mod.lazy_import("onnx_graphsurgeon")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")


class BaseSurgeonSubtool(Tool):
    def __init__(self, name):
        super().__init__(name)

    def load_model(self, log_model=True):
        model = self.arg_groups[OnnxLoadArgs].load_onnx()
        if log_model:
            G_LOGGER.info(f"Original Model:\n{onnx_util.str_from_onnx(model)}\n\n")
        return model

    # Since new graph outputs may be added, and we don't know the types,
    # we skip type checks in ONNX-GraphSurgeon.
    def export_graph(self, graph, do_type_check=False):
        return gs.export_onnx(graph, do_type_check=do_type_check)

    def save_model(self, model, log_model=True):
        model = self.arg_groups[OnnxSaveArgs].save_onnx(model)
        if log_model:
            G_LOGGER.info(f"New Model:\n{onnx_util.str_from_onnx(model)}\n\n")

    def run_impl(self, args):
        raise NotImplementedError("Subclasses must implement run_impl!")

    def run_impl(self, args):
        def set_onnx_gs_logging_level(severity_trie):
            import os

            ONNX_GS_LOGGER = gs.logger.G_LOGGER

            sev = severity_trie.get(os.path.join("tools", "surgeon"))
            if sev >= G_LOGGER.CRITICAL:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.CRITICAL
            elif sev >= G_LOGGER.ERROR:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.ERROR
            elif sev >= G_LOGGER.WARNING:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.WARNING
            elif sev >= G_LOGGER.INFO:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.INFO
            elif sev >= G_LOGGER.EXTRA_VERBOSE:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.DEBUG
            elif sev >= G_LOGGER.SUPER_VERBOSE:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.VERBOSE
            else:
                ONNX_GS_LOGGER.severity = ONNX_GS_LOGGER.ULTRA_VERBOSE

            fmts = self.arg_groups[LoggerArgs].log_format
            for fmt in fmts:
                if fmt == "no-colors":
                    ONNX_GS_LOGGER.colors = False
                elif fmt == "timestamp":
                    ONNX_GS_LOGGER.timestamp = True
                elif fmt == "line-info":
                    ONNX_GS_LOGGER.line_info = True

        G_LOGGER.register_callback(set_onnx_gs_logging_level)
        return self.run_impl(args)
