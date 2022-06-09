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
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.args.backend.onnxrt.loader import OnnxrtSessionArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class OnnxrtRunnerArgs(BaseRunnerArgs):
    """
    ONNX-Runtime Inference: running inference with ONNX-Runtime.

    Depends on:

        - OnnxrtSessionArgs
    """

    def get_name_opt_impl(self):
        return "ONNX-Runtime", "onnxrt"

    def add_to_script_impl(self, script):
        script.add_import(imports=["OnnxrtRunner"], frm="polygraphy.backend.onnxrt")
        script.add_runner(make_invocable("OnnxrtRunner", self.arg_groups[OnnxrtSessionArgs].add_to_script(script)))
