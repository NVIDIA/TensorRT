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
from polygraphy import constants
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, TrtConfigArgs
from polygraphy.tools.script import Script, inline, safe
from polygraphy.tools.template.subtool.base import BaseTemplateTool


class TrtConfig(BaseTemplateTool):
    """
    Generate a template script to create a TensorRT builder configuration.
    """

    def __init__(self):
        super().__init__("trt-config")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=False),
            # For INT8 calibration
            DataLoaderArgs(),
            TrtConfigArgs(),
        ]

    def run_impl(self, args):
        script = Script(summary="Creates a TensorRT Builder Configuration.", always_create_runners=False)
        script.add_import(imports=["func"], frm="polygraphy")
        script.add_import(imports="tensorrt", imp_as="trt")

        loader_name = self.arg_groups[TrtConfigArgs].add_to_script(script)
        if not loader_name:
            script.add_import(imports=["CreateConfig"], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(safe("CreateConfig()"), "create_trt_config")
        params = safe("builder, network, config")

        script.append_suffix(safe("@func.extend({:})", inline(loader_name)))
        script.append_suffix(safe("def load_config({:}):", inline(params)))
        script.append_suffix(
            safe(
                f"{constants.TAB}pass # TODO: Set up the builder configuration here. This function should not return anything."
            )
        )

        script.save(args.output)
