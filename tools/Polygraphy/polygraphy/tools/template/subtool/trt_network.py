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
from polygraphy.tools.args import (
    ModelArgs,
    OnnxFromTfArgs,
    OnnxLoadArgs,
    TfLoadArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
)
from polygraphy.tools.script import Script, inline, safe
from polygraphy.tools.template.subtool.base import BaseTemplateTool


class TrtNetwork(BaseTemplateTool):
    """
    Generate a template script to create a TensorRT network using the TensorRT network API,
    optionally starting from an existing model.
    """

    def __init__(self):
        super().__init__("trt-network")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=False, input_shapes_opt_name=False),
            TfLoadArgs(allow_artifacts=False),
            OnnxFromTfArgs(),
            OnnxLoadArgs(allow_shape_inference=False, allow_from_tf=True),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
        ]

    def run_impl(self, args):
        script = Script(summary="Creates a TensorRT Network using the Network API.", always_create_runners=False)
        script.add_import(imports=["func"], frm="polygraphy")
        script.add_import(imports="tensorrt", imp_as="trt")

        if self.arg_groups[ModelArgs].path is not None:
            loader_name = self.arg_groups[TrtLoadNetworkArgs].add_to_script(script)
            params = safe("builder, network, parser")
        else:
            script.add_import(imports=["CreateNetwork"], frm="polygraphy.backend.trt")
            loader_name = safe("CreateNetwork()")
            params = safe("builder, network")

        script.append_suffix(safe("@func.extend({:})", inline(loader_name)))
        script.append_suffix(safe("def load_network({:}):", inline(params)))
        script.append_suffix(
            safe(f"{constants.TAB}pass # TODO: Set up the network here. This function should not return anything.")
        )

        script.save(args.output)
