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
from textwrap import dedent

from polygraphy import util
from polygraphy.tools.args import ModelArgs, OnnxInferShapesArgs, OnnxLoadArgs
from polygraphy.tools.script import Script, make_invocable, safe
from polygraphy.tools.template.subtool.base import BaseTemplateTool


class OnnxGs(BaseTemplateTool):
    """
    [EXPERIMENTAL] Generate a template script to process an ONNX model with ONNX-GraphSurgeon.
    """

    def __init__(self):
        super().__init__("onnx-gs")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, required_model_type="onnx"),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(),
        ]

    def run_impl(self, args):
        script = Script(summary="Processes an ONNX model.", always_create_runners=False)
        script.add_import(imports="onnx")
        script.add_import(imports="onnx_graphsurgeon", imp_as="gs")
        script.add_import(imports="GsFromOnnx", frm="polygraphy.backend.onnx")

        loader_name = self.arg_groups[OnnxLoadArgs].add_to_script(script)
        loader_name = script.add_loader(make_invocable("GsFromOnnx", loader_name), "load_gs")

        new_model_path = util.add_file_suffix(self.arg_groups[ModelArgs].path, "_updated")

        content = safe(
            dedent(
                """
                graph = load_gs()

                # To edit inputs or outputs:
                #
                # graph.inputs = [...]
                # graph.outputs = [...]

                # To access all tensors in the model:
                #
                # tensor_map = graph.tensors() # tensor_map is a Dict[str, gs.Tensor]

                # To walk over nodes:
                #
                # for node in graph.nodes:
                #     print(f"Node: {{node.name}}, Op: {{node.op}}\\nInputs: {{node.inputs}}\\nOutputs: {{node.outputs}}")

                # Finally, you can re-export and save the edited graph.
                #
                # TIP: You may want to clean up and topologically sort the graph prior to exporting:
                #
                # graph.cleanup()
                # graph.toposort()
                #
                onnx.save(gs.export_onnx(graph), {})
                """
            ),
            new_model_path,
        )

        script.append_suffix(content)

        script.save(args.output)
