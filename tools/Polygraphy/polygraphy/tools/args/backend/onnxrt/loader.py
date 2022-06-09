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
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.args.backend.onnx.loader import OnnxLoadArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class OnnxrtSessionArgs(BaseArgs):
    """
    ONNX-Runtime Session Creation: creating an ONNX-Runtime Inference Session

    Depends on:

        - OnnxLoadArgs
        - ModelArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--providers",
            "--execution-providers",
            dest="providers",
            help="A list of execution providers to use in order of priority. "
            "Each provider may be either an exact match or a case-insensitive partial match "
            "for the execution providers available in ONNX-Runtime. For example, a value of 'cpu' would "
            "match the 'CPUExecutionProvider'",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            providers (List[str]): A list of execution providers.
        """
        self.providers = args_util.get(args, "providers")

    def add_to_script_impl(self, script):
        if self.arg_groups[OnnxLoadArgs].must_use_onnx_loader():
            onnx_name = self.arg_groups[OnnxLoadArgs].add_to_script(script, serialize_model=True)
        else:
            onnx_name = self.arg_groups[ModelArgs].path

        script.add_import(imports=["SessionFromOnnx"], frm="polygraphy.backend.onnxrt")
        loader_name = script.add_loader(
            make_invocable("SessionFromOnnx", onnx_name, providers=self.providers), "build_onnxrt_session"
        )
        return loader_name

    def load_onnxrt_session(self):
        """
        Loads an ONNX-Runtime Inference Session according to arguments provided on the command-line.

        Returns:
            onnxruntime.InferenceSession
        """
        loader = args_util.run_script(self.add_to_script)
        return loader()
