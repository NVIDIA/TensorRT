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
from polygraphy import mod
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class Tf2OnnxLoaderArgs(BaseArgs):
    def add_to_parser(self, parser):
        tf_onnx_args = parser.add_argument_group("TensorFlow-ONNX Loader", "Options for TensorFlow-ONNX conversion")
        tf_onnx_args.add_argument("--opset", help="Opset to use when converting to ONNX", default=None, type=int)
        tf_onnx_args.add_argument(
            "--no-const-folding",
            help="Do not fold constants in the TensorFlow graph prior to conversion",
            action="store_true",
            default=None,
        )

    def register(self, maker):
        from polygraphy.tools.args.tf.loader import TfLoaderArgs

        if isinstance(maker, TfLoaderArgs):
            self.tf_loader_args = maker

    def check_registered(self):
        assert self.tf_loader_args is not None, "TfLoaderArgs is required!"

    def parse(self, args):
        self.opset = args_util.get(args, "opset")
        self.fold_constant = False if args_util.get(args, "no_const_folding") else None

    def add_to_script(self, script, suffix=None):
        G_LOGGER.verbose(
            "Attempting to load as a TensorFlow model, using TF2ONNX to convert to ONNX. "
            "If this is not correct, please specify --model-type",
            mode=LogMode.ONCE,
        )
        script.add_import(imports=["OnnxFromTfGraph"], frm="polygraphy.backend.onnx")
        loader_str = make_invocable(
            "OnnxFromTfGraph",
            self.tf_loader_args.add_to_script(script, disable_custom_outputs=True, suffix=suffix),
            opset=self.opset,
            fold_constant=self.fold_constant,
        )
        loader_name = script.add_loader(loader_str, "export_onnx_from_tf", suffix=suffix)
        return loader_name
