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
from polygraphy.common import constants
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Script
from polygraphy.util import misc


class TrtLegacyArgs(BaseArgs):
    def add_to_parser(self, parser):
        trt_legacy_args = parser.add_argument_group("TensorRT Legacy", "[DEPRECATED] Options for TensorRT Legacy. Reuses TensorRT options, but does not support int8 mode, or dynamic shapes")
        trt_legacy_args.add_argument("-p", "--preprocessor", help="The preprocessor to use for the UFF converter", default=None)
        trt_legacy_args.add_argument("--uff-order", help="The order of the input", default=None)
        trt_legacy_args.add_argument("--batch-size", metavar="SIZE", help="The batch size to use in TensorRT when it cannot be automatically determined", type=int, default=None)
        trt_legacy_args.add_argument("--model", help="Model file for Caffe models. The deploy file should be provided as the model_file positional argument", dest="caffe_model")
        trt_legacy_args.add_argument("--save-uff", help="Save intermediate UFF files", action="store_true", default=None)


    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs
        from polygraphy.tools.args.tf.loader import TfLoaderArgs
        from polygraphy.tools.args.trt.loader import TrtLoaderArgs
        from polygraphy.tools.args.trt.runner import TrtRunnerArgs

        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, TrtLoaderArgs):
            self.trt_loader_args = maker
        if isinstance(maker, TfLoaderArgs):
            self.tf_loader_args = maker
        if isinstance(maker, TrtRunnerArgs):
            self.trt_runner_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_loader_args is not None, "TrtLoaderArgs is required!"


    def parse(self, args):
        self.trt_outputs = tools_util.get(args, "trt_outputs")
        self.caffe_model = tools_util.get(args, "caffe_model")
        self.batch_size = tools_util.get(args, "batch_size")
        self.save_uff = tools_util.get(args, "save_uff")
        self.uff_order = tools_util.get(args, "uff_order")
        self.preprocessor = tools_util.get(args, "preprocessor")


    def add_to_script(self, script):
        script.add_import(imports=["TrtLegacyRunner"], frm="polygraphy.backend.trt_legacy")
        G_LOGGER.warning("Legacy TensorRT runner only supports implicit batch TensorFlow/UFF, ONNX, and Caffe models")

        if self.model_args.model_type == "onnx":
            script.add_import(imports=["ParseNetworkFromOnnxLegacy"], frm="polygraphy.backend.trt_legacy")
            onnx_loader = self.onnx_loader_args.add_onnx_loader(script, disable_outputs=True)
            loader_name = script.add_loader(Script.format_str("ParseNetworkFromOnnxLegacy({:})", onnx_loader), "parse_network_from_onnx_legacy")
        elif self.model_args.model_type == "caffe":
            script.add_import(imports=["LoadNetworkFromCaffe"], frm="polygraphy.backend.trt_legacy")
            loader_name = script.add_loader(Script.format_str("LoadNetworkFromCaffe({:}, {:}, {:}, {:})", self.model_args.model_file, self.caffe_model,
                                                                self.trt_outputs, self.batch_size), "parse_network_from_caffe")
        else:
            script.add_import(imports=["LoadNetworkFromUff"], frm="polygraphy.backend.trt_legacy")
            if self.model_args.model_type == "uff":
                script.add_import(imports=["LoadUffFile"], frm="polygraphy.backend.trt_legacy")
                shapes = {name: shape for name, (_, shape) in self.trt_loader_args.input_shapes.items()}
                loader_name = script.add_loader(Script.format_str("LoadUffFile({:}, {:}, {:})", self.model_args.model_file, misc.default_value(shapes, {}), self.trt_outputs), "load_uff_file")
            else:
                script.add_import(imports=["ConvertToUff"], frm="polygraphy.backend.trt_legacy")
                loader_name = script.add_loader(Script.format_str("ConvertToUff({:}, save_uff={:}, preprocessor={:})", self.tf_loader_args.add_to_script(script), self.save_uff, self.preprocessor), "convert_to_uff")
            loader_name = script.add_loader(Script.format_str("LoadNetworkFromUff({:}, uff_order={:})", loader_name, self.uff_order), "uff_network_loader")


        runner_str = Script.format_str("TrtLegacyRunner({:}, {:}, {:}, fp16={:}, tf32={:}, load_engine={:}, save_engine={:}, layerwise={:}, plugins={:})",
                                        loader_name, self.trt_loader_args.workspace, self.batch_size, self.trt_loader_args.fp16, self.trt_loader_args.tf32,
                                        self.model_args.model_file if self.model_args.model_type == "engine" else None,
                                        self.trt_runner_args.save_engine, self.trt_outputs==constants.MARK_ALL, self.trt_loader_args.plugins)


        runner_name = script.add_loader(runner_str, "trt_legacy_runner")
        script.add_runner(runner_name)
        return runner_name
