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
import os

from polygraphy.common.struct import TensorMetadata
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.util import misc


class ModelArgs(BaseArgs):
    def __init__(self, model_required=False, inputs="--inputs", model_type=None):
        self._model_required = model_required
        self._inputs = inputs
        # If model type is provided, it means the tool only supports a single type of model.
        self._model_type = model_type


    def add_to_parser(self, parser):
        model_args = parser.add_argument_group("Model", "Options for the model")
        model_args.add_argument("model_file", help="Path to the model", nargs=None if self._model_required else '?')
        if self._model_type is None:
            model_args.add_argument("--model-type", help="The type of the input model: {{'frozen': TensorFlow frozen graph, 'keras': Keras model, "
                                    "'ckpt': TensorFlow checkpoint directory, 'onnx': ONNX model, 'engine': TensorRT engine, 'uff': UFF file [deprecated], "
                                    "'caffe': Caffe prototxt [deprecated]}}", choices=["frozen", "keras", "ckpt", "onnx", "uff", "caffe", "engine"],
                                    default=None)
        if self._inputs:
            model_args.add_argument(self._inputs, self._inputs.replace("inputs", "input") + "-shapes", help="Model input(s) and their shape(s). Format: {arg_name} <name>,<shape>. "
                                    "For example: {arg_name} image:1,1x3x224x224 other_input,10".format(arg_name=self._inputs), nargs="+", default=None, dest="input_shapes")


    def parse(self, args):
        def determine_model_type():
            if tools_util.get(args, "model_type") is not None:
                return args.model_type.lower()

            if tools_util.get(args, "model_file") is None:
                return None

            def use_ext(ext_mapping):
                file_ext = os.path.splitext(args.model_file)[-1]
                if file_ext in ext_mapping:
                    return ext_mapping[file_ext]


            runners = misc.default_value(tools_util.get(args, "runners"), [])
            if tools_util.get(args, "ckpt") or os.path.isdir(args.model_file):
                return "ckpt"
            elif "tf" in runners or "trt_legacy" in runners:
                if args.caffe_model:
                    return "caffe"
                ext_mapping = {".hdf5": "keras", ".uff": "uff", ".prototxt": "caffe", ".onnx": "onnx", ".engine": "engine", ".plan": "engine"}
                return use_ext(ext_mapping) or "frozen"
            else:
                # When no framework is provided, some extensions can be ambiguous
                ext_mapping = {".hdf5": "keras", ".graphdef": "frozen", ".onnx": "onnx", ".uff": "uff", ".engine": "engine", ".plan": "engine"}
                model_type = use_ext(ext_mapping)
                if model_type:
                    return model_type

            G_LOGGER.critical("Could not automatically determine model type for: {:}\n"
                            "Please explicitly specify the type with the --model-type option".format(
                                args.model_file))


        if tools_util.get(args, "model_file"):
            G_LOGGER.verbose("Model: {:}".format(args.model_file))
            if not os.path.exists(args.model_file):
                G_LOGGER.warning("Model path does not exist: {:}".format(args.model_file))
            args.model_file = os.path.abspath(args.model_file)


        if tools_util.get(args, "input_shapes"):
            self.input_shapes = tools_util.parse_meta(tools_util.get(args, "input_shapes"), includes_dtype=False) # TensorMetadata
        else:
            self.input_shapes = TensorMetadata()


        self.model_file = args.model_file
        self.model_type = misc.default_value(self._model_type, determine_model_type())
