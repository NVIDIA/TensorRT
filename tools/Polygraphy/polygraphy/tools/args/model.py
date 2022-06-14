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

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs


@mod.export()
class ModelArgs(BaseArgs):
    EXT_MODEL_TYPE_MAPPING = {
        ".hdf5": "keras",
        ".uff": "uff",
        ".prototxt": "caffe",
        ".onnx": "onnx",
        ".engine": "engine",
        ".plan": "engine",
        ".graphdef": "frozen",
        ".py": "trt-network-script",
    }

    class ModelType(str):
        TF_TYPES = ["frozen", "keras", "ckpt"]
        ONNX_TYPES = ["onnx"]
        TRT_TYPES = ["engine", "uff", "trt-network-script"]
        OTHER_TYPES = ["caffe"]

        VALID_TYPES = TF_TYPES + ONNX_TYPES + TRT_TYPES + OTHER_TYPES

        def __new__(cls, model_type):
            assert model_type in ModelArgs.ModelType.VALID_TYPES or model_type is None
            return str.__new__(cls, model_type)

        def is_tf(self):
            return self in ModelArgs.ModelType.TF_TYPES

        def is_onnx(self):
            return self in ModelArgs.ModelType.ONNX_TYPES

        def is_trt(self):
            return self in ModelArgs.ModelType.TRT_TYPES

    def __init__(self, model_required=False, inputs="--inputs", model_type=None, inputs_doc=None):
        super().__init__()
        self._model_required = model_required
        self._inputs = inputs
        # If model type is provided, it means the tool only supports a single type of model.
        self._model_type = model_type
        self._inputs_doc = util.default(
            inputs_doc,
            "Model input(s) and their shape(s). "
            "Used to determine shapes to use while generating input data for inference",
        )

    def add_to_parser(self, parser):
        model_args = parser.add_argument_group("Model", "Options for the model")
        model_args.add_argument("model_file", help="Path to the model", nargs=None if self._model_required else "?")
        if self._model_type is None:
            model_args.add_argument(
                "--model-type",
                help="The type of the input model: {{'frozen': TensorFlow frozen graph, 'keras': Keras model, "
                "'ckpt': TensorFlow checkpoint directory, 'onnx': ONNX model, 'engine': TensorRT engine, 'trt-network-script': "
                "A Python script that defines a `load_network` function that takes no arguments and returns a TensorRT Builder, "
                "Network, and optionally Parser, "
                "'uff': UFF file [deprecated], 'caffe': Caffe prototxt [deprecated]}}",
                choices=ModelArgs.ModelType.VALID_TYPES,
                default=None,
            )
        if self._inputs:
            model_args.add_argument(
                self._inputs.replace("inputs", "input") + "-shapes",
                self._inputs,
                help="{:}. Format: {arg_name}-shapes <name>:<shape>. "
                "For example: {arg_name}-shapes image:[1,3,224,224] other_input:[10]".format(
                    self._inputs_doc, arg_name=self._inputs.replace("inputs", "input")
                ),
                nargs="+",
                default=None,
                dest="input_shapes",
            )

    def parse(self, args):
        def determine_model_type():
            if args_util.get(args, "model_type") is not None:
                return args.model_type.lower()

            if args_util.get(args, "model_file") is None:
                return None

            def use_ext(ext_mapping):
                file_ext = os.path.splitext(args.model_file)[-1]
                if file_ext in ext_mapping:
                    return ext_mapping[file_ext]

            runners = args_util.get(args, "runners", default=[])
            if args_util.get(args, "ckpt") or os.path.isdir(args.model_file):
                return "ckpt"
            elif "tf" in runners or "trt_legacy" in runners:
                if args.caffe_model:
                    return "caffe"
                return use_ext(ModelArgs.EXT_MODEL_TYPE_MAPPING) or "frozen"
            else:
                model_type = use_ext(ModelArgs.EXT_MODEL_TYPE_MAPPING)
                if model_type:
                    return model_type

            G_LOGGER.critical(
                "Could not automatically determine model type for: {:}\n"
                "Please explicitly specify the type with the --model-type option".format(args.model_file)
            )

        if args_util.get(args, "input_shapes"):
            self.input_shapes = args_util.parse_meta(
                args_util.get(args, "input_shapes"), includes_dtype=False
            )  # TensorMetadata
        else:
            self.input_shapes = TensorMetadata()

        self.model_file = args_util.get(args, "model_file")

        if self.model_file:
            G_LOGGER.verbose("Model: {:}".format(self.model_file))
            if not os.path.exists(self.model_file):
                G_LOGGER.warning("Model path does not exist: {:}".format(self.model_file))
            self.model_file = os.path.abspath(self.model_file)

        model_type_str = self._model_type if self._model_type else determine_model_type()
        self.model_type = ModelArgs.ModelType(model_type_str) if model_type_str else None

        if self.model_type == "trt-network-script" and (not self.model_file or not self.model_file.endswith(".py")):
            G_LOGGER.critical(
                "TensorRT network scripts must exist and have '.py' extensions.\n"
                "Note: Provided network script path was: {:}".format(self.model_file)
            )
