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
import os

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.backend.runner_select import RunnerSelectArgs


@mod.export()
class ModelArgs(BaseArgs):
    """
    Model: the model

    Depends on:

        - RunnerSelectArgs: if guess_model_type_from_runners == True
    """

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

    def __init__(
        self,
        model_opt_required: bool = None,
        required_model_type: str = None,
        input_shapes_opt_name: str = None,
        input_shapes_opt_doc: str = None,
        guess_model_type_from_runners: bool = None,
    ):
        """
        Args:
            model_opt_required (bool):
                    Whether the model argument is required.
                    Defaults to False.
            required_model_type (str):
                    The required type of model. Use a value of ``None`` for tools that work with multiple model types.
                    If provided, it causes the tool to support only one type of model and disables the ``--model-type`` option.
                    Defaults to None.
            input_shapes_opt_name (str):
                    The name of the option used to specify input shapes.
                    A second option name will be automatically added by dropping the final ``s`` in the specified
                    option and suffixing ``-shapes``. For example, a value of "inputs" would generate an
                    alias called "--input-shapes".
                    Defaults to "inputs".
                    Use a value of ``False`` to disable the option.
            input_shapes_opt_doc (str):
                    Custom help text output for the input shapes option.
            guess_model_type_from_runners (bool):
                    Whether to guess the model type based on which runners have been specified, if any.
                    Defaults to False.
        """
        super().__init__()
        self._model_opt_required = util.default(model_opt_required, False)
        self._input_shapes_opt_name = util.default(input_shapes_opt_name, "inputs")
        # If model type is provided, it means the tool only supports a single type of model.
        self._required_model_type = required_model_type
        self._input_shapes_opt_doc = util.default(
            input_shapes_opt_doc,
            "Model input(s) and their shape(s). "
            "Used to determine shapes to use while generating input data for inference",
        )
        self._guess_model_type_from_runners = util.default(guess_model_type_from_runners, False)

    def add_parser_args_impl(self):
        self.group.add_argument("model_file", help="Path to the model", nargs=None if self._model_opt_required else "?")

        if self._required_model_type is None:
            self.group.add_argument(
                "--model-type",
                help="The type of the input model: {{'frozen': TensorFlow frozen graph; 'keras': Keras model; "
                "'ckpt': TensorFlow checkpoint directory; 'onnx': ONNX model; 'engine': TensorRT engine; 'trt-network-script': "
                "A Python script that defines a `load_network` function that takes no arguments and returns a TensorRT Builder, "
                "Network, and optionally Parser. If the function name is not `load_network`, it can be specified after the model file, "
                "separated by a colon. For example: `my_custom_script.py:my_func`; "
                "'uff': UFF file [deprecated]; 'caffe': Caffe prototxt [deprecated]}}",
                choices=ModelArgs.ModelType.VALID_TYPES,
                default=None,
            )

        if self._input_shapes_opt_name:
            arg_name = f"--{self._input_shapes_opt_name.rstrip('s')}-shapes"
            self.group.add_argument(
                arg_name,
                f"--{self._input_shapes_opt_name}",
                help=f"{self._input_shapes_opt_doc}. Format: {arg_name} <name>:<shape>. "
                f"For example: {arg_name} image:[1,3,224,224] other_input:[10]",
                nargs="+",
                default=None,
                dest="input_shapes",
            )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            input_shapes (TensorMetadata): Input names and their shapes.
            path (str): Path to the model.
            model_type (ModelArgs.ModelType): The type of model.
            extra_model_info (str):
                    Any extra model information specified after the model argument, separated by a colon.
                    The meaning of this information may be specific to each model type.
                    In most cases, no extra model information is provided.
        """

        def determine_model_type(model_file):
            model_type = args_util.get(args, "model_type")
            if model_type is not None:
                return model_type.lower()

            if model_file is None:
                return None

            def use_ext(ext_mapping):
                file_ext = os.path.splitext(model_file)[-1]
                if file_ext in ext_mapping:
                    return ext_mapping[file_ext]

            runner_opts = []
            if self._guess_model_type_from_runners:
                if not hasattr(self.arg_groups[RunnerSelectArgs], "runners"):
                    G_LOGGER.internal_error(
                        "RunnerSelectArgs must be parsed before ModelArgs when `guess_model_type_from_runners` is enabled!"
                    )
                runner_opts = list(self.arg_groups[RunnerSelectArgs].runners.keys())

            if args_util.get(args, "ckpt") or os.path.isdir(model_file):
                return "ckpt"
            elif "tf" in runner_opts or "trt-legacy" in runner_opts:
                if args_util.get(args, "caffe_model"):
                    return "caffe"
                return use_ext(ModelArgs.EXT_MODEL_TYPE_MAPPING) or "frozen"
            else:
                model_type = use_ext(ModelArgs.EXT_MODEL_TYPE_MAPPING)
                if model_type:
                    return model_type

            G_LOGGER.critical(
                f"Could not automatically determine model type for: {model_file}"
                f"\nPlease explicitly specify the type with the --model-type option"
            )

        self.input_shapes = TensorMetadata()
        if args_util.get(args, "input_shapes"):
            self.input_shapes = args_util.parse_meta(args_util.get(args, "input_shapes"), includes_dtype=False)

        self.path = None
        self.extra_model_info = None

        self.path, self.extra_model_info = args_util.parse_script_and_func_name(args_util.get(args, "model_file"))

        if self.path is not None:
            G_LOGGER.verbose(f"Model: {self.path}")
            if not os.path.exists(self.path):
                G_LOGGER.warning(f"Model path does not exist: {self.path}")
            self.path = os.path.abspath(self.path)

        model_type_str = self._required_model_type if self._required_model_type else determine_model_type(self.path)
        self.model_type = ModelArgs.ModelType(model_type_str) if model_type_str else None

        # Set up extra_model_info defaults for each model type
        if self.model_type == "trt-network-script":
            if not self.path or not self.path.endswith(".py"):
                G_LOGGER.critical(
                    f"TensorRT network scripts must exist and have '.py' extensions.\n"
                    f"Note: Provided network script path was: {self.path}"
                )

            self.extra_model_info = util.default(self.extra_model_info, "load_network")
