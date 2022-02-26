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

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    DataLoaderArgs,
    ModelArgs,
    OnnxLoaderArgs,
    OnnxSaveArgs,
    OnnxShapeInferenceArgs,
    Tf2OnnxLoaderArgs,
    TfLoaderArgs,
    TrtConfigArgs,
    TrtEngineLoaderArgs,
    TrtEngineSaveArgs,
    TrtNetworkLoaderArgs,
    TrtPluginLoaderArgs,
)
from polygraphy.tools.base import Tool

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
trt_backend = mod.lazy_import("polygraphy.backend.trt")


class Convert(Tool):
    """
    Convert models to other formats.
    """

    def __init__(self):
        super().__init__("convert")
        self.subscribe_args(ModelArgs(model_required=True))
        self.subscribe_args(TfLoaderArgs(artifacts=False))
        self.subscribe_args(Tf2OnnxLoaderArgs())
        self.subscribe_args(OnnxShapeInferenceArgs())
        self.subscribe_args(OnnxLoaderArgs())
        self.subscribe_args(OnnxSaveArgs(output=False))
        self.subscribe_args(DataLoaderArgs())  # For int8 calibration
        self.subscribe_args(TrtConfigArgs())
        self.subscribe_args(TrtPluginLoaderArgs())
        self.subscribe_args(TrtNetworkLoaderArgs())
        self.subscribe_args(TrtEngineLoaderArgs())
        self.subscribe_args(TrtEngineSaveArgs(output=False))

    def add_parser_args(self, parser):
        parser.add_argument("-o", "--output", help="Path to save the converted model", required=True)
        parser.add_argument(
            "--convert-to",
            help="The format to attempt to convert the model to."
            "'onnx-like-trt-network' is EXPERIMETNAL and converts a TensorRT network to a format usable for visualization. "
            "See 'OnnxLikeFromNetwork' for details. ",
            choices=["onnx", "trt", "onnx-like-trt-network"],
        )

        onnx_args = self.arg_groups[OnnxLoaderArgs].group
        onnx_args.add_argument(
            "--fp-to-fp16",
            help="Convert all floating point tensors in an ONNX model to 16-bit precision. "
            "This is *not* needed in order to use TensorRT's fp16 precision, but may be useful for other backends. "
            "Requires onnxmltools. ",
            action="store_true",
            default=None,
        )

    def run(self, args):
        if not args.convert_to:
            _, ext = os.path.splitext(args.output)
            if ext not in ModelArgs.EXT_MODEL_TYPE_MAPPING:
                G_LOGGER.critical(
                    "Could not automatically determine model type based on output path: {:}\n"
                    "Please specify the desired output format with --convert-to".format(args.output)
                )
            convert_type = ModelArgs.ModelType(ModelArgs.EXT_MODEL_TYPE_MAPPING[ext])
        elif args.convert_to == "onnx-like-trt-network":
            convert_type = "onnx-like-trt-network"
        else:
            CONVERT_TO_MODEL_TYPE_MAPPING = {"onnx": "onnx", "trt": "engine"}
            convert_type = ModelArgs.ModelType(CONVERT_TO_MODEL_TYPE_MAPPING[args.convert_to])

        if convert_type == "onnx-like-trt-network":
            onnx_like = trt_backend.onnx_like_from_network(self.arg_groups[TrtNetworkLoaderArgs].get_network_loader())
            onnx_backend.save_onnx(onnx_like, args.output)
        elif convert_type.is_onnx():
            model = self.arg_groups[OnnxLoaderArgs].load_onnx()
            if args.fp_to_fp16:
                model = onnx_backend.convert_to_fp16(model)
            self.arg_groups[OnnxSaveArgs].save_onnx(model, args.output)
        elif convert_type.is_trt():
            with self.arg_groups[TrtEngineLoaderArgs].build_engine() as engine:
                self.arg_groups[TrtEngineSaveArgs].save_engine(engine, args.output)
        else:
            G_LOGGER.critical("Cannot convert to model type: {:}".format(convert_type))
