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

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    DataLoaderArgs,
    ModelArgs,
    OnnxFromTfArgs,
    OnnxInferShapesArgs,
    OnnxLoadArgs,
    OnnxSaveArgs,
    TfLoadArgs,
    TrtConfigArgs,
    TrtLoadEngineArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtSaveEngineArgs,
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

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True),
            TfLoadArgs(allow_artifacts=False),
            OnnxFromTfArgs(),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(allow_from_tf=True),
            OnnxSaveArgs(output_opt=False),
            DataLoaderArgs(),  # For int8 calibration
            TrtConfigArgs(allow_engine_capability=True, allow_tensor_formats=True),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(allow_tensor_formats=True),
            TrtLoadEngineArgs(),
            TrtSaveEngineArgs(output_opt=False),
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument("-o", "--output", help="Path to save the converted model", required=True)
        parser.add_argument(
            "--convert-to",
            help="The format to attempt to convert the model to."
            "'onnx-like-trt-network' is EXPERIMETNAL and converts a TensorRT network to a format usable for visualization. "
            "See 'OnnxLikeFromNetwork' for details. ",
            choices=["onnx", "trt", "onnx-like-trt-network"],
        )

        onnx_args = self.arg_groups[OnnxLoadArgs].group
        onnx_args.add_argument(
            "--fp-to-fp16",
            help="Convert all floating point tensors in an ONNX model to 16-bit precision. "
            "This is *not* needed in order to use TensorRT's fp16 precision, but may be useful for other backends. "
            "Requires onnxmltools. ",
            action="store_true",
            default=None,
        )

    def run_impl(self, args):
        if not args.convert_to:
            _, ext = os.path.splitext(args.output)
            if ext not in ModelArgs.EXT_MODEL_TYPE_MAPPING:
                G_LOGGER.critical(
                    f"Could not automatically determine model type based on output path: {args.output}\nPlease specify the desired output format with --convert-to"
                )
            convert_type = ModelArgs.ModelType(ModelArgs.EXT_MODEL_TYPE_MAPPING[ext])
        elif args.convert_to == "onnx-like-trt-network":
            convert_type = "onnx-like-trt-network"
        else:
            CONVERT_TO_MODEL_TYPE_MAPPING = {"onnx": "onnx", "trt": "engine"}
            convert_type = ModelArgs.ModelType(CONVERT_TO_MODEL_TYPE_MAPPING[args.convert_to])

        if convert_type == "onnx-like-trt-network":
            onnx_like = trt_backend.onnx_like_from_network(self.arg_groups[TrtLoadNetworkArgs].load_network())
            onnx_backend.save_onnx(onnx_like, args.output)
        elif convert_type.is_onnx():
            model = self.arg_groups[OnnxLoadArgs].load_onnx()
            if args.fp_to_fp16:
                model = onnx_backend.convert_to_fp16(model)
            self.arg_groups[OnnxSaveArgs].save_onnx(model, args.output)
        elif convert_type.is_trt():
            with self.arg_groups[TrtLoadEngineArgs].load_engine() as engine:
                self.arg_groups[TrtSaveEngineArgs].save_engine(engine, args.output)
        else:
            G_LOGGER.critical(f"Cannot convert to model type: {convert_type}")
