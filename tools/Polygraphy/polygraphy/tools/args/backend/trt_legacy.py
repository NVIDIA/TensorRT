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

from polygraphy import constants, mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.onnx.loader import OnnxLoadArgs
from polygraphy.tools.args.backend.trt.config import TrtConfigArgs
from polygraphy.tools.args.backend.trt.loader import TrtLoadPluginsArgs, TrtSaveEngineArgs
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.args.comparator.data_loader import DataLoaderArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.script import inline_identifier, inline, make_invocable, safe


@mod.export()
class TrtLegacyRunnerArgs(BaseRunnerArgs):
    """
    TensorRT Legacy API (UFF, Caffe) Inference: inference with deprecated TensorRT APIs.

    Depends on:

        - ModelArgs
        - TrtLoadPluginsArgs
        - TrtConfigArgs
        - TfLoadArgs
        - TrtSaveEngineArgs
        - DataLoaderArgs
        - OnnxLoadArgs

    [DEPRECATED] Options related to inference using deprecated TensorRT APIs and import paths, like UFF and Caffe.
    """

    def get_name_opt_impl(self):
        return "Legacy TensorRT", "trt-legacy"

    def get_extra_help_text_impl(self):
        return "Only supports networks using implicit batch mode."

    def add_parser_args_impl(self):
        self.group.add_argument(
            "-p", "--preprocessor", help="The preprocessor to use for the UFF converter", default=None
        )
        self.group.add_argument("--uff-order", help="The order of the input", default=None)
        self.group.add_argument(
            "--batch-size",
            metavar="SIZE",
            help="The batch size to use in TensorRT when it cannot be automatically determined",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--model",
            help="Model file for Caffe models. The deploy file should be provided as the model_file positional argument",
            dest="caffe_model",
        )
        self.group.add_argument("--save-uff", help="Save intermediate UFF files", action="store_true", default=None)

    def parse_impl(self, args):
        self.trt_outputs = args_util.get_outputs(args, "trt_outputs")
        self.caffe_model = args_util.get(args, "caffe_model")
        self.batch_size = args_util.get(args, "batch_size")
        self.save_uff = args_util.get(args, "save_uff")
        self.uff_order = args_util.get(args, "uff_order")
        self.preprocessor = args_util.get(args, "preprocessor")

        self.calibration_cache = args_util.get(args, "calibration_cache")
        calib_base = args_util.get(args, "calibration_base_class")
        self.calibration_base_class = None
        if calib_base is not None:
            self.calibration_base_class = inline(safe("trt.{:}", inline_identifier(calib_base)))

        self.quantile = args_util.get(args, "quantile")
        self.regression_cutoff = args_util.get(args, "regression_cutoff")

        self.use_dla = args_util.get(args, "use_dla")
        self.allow_gpu_fallback = args_util.get(args, "allow_gpu_fallback")

    def add_to_script_impl(self, script):
        script.add_import(imports=["TrtLegacyRunner"], frm="polygraphy.backend.trt_legacy")
        G_LOGGER.warning("Legacy TensorRT runner only supports implicit batch TensorFlow/UFF, ONNX, and Caffe models")

        load_engine = self.arg_groups[ModelArgs].path if self.arg_groups[ModelArgs].model_type == "engine" else None

        loader_name = None
        if self.arg_groups[ModelArgs].model_type == "onnx":
            script.add_import(imports=["ParseNetworkFromOnnxLegacy"], frm="polygraphy.backend.trt_legacy")
            onnx_loader = self.arg_groups[OnnxLoadArgs].add_to_script(script, disable_custom_outputs=True)
            loader_name = script.add_loader(
                make_invocable("ParseNetworkFromOnnxLegacy", onnx_loader), "parse_network_from_onnx_legacy"
            )
        elif self.arg_groups[ModelArgs].model_type == "caffe":
            script.add_import(imports=["LoadNetworkFromCaffe"], frm="polygraphy.backend.trt_legacy")
            loader_name = script.add_loader(
                make_invocable(
                    "LoadNetworkFromCaffe",
                    self.arg_groups[ModelArgs].path,
                    self.caffe_model,
                    self.trt_outputs,
                    self.batch_size,
                ),
                "parse_network_from_caffe",
            )
        elif load_engine is None:
            script.add_import(imports=["LoadNetworkFromUff"], frm="polygraphy.backend.trt_legacy")
            if self.arg_groups[ModelArgs].model_type == "uff":
                script.add_import(imports=["LoadUffFile"], frm="polygraphy.backend.trt_legacy")
                shapes = {name: shape for name, (_, shape) in self.arg_groups[ModelArgs].input_shapes.items()}
                loader_name = script.add_loader(
                    make_invocable(
                        "LoadUffFile", self.arg_groups[ModelArgs].path, util.default(shapes, {}), self.trt_outputs
                    ),
                    "load_uff_file",
                )
            else:
                from polygraphy.tools.args.backend.tf.loader import TfLoadArgs

                script.add_import(imports=["ConvertToUff"], frm="polygraphy.backend.trt_legacy")
                loader_name = script.add_loader(
                    make_invocable(
                        "ConvertToUff",
                        self.arg_groups[TfLoadArgs].add_to_script(script),
                        save_uff=self.save_uff,
                        preprocessor=self.preprocessor,
                    ),
                    "convert_to_uff",
                )
            loader_name = script.add_loader(
                make_invocable("LoadNetworkFromUff", loader_name, uff_order=self.uff_order), "uff_network_loader"
            )

        calibrator = None
        if (
            self.arg_groups[TrtConfigArgs].int8 and DataLoaderArgs in self.arg_groups
        ):  # We cannot do calibration if there is no data loader.
            script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
            script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
            data_loader_name = self.arg_groups[DataLoaderArgs].add_to_script(script)
            if self.calibration_base_class:
                script.add_import(imports="tensorrt", imp_as="trt")

            calibrator = make_invocable(
                "Calibrator",
                data_loader=data_loader_name if data_loader_name else inline(safe("DataLoader()")),
                cache=self.calibration_cache,
                BaseClass=self.calibration_base_class,
                quantile=self.quantile,
                regression_cutoff=self.regression_cutoff,
            )

        runner_str = make_invocable(
            "TrtLegacyRunner",
            network_loader=loader_name,
            max_workspace_size=self.arg_groups[TrtConfigArgs]._workspace,
            max_batch_size=self.batch_size,
            fp16=self.arg_groups[TrtConfigArgs].fp16,
            tf32=self.arg_groups[TrtConfigArgs].tf32,
            load_engine=load_engine,
            save_engine=self.arg_groups[TrtSaveEngineArgs].path,
            layerwise=self.trt_outputs == constants.MARK_ALL,
            plugins=self.arg_groups[TrtLoadPluginsArgs].plugins,
            int8=self.arg_groups[TrtConfigArgs].int8,
            fp8=self.arg_groups[TrtConfigArgs].fp8,
            calibrator=calibrator,
            use_dla=self.use_dla,
            allow_gpu_fallback=self.allow_gpu_fallback,
        )

        script.add_runner(runner_str)
