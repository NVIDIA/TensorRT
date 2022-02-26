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

from polygraphy import constants, mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import assert_identifier, inline, make_invocable, safe


@mod.export()
class TrtLegacyArgs(BaseArgs):
    def add_to_parser(self, parser):
        trt_legacy_args = parser.add_argument_group(
            "TensorRT Legacy",
            "[DEPRECATED] Options for TensorRT Legacy. Reuses TensorRT options, but does not support int8 mode, or dynamic shapes",
        )
        trt_legacy_args.add_argument(
            "-p", "--preprocessor", help="The preprocessor to use for the UFF converter", default=None
        )
        trt_legacy_args.add_argument("--uff-order", help="The order of the input", default=None)
        trt_legacy_args.add_argument(
            "--batch-size",
            metavar="SIZE",
            help="The batch size to use in TensorRT when it cannot be automatically determined",
            type=int,
            default=None,
        )
        trt_legacy_args.add_argument(
            "--model",
            help="Model file for Caffe models. The deploy file should be provided as the model_file positional argument",
            dest="caffe_model",
        )
        trt_legacy_args.add_argument(
            "--save-uff", help="Save intermediate UFF files", action="store_true", default=None
        )

    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.data_loader import DataLoaderArgs
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs
        from polygraphy.tools.args.tf.loader import TfLoaderArgs
        from polygraphy.tools.args.trt.config import TrtConfigArgs
        from polygraphy.tools.args.trt.loader import TrtEngineLoaderArgs, TrtEngineSaveArgs
        from polygraphy.tools.args.trt.runner import TrtRunnerArgs

        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, TfLoaderArgs):
            self.tf_loader_args = maker
        if isinstance(maker, TrtConfigArgs):
            self.trt_config_args = maker
        if isinstance(maker, TrtEngineLoaderArgs):
            self.trt_engine_loader_args = maker
        if isinstance(maker, TrtEngineSaveArgs):
            self.trt_engine_save_args = maker
        if isinstance(maker, TrtRunnerArgs):
            self.trt_runner_args = maker
        if isinstance(maker, DataLoaderArgs):
            self.data_loader_args = maker

    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_engine_loader_args is not None, "TrtEngineLoaderArgs is required!"

    def parse(self, args):
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
            calib_base = safe(assert_identifier(calib_base))
            self.calibration_base_class = inline(safe("trt.{:}", inline(calib_base)))

        self.quantile = args_util.get(args, "quantile")
        self.regression_cutoff = args_util.get(args, "regression_cutoff")

        self.use_dla = args_util.get(args, "use_dla")
        self.allow_gpu_fallback = args_util.get(args, "allow_gpu_fallback")

    def add_to_script(self, script):
        script.add_import(imports=["TrtLegacyRunner"], frm="polygraphy.backend.trt_legacy")
        G_LOGGER.warning("Legacy TensorRT runner only supports implicit batch TensorFlow/UFF, ONNX, and Caffe models")

        load_engine = self.model_args.model_file if self.model_args.model_type == "engine" else None

        loader_name = None
        if self.model_args.model_type == "onnx":
            script.add_import(imports=["ParseNetworkFromOnnxLegacy"], frm="polygraphy.backend.trt_legacy")
            onnx_loader = self.onnx_loader_args.add_onnx_loader(script, disable_custom_outputs=True)
            loader_name = script.add_loader(
                make_invocable("ParseNetworkFromOnnxLegacy", onnx_loader), "parse_network_from_onnx_legacy"
            )
        elif self.model_args.model_type == "caffe":
            script.add_import(imports=["LoadNetworkFromCaffe"], frm="polygraphy.backend.trt_legacy")
            loader_name = script.add_loader(
                make_invocable(
                    "LoadNetworkFromCaffe",
                    self.model_args.model_file,
                    self.caffe_model,
                    self.trt_outputs,
                    self.batch_size,
                ),
                "parse_network_from_caffe",
            )
        elif load_engine is None:
            script.add_import(imports=["LoadNetworkFromUff"], frm="polygraphy.backend.trt_legacy")
            if self.model_args.model_type == "uff":
                script.add_import(imports=["LoadUffFile"], frm="polygraphy.backend.trt_legacy")
                shapes = {name: shape for name, (_, shape) in self.model_args.input_shapes.items()}
                loader_name = script.add_loader(
                    make_invocable(
                        "LoadUffFile", self.model_args.model_file, util.default(shapes, {}), self.trt_outputs
                    ),
                    "load_uff_file",
                )
            else:
                script.add_import(imports=["ConvertToUff"], frm="polygraphy.backend.trt_legacy")
                loader_name = script.add_loader(
                    make_invocable(
                        "ConvertToUff",
                        self.tf_loader_args.add_to_script(script),
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
            self.trt_config_args.int8 and self.data_loader_args is not None
        ):  # We cannot do calibration if there is no data loader.
            script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
            script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
            data_loader_name = self.data_loader_args.add_data_loader(script)
            if self.calibration_base_class:
                script.add_import(imports=["tensorrt as trt"])

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
            max_workspace_size=self.trt_config_args.workspace,
            max_batch_size=self.batch_size,
            fp16=self.trt_config_args.fp16,
            tf32=self.trt_config_args.tf32,
            load_engine=load_engine,
            save_engine=self.trt_engine_save_args.path,
            layerwise=self.trt_outputs == constants.MARK_ALL,
            plugins=self.trt_engine_loader_args.plugins,
            int8=self.trt_config_args.int8,
            calibrator=calibrator,
            use_dla=self.use_dla,
            allow_gpu_fallback=self.allow_gpu_fallback,
        )

        script.add_runner(runner_str)
