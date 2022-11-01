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

from polygraphy import constants, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.comparator import IterationResult
from polygraphy.comparator.data_loader import DataLoaderCache
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.comparator.data_loader import DataLoaderArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.script import Script, make_invocable, make_invocable_if_nondefault_kwargs

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnxrt_backend = mod.lazy_import("polygraphy.backend.onnxrt")


@mod.export()
class OnnxInferShapesArgs(BaseArgs):
    """
    ONNX Shape Inference: ONNX shape inference.

    Depends on:

        - OnnxLoadArgs
        - DataLoaderArgs: if allow_force_fallback == True
    """

    def __init__(self, default: bool = None, allow_force_fallback: bool = None):
        """
        Args:
            default (bool):
                    Whether shape inference should be enabled by default.
                    Defaults to False.
            allow_force_fallback (bool):
                    Whether fallback shape inference using ONNX-Runtime should be allowed.
                    Defaults to False.
        """
        super().__init__()
        self._default = util.default(default, False)
        self._allow_force_fallback = util.default(allow_force_fallback, False)

    def add_parser_args_impl(self):
        shape_infer_group = self.group.add_mutually_exclusive_group()
        if self._default:
            shape_infer_group.add_argument(
                "--no-shape-inference",
                help="Disable ONNX shape inference when loading the model",
                dest="do_shape_inference",
                action="store_false",
                default=True,
            )
        else:
            shape_infer_group.add_argument(
                "--shape-inference",
                "--do-shape-inference",
                help="Enable ONNX shape inference when loading the model",
                dest="do_shape_inference",
                action="store_true",
                default=False,
            )

        if self._allow_force_fallback:
            shape_infer_group.add_argument(
                "--force-fallback-shape-inference",
                help="Force Polygraphy to use ONNX-Runtime to determine metadata for "
                "tensors in the graph. This can be useful in cases where ONNX shape inference does not generate correct information. "
                "Note that this will cause dynamic dimensions to become static. ",
                action="store_true",
                default=None,
            )

        self.group.add_argument(
            "--no-onnxruntime-shape-inference",
            help="Disable using ONNX-Runtime's shape inference utilities. "
            "This will force Polygraphy to use `onnx.shape_inference` instead. "
            "Note that ONNX-Runtime's shape inference utilities may be more performant and memory-efficient. ",
            dest="allow_onnxruntime",
            action="store_false",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            do_shape_inference (bool): Whether to do shape inference.
            force_fallback (bool): Whether to force fallback shape inference.
            allow_onnxruntime (bool): Whether to allow ONNX-Runtime's shape inference utilities to be used.
        """
        self.do_shape_inference = args_util.get(args, "do_shape_inference")
        self.force_fallback = args_util.get(args, "force_fallback_shape_inference")
        self.allow_onnxruntime = args_util.get(args, "allow_onnxruntime")

        # No point is running ONNX shape inference if we're going to use fallback inference.
        if self.force_fallback:
            self.do_shape_inference = False

    def add_to_script_impl(self, script, loader_name):
        """
        Note that this method does not take fallback shape inference into account.
        To support fallback shape inference, the tool must call `fallback_inference()` manually.

        Args:
            loader_name (str):
                    The name of the loader which should be consumed by the ``InferShapes`` loader.

        Returns:
            str: The name of the ``InferShapes`` loader added to the script.
        """
        if self.do_shape_inference:
            script.add_import(imports=["InferShapes"], frm="polygraphy.backend.onnx")
            loader_name = script.add_loader(
                make_invocable(
                    "InferShapes",
                    loader_name,
                    external_data_dir=self.arg_groups[OnnxLoadArgs].external_data_dir,
                    allow_onnxruntime=self.allow_onnxruntime,
                ),
                "infer_shapes",
            )
        return loader_name

    def infer_shapes(self, model, force=None):
        """
        Run shape inference on an ONNX model if `do_shape_inference` is True
        according to arguments provided on the command-line.

        Args:
            model (onnx.ModelProto): The model in which to infer shapes.
            force (bool):
                    Force shape inference to run even if `do_shape_inference` is False.
                    Defaults to False.

        Returns:
            onnx.ModelProto: The model with shapes inferred.
        """
        force = util.default(force, False)
        with util.TempAttrChange(self, {"do_shape_inference": True if force else self.do_shape_inference}):
            loader = args_util.run_script(self.add_to_script, model)
            return util.invoke_if_callable(loader)[0]

    def fallback_inference(self, onnx_model, outputs=None):
        """
        Run inference with ONNX-Runtime.

        This can be used to retrieve values/shapes/data types for all
        tensors in the model when other shape inference approaches fail.

        Args:
            onnx_model (onnx.ModelProto):
                    The ONNX model in which to infer shapes.


            outputs (List[str]):
                    The names of the outputs to retrieved.
                    Defaults to constants.MARK_ALL

        Returns:
            (IterationResult, TensorMetadata):
                    A tuple containing two elements:
                    1. Mapping of values for all tensors in the model, including inputs.
                    2. Metadata for every tensor in the model.
        """
        outputs = util.default(outputs, constants.MARK_ALL)
        with G_LOGGER.verbosity(G_LOGGER.module_severity.get(G_LOGGER.module_path(__file__)) + 10):
            load_model = onnx_backend.ModifyOutputs(onnx_model, outputs=outputs, copy=True)
            with onnxrt_backend.OnnxrtRunner(
                onnxrt_backend.SessionFromOnnx(onnx_backend.BytesFromOnnx(load_model))
            ) as runner:
                data_loader = self.arg_groups[DataLoaderArgs].get_data_loader()
                loader_cache = DataLoaderCache(data_loader)
                loader_cache.set_input_metadata(runner.get_input_metadata())

                feed_dict = loader_cache[0]

                with G_LOGGER.verbosity(G_LOGGER.module_severity.get(G_LOGGER.module_path(__file__)) - 10):
                    G_LOGGER.info(
                        f"Running fallback shape inference using input metadata:\n{TensorMetadata.from_feed_dict(feed_dict)}"
                    )

                outputs = runner.infer(feed_dict)
                # We include the inputs here so that we have values for all tensors in the model.
                outputs.update(feed_dict)
                # Use IterationResult here since it can handle very large tensors by saving to disk.
                # Layerwise outputs might otherwise take up too much memory.
                return IterationResult(outputs), TensorMetadata.from_feed_dict(outputs)


@mod.export()
class OnnxSaveArgs(BaseArgs):
    """
    ONNX Model Saving: saving ONNX models.

    Depends on:

        - OnnxInferShapesArgs: if allow_shape_inference == True
    """

    def __init__(
        self,
        allow_shape_inference: bool = None,
        output_opt: str = None,
        output_short_opt: str = None,
        output_opt_required: bool = None,
        output_default_path: str = None,
        allow_multiple_models: bool = None,
    ):
        """
        Args:
            allow_shape_inference (bool):
                    Whether to allow shape inference when saving models.
                    Defaults to False.
            output_opt (str):
                    The name of the output path option.
                    Defaults to "output".
                    Use a value of ``False`` to disable the option.
            output_short_opt (str):
                    The short option to use for the output path.
                    Defaults to "-o".
                    Use a value of ``False`` to disable the short option.
            output_opt_required (bool):
                    Whether the output path is a required argument.
                    Defaults to False.
            output_default_path (str):
                    The default value to use for the output path option.
                    Defaults to None.
            allow_multiple_models (bool):
                    Whether to enable support for saving more than one model.
                    If this is True, the output path is expected to be a directory.
                    Defaults to False.
        """
        super().__init__()

        self._allow_shape_inference = util.default(allow_shape_inference, False)
        self._output_opt = util.default(output_opt, "output")
        self._output_short_opt = util.default(output_short_opt, "-o")
        self._output_opt_required = util.default(output_opt_required, False)
        self._output_default_path = output_default_path
        self._allow_multiple_models = util.default(allow_multiple_models, False)

        # add_to_script should never be called when `allow_multiple_models` is enabled.
        # The one exception is that `save_onnx` should be able to call it, which is why we need this escape hatch.
        self._disable_add_to_script_check = False

    def add_parser_args_impl(self):
        if self._output_opt:
            params = ([self._output_short_opt] if self._output_short_opt else []) + [f"--{self._output_opt}"]
            help_msg = "Path to save the ONNX model"
            if self._allow_multiple_models:
                help_msg = "Path to a directory in which to save ONNX model(s)"

            self.group.add_argument(
                *params,
                help=help_msg,
                dest="save_onnx",
                default=self._output_default_path,
                required=self._output_opt_required,
            )

        self.group.add_argument(
            "--save-external-data",
            "--external-data-path",
            help="Whether to save weight data in external file(s). "
            + (
                "You may optionally provide a value to this argument which will be used as a suffix for the external data files"
                if self._allow_multiple_models
                else "To use a non-default path, supply the desired path as an argument. This is always a relative path; "
                "external data is always written to the same directory as the model. "
            ),
            default=None,
            action="append",
            nargs="?",
            dest="external_data_path",
        )
        self.group.add_argument(
            "--external-data-size-threshold",
            help="The size threshold, in bytes, above which tensor data will be stored in the external file. "
            "Tensors smaller that this threshold will remain in the ONNX file. "
            "Optionally, use a `K`, `M`, or `G` suffix to indicate KiB, MiB, or GiB respectively. "
            "For example, `--external-data-size-threshold=16M` is equivalent to `--external-data-size-threshold=16777216`. "
            "Has no effect if `--save-external-data` is not set. Defaults to 1024 bytes.",
            default=None,
        )
        self.group.add_argument(
            "--no-save-all-tensors-to-one-file",
            help="Do not save all tensors to a single file when saving external data. "
            "Has no effect if `--save-external-data` is not set",
            dest="all_tensors_to_one_file",
            default=None,
            action="store_false",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            path (str): The path at which to save the ONNX model.
            external_data_path (str): The path at which to save external data.
            size_threshold (int): The size threshold above which external data is saved.
            all_tensors_to_one_file (bool): Whether all external data should be written to a single file.
        """
        self.path = args_util.get(args, "save_onnx")

        external_data_path = args_util.get(args, "external_data_path")
        if external_data_path is not None:
            external_data_path = external_data_path[0] or ""
        self.external_data_path = external_data_path

        self.size_threshold = args_util.parse_num_bytes(args_util.get(args, "external_data_size_threshold"))
        self.all_tensors_to_one_file = args_util.get(args, "all_tensors_to_one_file")

    def add_to_script_impl(self, script, loader_name):
        """
        Args:
            loader_name (str):
                    The name of the loader which should be consumed by the ``SaveOnnx`` loader.

        Returns:
            str: The name of the ``SaveOnnx`` loader added to the script.
        """
        if self._allow_multiple_models and not self._disable_add_to_script_check:
            G_LOGGER.internal_error(
                "OnnxSaveArgs.add_to_script() should never be called when `allow_multiple_models` is enabled"
            )

        if self.path is None:
            return loader_name

        # Need to run shape inference again after processing the graph since it may have changed.
        if self._allow_shape_inference:
            loader_name = self.arg_groups[OnnxInferShapesArgs].add_to_script(script, loader_name)

        script.add_import(imports=["SaveOnnx"], frm="polygraphy.backend.onnx")
        loader_name = script.add_loader(
            make_invocable(
                "SaveOnnx",
                loader_name,
                path=self.path,
                external_data_path=self.external_data_path,
                size_threshold=self.size_threshold,
                all_tensors_to_one_file=self.all_tensors_to_one_file,
            ),
            "save_onnx",
        )

        return loader_name

    def save_onnx(self, model, path: str = None):
        """
        Saves an ONNX model according to arguments provided on the command-line.

        Args:
            model (onnx.ModelProto): The ONNX model to save.

            path (str):
                    The path at which to save the model.
                    If no path is provided, it is determined from command-line arguments.

        Returns:
            onnx.ModelProto: The model that was saved.
        """
        attrs = {"path": path, "_disable_add_to_script_check": True}
        if self._allow_multiple_models:
            if self.external_data_path is not None:
                attrs["external_data_path"] = os.path.basename(os.path.splitext(path)[0]) + (
                    self.external_data_path or "_ext_data"
                )

        with util.TempAttrChange(self, attrs):
            loader = args_util.run_script(self.add_to_script, model)
            return loader()


@mod.export()
class OnnxLoadArgs(BaseArgs):
    """
    ONNX Model Loading: loading ONNX models.

    Depends on:

        - ModelArgs
        - OnnxInferShapesArgs: if allow_shape_inference == True
        - OnnxSaveArgs: if allow_saving == True
        - OnnxFromTfArgs: if allow_from_tf == True
    """

    def __init__(
        self,
        allow_saving: bool = None,
        outputs_opt_prefix: str = None,
        allow_shape_inference: bool = None,
        allow_from_tf: bool = None,
    ):
        """
        Args:
            allow_saving (bool):
                    Whether to allow loaded models to be saved.
                    Defaults to False.
            outputs_opt_prefix (str):
                    The prefix to use for the outputs option, which controls which tensors are marked as outputs.
                    Defaults to "onnx-".
                    Use a value of ``False`` to disable the option.
            allow_shape_inference (bool):
                    Whether to allow shape inference when loading models.
                    Defaults to True.
            allow_from_tf (bool):
                    Whether to allow conversion of TensorFlow models to ONNX.
                    Defaults to False.
        """
        super().__init__()
        self._allow_saving = util.default(allow_saving, False)
        self._allow_shape_inference = util.default(allow_shape_inference, True)
        self._outputs_opt_prefix = util.default(outputs_opt_prefix, "onnx-")
        self._allow_from_tf = util.default(allow_from_tf, False)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--external-data-dir",
            "--load-external-data",
            "--ext",
            dest="external_data_dir",
            help="Path to a directory containing external data for the model. "
            "Generally, this is only required if the external data is not stored in the model directory.",
        )
        self.group.add_argument(
            "--ignore-external-data",
            help="Ignore external data and just load the model structure without any weights. "
            "The model will be usable only for purposes that don't require weights, such as extracting "
            "subgraphs or inspecting model structure. "
            "This can be useful in cases where external data is not available.",
            action="store_true",
            default=None,
        )

        if self._outputs_opt_prefix is not False:  # Empty strings should not disable the option
            self.group.add_argument(
                f"--{self._outputs_opt_prefix}outputs",
                help="Name(s) of ONNX tensor(s) to mark as output(s). "
                "Using the special value 'mark all' indicates that all tensors should be used as outputs",
                nargs="+",
                default=None,
                dest="onnx_outputs",
            )
            self.group.add_argument(
                f"--{self._outputs_opt_prefix}exclude-outputs",
                help="[EXPERIMENTAL] Name(s) of ONNX output(s) to unmark as outputs.",
                nargs="+",
                default=None,
                dest="onnx_exclude_outputs",
            )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            outputs (List[str]): Names of output tensors.
            exclude_outputs (List[str]): Names of tensors which should be unmarked as outputs.
            external_data_dir (str): Path to a directory from which to load external data.
            ignore_external_data (bool): Whether to ignore loading external data.
        """
        self.outputs = args_util.get_outputs(args, "onnx_outputs")
        self.exclude_outputs = args_util.get(args, "onnx_exclude_outputs")
        self.external_data_dir = args_util.get(args, "external_data_dir")
        self.ignore_external_data = args_util.get(args, "ignore_external_data")

    def _add_modify_onnx_outputs(self, script, loader_name, disable_custom_outputs: bool = None):
        if disable_custom_outputs:
            outputs = None
            exclude_outputs = None
        else:
            outputs = args_util.get_outputs_for_script(script, self.outputs)
            exclude_outputs = self.exclude_outputs

        modify_outputs_loader = make_invocable_if_nondefault_kwargs(
            "ModifyOnnxOutputs", loader_name, outputs=outputs, exclude_outputs=exclude_outputs
        )
        if modify_outputs_loader is not None:
            script.add_import(imports="ModifyOutputs", frm="polygraphy.backend.onnx", imp_as="ModifyOnnxOutputs")
            loader_name = script.add_loader(
                modify_outputs_loader,
                "modify_outputs",
            )

        return loader_name

    def add_to_script_impl(self, script, disable_custom_outputs: bool = None, serialize_model: bool = None):
        """
        Args:
            disable_custom_outputs (bool):
                    Whether to disallow modifying outputs according to the `outputs` and `exclude_outputs` attributes.
                    Defaults to False.
            serialize_model (bool):
                    Whether to serialize the model.
                    Defaults to False.

        Returns:
            str: The name of the ONNX loader added in the script.
        """
        model_type = self.arg_groups[ModelArgs].model_type
        if model_type.is_onnx():
            loader_name = self.arg_groups[ModelArgs].path
            if self._allow_shape_inference:
                loader_name = self.arg_groups[OnnxInferShapesArgs].add_to_script(script, loader_name)

            if loader_name == self.arg_groups[ModelArgs].path:  # Shape inference loader isn't being used, have to load.
                script.add_import(imports=["OnnxFromPath"], frm="polygraphy.backend.onnx")
                loader_str = make_invocable(
                    "OnnxFromPath",
                    self.arg_groups[ModelArgs].path,
                    external_data_dir=self.external_data_dir,
                    ignore_external_data=self.ignore_external_data,
                )
                loader_name = script.add_loader(loader_str, "load_onnx")
        elif model_type.is_tf() and self._allow_from_tf:
            from polygraphy.tools.args.backend.onnx.loader import OnnxFromTfArgs

            loader_name = self.arg_groups[OnnxFromTfArgs].add_to_script(script)
        else:
            G_LOGGER.critical(f"Model type: {model_type} could not be converted to an ONNX model.")

        loader_name = self._add_modify_onnx_outputs(script, loader_name, disable_custom_outputs=disable_custom_outputs)

        if self._allow_saving:
            loader_name = self.arg_groups[OnnxSaveArgs].add_to_script(script, loader_name)

        if serialize_model:
            script.add_import(imports=["BytesFromOnnx"], frm="polygraphy.backend.onnx")
            loader_name = script.add_loader(make_invocable("BytesFromOnnx", loader_name), "serialize_onnx")

        return loader_name

    def must_use_onnx_loader(self, disable_custom_outputs: bool = None):
        """
        Whether this model needs to be loaded via a Polygraphy ONNX loader, e.g., in case it
        needs modifications.

        Args:
            disable_custom_outputs (bool):
                    Whether to disallow modifying outputs according to the `outputs` and `exclude_outputs` attributes.

        Returns:
            bool
        """
        tmp_script = Script()
        inp_loader = "check_needs_modify"
        needs_modify = self._add_modify_onnx_outputs(tmp_script, inp_loader, disable_custom_outputs) != inp_loader
        needs_shape_inference = self._allow_shape_inference and self.arg_groups[OnnxInferShapesArgs].do_shape_inference
        needs_save = self._allow_saving and self.arg_groups[OnnxSaveArgs].path is not None
        # Currently, other loaders do not support external data, so we must fall back to the ONNX loader if it's present.
        return (
            not self.arg_groups[ModelArgs].model_type.is_onnx()
            or needs_modify
            or self.external_data_dir
            or needs_shape_inference
            or needs_save
        )

    def load_onnx(self):
        """
        Loads an ONNX model according to arguments provided on the command-line.

        Returns:
            onnx.ModelProto: The model that was loaded.
        """
        loader = args_util.run_script(self.add_to_script)
        return loader()


@mod.export()
class OnnxFromTfArgs(BaseArgs):
    """
    TensorFlow-ONNX Model Conversion: converting TensorFlow models to ONNX.

    Depends on:

        - TfLoadArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument("--opset", help="Opset to use when converting to ONNX", default=None, type=int)

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            opset (int): The ONNX opset version to use during conversion.
        """
        self.opset = args_util.get(args, "opset")

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.backend.tf.loader import TfLoadArgs

        G_LOGGER.verbose(
            "Attempting to load as a TensorFlow model, using TF2ONNX to convert to ONNX. "
            "If this is not correct, please specify --model-type",
            mode=LogMode.ONCE,
        )
        script.add_import(imports=["OnnxFromTfGraph"], frm="polygraphy.backend.onnx")
        loader_str = make_invocable(
            "OnnxFromTfGraph",
            self.arg_groups[TfLoadArgs].add_to_script(script, disable_custom_outputs=True),
            opset=self.opset,
        )
        loader_name = script.add_loader(loader_str, "export_onnx_from_tf")
        return loader_name
