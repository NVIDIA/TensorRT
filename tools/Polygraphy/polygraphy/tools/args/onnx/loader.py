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
from polygraphy import constants, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import Script, make_invocable

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnxrt_backend = mod.lazy_import("polygraphy.backend.onnxrt")


@mod.export()
class OnnxSaveArgs(BaseArgs):
    def __init__(
        self,
        infer_shapes=False,
        output="output",
        short_opt="-o",
        required=False,
        allow_ext_data_path=True,
        custom_help=None,
        default_output_path=None,
    ):
        super().__init__()
        self._infer_shapes = infer_shapes
        self._output = output
        self._short_opt = short_opt
        self._required = required
        self._allow_ext_data_path = allow_ext_data_path
        self._custom_help = custom_help
        self._default_output_path = default_output_path
        self.onnx_shape_inference_args = None

    def register(self, maker):
        if self._infer_shapes and isinstance(maker, OnnxShapeInferenceArgs):
            self.onnx_shape_inference_args = maker

    def add_to_parser(self, parser):
        self.group = parser.add_argument_group("ONNX Save Options", "Options for saving ONNX models")
        if self._output:
            flag = "--{:}".format(self._output)
            short = self._short_opt or flag
            self.group.add_argument(
                short,
                flag,
                help=self._custom_help or "Path to save the ONNX model",
                dest="save_onnx",
                default=self._default_output_path,
                required=self._required,
            )

        if self._allow_ext_data_path:
            ext_data_params = {
                "action": "append",
                "nargs": "?",
            }
        else:
            ext_data_params = {
                "action": "append_const",
                "const": "",
            }

        self.group.add_argument(
            "--save-external-data",
            help="Whether to save weight data in external file(s). "
            + (
                "To use a non-default path, supply the desired path as an argument. This is always a relative path; "
                "external data is always written to the same directory as the model. "
                if self._allow_ext_data_path
                else ""
            ),
            default=None,
            **ext_data_params,
        )
        self.group.add_argument(
            "--external-data-size-threshold",
            help="The size threshold, in bytes, above which tensor data will be stored in the external file. "
            "Tensors smaller that this threshold will remain in the ONNX file. "
            "Optionally, use a `K`, `M`, or `G` suffix to indicate KiB, MiB, or GiB respectively."
            "For example, `--external-data-size-threshold=16M` is equivalent to `--external-data-size-threshold=16777216`"
            "Has no effect if `--save-external-data` is not set",
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

    def parse(self, args):
        self.path = args_util.get(args, "save_onnx")
        save_external_data = args_util.get(args, "save_external_data")
        if save_external_data is not None:
            save_external_data = save_external_data[0] or ""
        self.save_external_data = save_external_data

        self.size_threshold = args_util.parse_num_bytes(args_util.get(args, "external_data_size_threshold"))
        self.all_tensors_to_one_file = args_util.get(args, "all_tensors_to_one_file")

    def add_save_onnx(self, script, loader_name):
        if self.path is None:
            return loader_name

        # Need to run shape inference again after processing the graph since it may have changed.
        if self.onnx_shape_inference_args is not None:
            loader_name = self.onnx_shape_inference_args.add_to_script(script, loader_name)

        script.add_import(imports=["SaveOnnx"], frm="polygraphy.backend.onnx")
        loader_name = script.add_loader(
            make_invocable(
                "SaveOnnx",
                loader_name,
                path=self.path,
                external_data_path=self.save_external_data,
                size_threshold=self.size_threshold,
                all_tensors_to_one_file=self.all_tensors_to_one_file,
            ),
            "save_onnx",
        )

        return loader_name

    def save_onnx(self, model, path=None):
        with util.TempAttrChange(self, "path", path):
            loader = args_util.run_script(self.add_save_onnx, model)
            return loader()


@mod.export()
class OnnxShapeInferenceArgs(BaseArgs):
    # NOTE: force_fallback is not implemented under add_to_script, and must be implemented
    # manually in tools that use this group.
    def __init__(self, default=False, enable_force_fallback=False):
        super().__init__()
        self._default = default
        self._enable_force_fallback = enable_force_fallback
        self.onnx_loader_args = None

    def add_to_parser(self, parser):
        self.group = parser.add_argument_group("ONNX Shape Inference", "Options for ONNX Shape Inference")
        g = self.group.add_mutually_exclusive_group()

        if self._default:
            g.add_argument(
                "--no-shape-inference",
                help="Disable ONNX shape inference when loading the model",
                dest="do_shape_inference",
                action="store_false",
                default=True,
            )
        else:
            g.add_argument(
                "--shape-inference",
                help="Enable ONNX shape inference when loading the model",
                dest="do_shape_inference",
                action="store_true",
                default=False,
            )

        if self._enable_force_fallback:
            g.add_argument(
                "--force-fallback-shape-inference",
                help="Force Polygraphy to use ONNX-Runtime to determine metadata for "
                "tensors in the graph. This can be useful in cases where ONNX shape inference does not generate correct information. "
                "Note that this will cause dynamic dimensions to become fixed. ",
                action="store_true",
                default=None,
            )

    def register(self, maker):
        from polygraphy.tools.args.data_loader import DataLoaderArgs

        if isinstance(maker, DataLoaderArgs):
            self.data_loader_args = maker
        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker

    def check_registered(self):
        assert (
            not self._enable_force_fallback or self.data_loader_args
        ), "DataLoaderArgs is required if force fallback shape inference is enabled!"

    def parse(self, args):
        self.do_shape_inference = args_util.get(args, "do_shape_inference")
        self.force_fallback = args_util.get(args, "force_fallback_shape_inference")

        # No point is running ONNX shape inference if we're going to use fallback inference.
        if self.force_fallback:
            self.do_shape_inference = False

    def add_to_script(self, script, loader_name):
        if self.do_shape_inference:
            script.add_import(imports=["InferShapes"], frm="polygraphy.backend.onnx")
            external_data_dir = self.onnx_loader_args.load_external_data if self.onnx_loader_args is not None else None
            loader_name = script.add_loader(
                make_invocable("InferShapes", loader_name, external_data_dir=external_data_dir), "infer_shapes"
            )
        return loader_name

    def fallback_inference(self, onnx_model):
        """
        Run inference with ONNX-Runtime.

        This can be used to retrieve values/shapes/data types for all
        tensors in the model when other shape inference approaches fail.

        Args:
            onnx_model (onnx.ModelProto):
                    The ONNX model in which to infer shapes.
            data_loader_args (DataLoaderArgs):
                    The data loader argument group to use to generate input data.

        Returns:
            (OrderedDict[str, np.ndarray], TensorMetadata):
                    1. Mapping of values for all tensors in the model, including inputs.
                        Values are loaded lazily when first accessed so as to save memory.
                    2. Metadata for every tensor in the model.
        """
        from polygraphy.comparator import IterationResult

        with G_LOGGER.verbosity(G_LOGGER.severity + 10):
            load_model = onnx_backend.ModifyOutputs(onnx_model, outputs=constants.MARK_ALL, copy=True)
            with onnxrt_backend.OnnxrtRunner(
                onnxrt_backend.SessionFromOnnx(onnx_backend.BytesFromOnnx(load_model))
            ) as runner:
                # We want to set input_metadata only - not user_input_metadata, so that user_input_metadata
                # will be populated by the --model-inputs argument.
                data_loader = self.data_loader_args.get_data_loader()
                data_loader.input_metadata = runner.get_input_metadata()
                feed_dict = data_loader[0]

                with G_LOGGER.verbosity(G_LOGGER.severity - 10):
                    G_LOGGER.info(
                        "Running fallback shape inference using input metadata:\n{:}".format(
                            TensorMetadata.from_feed_dict(feed_dict)
                        )
                    )

                outputs = runner.infer(feed_dict)
                # We include the inputs here so that we have values for all tensors in the model.
                outputs.update(feed_dict)
                # Use IterationResult here since it can handle very large tensors by saving to disk.
                # Layerwise outputs might otherwise take up too much memory.
                return IterationResult(outputs), TensorMetadata.from_feed_dict(outputs)


@mod.export()
class OnnxLoaderArgs(BaseArgs):
    def __init__(self, save=False, output_prefix="onnx-"):
        super().__init__()
        self.tf2onnx_loader_args = None
        self.onnx_save_args = None
        self.onnx_shape_inference_args = None

        self._save = save
        self._output_prefix = output_prefix

    def add_to_parser(self, parser):
        self.group = parser.add_argument_group("ONNX Loader", "Options for the ONNX Loader")
        self.group.add_argument(
            "--external-data-dir",
            "--load-external-data",
            "--ext",
            dest="load_external_data",
            help="Path to a directory containing external data for the model. "
            "Generally, this is only required if the external data is not stored in the model directory.",
        )

        if self._output_prefix is not None:
            self.group.add_argument(
                "--{:}outputs".format(self._output_prefix),
                help="Name(s) of ONNX tensor(s) to mark as output(s). "
                "Using the special value 'mark all' indicates that all tensors should be used as outputs",
                nargs="+",
                default=None,
                dest="onnx_outputs",
            )
            self.group.add_argument(
                "--{:}exclude-outputs".format(self._output_prefix),
                help="[EXPERIMENTAL] Name(s) of ONNX output(s) to unmark as outputs.",
                nargs="+",
                default=None,
                dest="onnx_exclude_outputs",
            )

    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.tf2onnx.loader import Tf2OnnxLoaderArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, Tf2OnnxLoaderArgs):
            self.tf2onnx_loader_args = maker
        if self._save and isinstance(maker, OnnxSaveArgs):
            self.onnx_save_args = maker
        if isinstance(maker, OnnxShapeInferenceArgs):
            self.onnx_shape_inference_args = maker

    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert not self._save or self.onnx_save_args is not None, "OnnxSaveArgs is required to use save=True"

    def parse(self, args):
        self.outputs = args_util.get_outputs(args, "onnx_outputs")
        self.exclude_outputs = args_util.get(args, "onnx_exclude_outputs")
        self.load_external_data = args_util.get(args, "load_external_data")

    def _get_modify_onnx_loader(self, script, loader_name, disable_custom_outputs=None):
        if disable_custom_outputs:
            outputs = None
            exclude_outputs = None
        else:
            outputs = args_util.get_outputs_for_script(script, self.outputs)
            exclude_outputs = self.exclude_outputs

        if outputs or exclude_outputs:
            script.add_import(imports=["ModifyOutputs as ModifyOnnxOutputs"], frm="polygraphy.backend.onnx")
            loader_name = script.add_loader(
                make_invocable("ModifyOnnxOutputs", loader_name, outputs=outputs, exclude_outputs=exclude_outputs),
                "modify_outputs",
            )

        return loader_name

    def add_onnx_loader(self, script, disable_custom_outputs=None, suffix=None):
        model_type = self.model_args.model_type
        if model_type.is_onnx():
            loader_name = self.model_args.model_file
            if self.onnx_shape_inference_args is not None:
                loader_name = self.onnx_shape_inference_args.add_to_script(script, loader_name)

            if loader_name == self.model_args.model_file:  # Shape inference loader isn't being used, have to load.
                script.add_import(imports=["OnnxFromPath"], frm="polygraphy.backend.onnx")
                loader_str = make_invocable(
                    "OnnxFromPath", self.model_args.model_file, external_data_dir=self.load_external_data
                )
                loader_name = script.add_loader(loader_str, "load_onnx", suffix=suffix)
        elif model_type.is_tf():
            if self.tf2onnx_loader_args is None:
                G_LOGGER.critical("Could not load: {:}. Is it an ONNX model?".format(self.model_args.model_file))
            loader_name = self.tf2onnx_loader_args.add_to_script(script)
        else:
            G_LOGGER.critical("Model type: {:} cannot be converted to ONNX.".format(model_type))

        loader_name = self._get_modify_onnx_loader(script, loader_name, disable_custom_outputs=disable_custom_outputs)

        if self.onnx_save_args is not None:
            loader_name = self.onnx_save_args.add_save_onnx(script, loader_name)

        return loader_name

    def should_use_onnx_loader(self, disable_custom_outputs=None):
        """
        Whether this model needs to be loaded via a Polygraphy ONNX loader, e.g., in case it
        needs modifications.
        """
        tmp_script = Script()
        inp_loader = "check_needs_modify"
        needs_modify = self._get_modify_onnx_loader(tmp_script, inp_loader, disable_custom_outputs) != inp_loader
        needs_shape_inference = (
            self.onnx_shape_inference_args is not None and self.onnx_shape_inference_args.do_shape_inference
        )
        needs_save = self.onnx_save_args is not None and self.onnx_save_args.path is not None
        # Currently, other loaders do not support external data, so we must fall back to the ONNX loader if it's present.
        return (
            not self.model_args.model_type.is_onnx()
            or needs_modify
            or self.load_external_data
            or needs_shape_inference
            or needs_save
        )

    def add_serialized_onnx_loader(self, script, disable_custom_outputs=None):
        script.add_import(imports=["BytesFromOnnx"], frm="polygraphy.backend.onnx")
        onnx_loader = self.add_onnx_loader(script, disable_custom_outputs=disable_custom_outputs)
        return script.add_loader(make_invocable("BytesFromOnnx", onnx_loader), "serialize_onnx")

    def load_onnx(self):
        loader = args_util.run_script(self.add_onnx_loader)
        return loader()
