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
import copy

from polygraphy import mod
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxInferShapesArgs, OnnxLoadArgs, OnnxSaveArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")


class Extract(BaseSurgeonSubtool):
    """
    Extract a subgraph from an ONNX model based on the specified inputs and outputs.
    """

    def __init__(self):
        super().__init__("extract")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(
                model_opt_required=True,
                input_shapes_opt_name="model-inputs",
                required_model_type="onnx",
                input_shapes_opt_doc="Input shapes to use when generating data to run fallback shape inference. "
                "Has no effect if fallback shape inference is not run",
            ),
            DataLoaderArgs(),
            OnnxInferShapesArgs(allow_force_fallback=True),
            OnnxLoadArgs(outputs_opt_prefix=False),
            OnnxSaveArgs(output_opt_required=True),
        ]

    def show_start_end_logging_impl(self, args):
        return True

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "--inputs",
            dest="input_meta",
            help="Input metadata for subgraph (names, shapes, and data types). "
            "Use 'auto' to make `extract` determine these automatically. Format: "
            "--inputs <name>:<shape>:<dtype>. "
            "For example: --inputs input0:[1,3,224,224]:float32 input1:auto:auto. "
            "If omitted, uses the current model inputs. ",
            nargs="+",
            default=[],
        )

        parser.add_argument(
            "--outputs",
            dest="output_meta",
            help="Output metadata for subgraph (names and data types). "
            "Use 'auto' to make `extract` determine these automatically. Format: "
            "--outputs <name>:<dtype>. "
            "For example: --outputs output0:float32 output1:auto. "
            "If omitted, uses the current model outputs. ",
            nargs="+",
            default=[],
        )

    def run_impl(self, args):
        def missing_meta_tensors(input_metadata, output_metadata):
            missing = TensorMetadata()
            for name, (dtype, shape) in input_metadata.items():
                if dtype is None or shape is None:
                    missing.add(name, dtype, shape)
            for name, (dtype, shape) in output_metadata.items():
                if dtype is None:
                    missing.add(name, dtype, shape)
            return missing

        model = super().load_model()

        user_input_metadata = args_util.parse_meta(args.input_meta)
        user_output_metadata = args_util.parse_meta(args.output_meta, includes_shape=False)

        # Loads an ONNX-GS graph and create new I/O metadata w/ info missing in user_input/output_metadata.
        def load_graph_and_io_meta(model):
            graph = onnx_backend.gs_from_onnx(model)
            TENSOR_MAP = graph.tensors()

            def get_tensor(name):
                if name not in TENSOR_MAP:
                    G_LOGGER.critical(f"Tensor: {name} does not exist in the model.")
                return TENSOR_MAP[name]

            # Makes a TensorMetadata for inputs/outputs using either the user provided information
            # or details derived from tensors.
            def make_io_meta(user_meta, tensors):
                if not user_meta:
                    return onnx_util.meta_from_gs_tensors(tensors)

                new_meta = copy.copy(user_meta)
                for name, (dtype, shape) in new_meta.items():
                    tensor = get_tensor(name)
                    new_meta.add(name, dtype or tensor.dtype, shape or tensor.shape)
                return new_meta

            input_metadata = make_io_meta(user_input_metadata, graph.inputs)
            output_metadata = make_io_meta(user_output_metadata, graph.outputs)
            return graph, input_metadata, output_metadata

        graph, input_metadata, output_metadata = load_graph_and_io_meta(model)

        # If we've already done ONNX shape inference, we should not do it again here.
        skip_shape_inference = (
            self.arg_groups[OnnxInferShapesArgs].force_fallback
            or self.arg_groups[OnnxInferShapesArgs].do_shape_inference
        )
        if missing_meta_tensors(input_metadata, output_metadata) and not skip_shape_inference:
            G_LOGGER.info(
                "Running shape inference to derive shapes and/or data types for `auto` arguments.\n"
                "To avoid this, you can specify the shapes and data types explicitly."
            )
            model = self.arg_groups[OnnxInferShapesArgs].infer_shapes(model, force=True)
            graph, input_metadata, output_metadata = load_graph_and_io_meta(model)

        missing_tensors = missing_meta_tensors(input_metadata, output_metadata)
        if missing_tensors or self.arg_groups[OnnxInferShapesArgs].force_fallback:
            # Use ONNX-Runtime with static shapes to infer shapes when all else fails
            # Returns a TensorMetadata for all tensors in the graph.
            if not self.arg_groups[OnnxInferShapesArgs].force_fallback:
                G_LOGGER.warning(
                    f"Some tensor shapes or dtypes are missing in the model. Note: Tensors with missing information:\n{missing_tensors}"
                    "\nWill run inference to determine shapes. This may cause some dynamic dimensions to become static."
                    "\nTo avoid this, please provide metadata on the command-line. "
                )
            else:
                G_LOGGER.info("Forcing fallback shape inference. This will cause dynamic dimensions to become static.")

            _, layerwise_meta = self.arg_groups[OnnxInferShapesArgs].fallback_inference(
                model, outputs=list(input_metadata.keys()) + list(output_metadata.keys())
            )

            def update_meta_from_layerwise(meta, user_meta, set_shapes=True):
                for name in meta:
                    # Choose between what the user set, what's in the model, and what
                    # fallback shape inference said.
                    def choose_meta(user, model, fallback):
                        if self.arg_groups[OnnxInferShapesArgs].force_fallback:
                            return user or fallback
                        return user or model or fallback

                    user_dtype, user_shape = None, None
                    if name in user_meta:
                        user_dtype, user_shape = user_meta[name].dtype, user_meta[name].shape

                    meta[name].dtype = choose_meta(user_dtype, meta[name].dtype, layerwise_meta[name].dtype)
                    if set_shapes:
                        meta[name].shape = choose_meta(user_shape, meta[name].shape, layerwise_meta[name].shape)
                    G_LOGGER.verbose(f"Updated tensor: {name} metadata to: {meta[name]}")
                return meta

            input_metadata = update_meta_from_layerwise(input_metadata, user_input_metadata)
            output_metadata = update_meta_from_layerwise(
                output_metadata, user_output_metadata, set_shapes=self.arg_groups[OnnxInferShapesArgs].force_fallback
            )

        graph = onnx_backend.extract_subgraph(graph, input_metadata, output_metadata)
        super().save_model(super().export_graph(graph))
