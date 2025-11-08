#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
import tempfile

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from onnxconverter_common.float16 import convert_float_to_float16
from polygraphy.backend.onnx.loader import fold_constants

from demo_diffusion.model import load
from demo_diffusion.utils_modelopt import (
    cast_fp8_mha_io,
    cast_resize_io,
    convert_fp16_io,
    convert_zp_fp8,
)

# FIXME update callsites after serialization support for torch.compile is added
TORCH_INFERENCE_MODELS = ["default", "reduce-overhead", "max-autotune"]


def optimize_checkpoint(model, torch_inference: str):
    """Optimize a torch model checkpoint using torch.compile."""
    if not torch_inference or torch_inference == "eager":
        return model
    assert torch_inference in TORCH_INFERENCE_MODELS
    return torch.compile(model, mode=torch_inference, dynamic=False, fullgraph=False)


class Optimizer:

    def __init__(self, onnx_graph, verbose=False, version=None):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose
        self.version = version

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if load.onnx_graph_needs_external_data(onnx_graph):
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, "model.onnx")
            onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
            onnx.save_model(
                onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, hidden_layer_offset, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers + hidden_layer_offset
                ):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers + hidden_layer_offset
                ):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
        )
        k_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
        )
        v_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
        )

        qs = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]),
                )
            )
        )
        ks = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]),
                )
            )
        )
        vs = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]),
                )
            )
        )

        removed = 0
        assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(["attn1" in tensor for tensor in [q, k, v]])
            is_mhca = all(["attn2" in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]
                removed += 2
            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]
                removed += 1
        print(f"Removed {removed} QDQ nodes")
        return removed  # expected 72 for L2.5

    def modify_fp8_graph(self, is_fp16_io=True):
        onnx_graph = gs.export_onnx(self.graph)
        # Convert INT8 Zero to FP8.
        onnx_graph = convert_zp_fp8(onnx_graph)

        # WAR for legacy SD pipelines
        legacy_versions = (
            "1.4",
            "1.5",
            "2.1",
        )
        if any(self.version.startswith(prefix) for prefix in legacy_versions):
            onnx_graph = convert_float_to_float16(onnx_graph, keep_io_types=False, disable_shape_infer=True)

        self.graph = gs.import_onnx(onnx_graph)
        # Add cast nodes to Resize I/O.
        cast_resize_io(self.graph)
        # Convert model inputs and outputs to fp16 I/O.
        if is_fp16_io:
            convert_fp16_io(self.graph)
        # Add cast nodes to MHA's BMM1 and BMM2's I/O.
        cast_fp8_mha_io(self.graph)

    def flux_convert_rope_weight_type(self):
        for node in self.graph.nodes:
            if node.op == "Einsum":
                print(f"Fixed RoPE (Rotary Position Embedding) weight type: {node.name}")
        return gs.export_onnx(self.graph)
