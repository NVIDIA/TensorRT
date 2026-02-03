#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch
import triton
import triton.language as tl

import tensorrt as trt
import cupy as cp
import numpy as np
import ast

from polygraphy.backend.trt import (
    CreateConfig,
    TrtRunner,
    create_network,
    engine_from_network,
)

import argparse

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("AliasedIOPlugin").setLevel(logging.INFO)
log = logging.getLogger("AliasedIOPlugin")

import sys

# An OpenAI Triton kernel to both perform the scatter-add and counts of each index
@triton.jit
def scatter_add_kernel(
    self_ptr,
    src_ptr,  # Source array
    index_ptr,  # Indices
    n_elements,  # Number of elements in the source/indices array
    n_labels,  # Number of labels (distinct indices)
    counts,  # Output counts of each distinct index
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load the source values and indices
    src = tl.load(src_ptr + offsets, mask=mask)
    indices = tl.load(index_ptr + offsets, mask=mask)

    # Iterate over n_labels
    for i in range(0, BLOCK_SIZE_C):
        idx = i + tl.program_id(1) * BLOCK_SIZE_C + 1
        if idx <= n_labels:
            l_mask = indices == idx
            # Perform the scatter-add operation
            tl.atomic_add(self_ptr + idx - 1, tl.sum(tl.where(l_mask, src, 0)))
            # Update count for idx
            tl.atomic_add(counts + idx - 1, tl.sum(tl.where(l_mask, 1, 0)))


def volume(d):
    return np.prod(d)


class UnownedMemory:
    def __init__(self, ptr, shape, dtype):
        mem = cp.cuda.UnownedMemory(ptr, volume(shape) * cp.dtype(dtype).itemsize, self)
        cupy_ptr = cp.cuda.MemoryPointer(mem, 0)
        self.d = cp.ndarray(shape, dtype=dtype, memptr=cupy_ptr)


class ScatterAddPlugin(
    trt.IPluginV3,
    trt.IPluginV3OneCore,
    trt.IPluginV3OneBuildV2,
    trt.IPluginV3OneRuntime,
):
    def __init__(self, fc=None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuildV2.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.plugin_namespace = ""
        self.plugin_name = "ScatterAddPlugin"
        self.plugin_version = "1"
        self.num_outputs = 2

    def get_capability_interface(self, type):
        return self

    def get_output_data_types(self, input_types):
        self.type = input_types[0]
        return [input_types[0], trt.int64]

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection([])

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [
            inputs[0],
            trt.DimsExprs([inputs[0][0], exprBuilder.constant(1)]),
        ]

        return output_dims

    def configure_plugin(self, inp, out):
        pass

    def on_shape_change(self, inp, out):
        pass

    def supports_format_combination(
        self, pos: int, in_out: "list[trt.PluginTensorDesc]", num_inputs: int
    ):
        assert num_inputs == 3
        assert pos < len(in_out)

        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # self, src and output have the same type
        if pos in [0, 1, 3]:
            return desc.type == self.type

        # indices anc the counts output are int64
        return desc.type == trt.int64

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        # No-copy operations to setup torch tensors over the I/O buffers
        inp_mem = UnownedMemory(
            inputs[0], input_desc[0].dims, trt.nptype(input_desc[0].type)
        )
        src_mem = UnownedMemory(
            inputs[1], input_desc[1].dims, trt.nptype(input_desc[1].type)
        )
        idx_mem = UnownedMemory(
            inputs[2], input_desc[2].dims, trt.nptype(input_desc[2].type)
        )
        counts_mem = UnownedMemory(
            outputs[1], output_desc[1].dims, trt.nptype(output_desc[1].type)
        )

        inp = torch.as_tensor(inp_mem.d, device="cuda")
        src = torch.as_tensor(src_mem.d, device="cuda")
        idx = torch.as_tensor(idx_mem.d, device="cuda")
        counts = torch.as_tensor(counts_mem.d, device="cuda")

        # Zero out the counts before passing to kernel
        counts.zero_()

        n_classes = inp.shape[0]
        n_elements = src.numel()

        # Block size definitions
        BLOCK_SIZE = 1024
        BLOCK_SIZE_C = 32

        # Calculate grid size
        grid_x = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_y = (n_classes + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C

        scatter_add_kernel[(grid_x, grid_y)](
            inp, src, idx, n_elements, n_classes, counts, BLOCK_SIZE, BLOCK_SIZE_C
        )

    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def get_aliased_input(self, output_index: int):
        if output_index == 0:
            return 0

        return -1

    def clone(self):
        cloned_plugin = ScatterAddPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class ScatterAddPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "ScatterAddPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        return ScatterAddPlugin()


def torch_ref(node_features, edges, W, precision):
    # Initialize an output tensor for aggregation
    aggregated = torch.zeros_like(node_features, dtype=precision, device="cuda")

    # Perform aggregation using scatter_add_
    aggregated.scatter_add_(0, edges[:, 1].unsqueeze(1), node_features[edges[:, 0]])

    # Get the counts of each distinct index
    bincounts = torch.bincount(edges[:, 1].contiguous())

    # Normalize and classify
    Y = W * (aggregated / bincounts.unsqueeze(1)).transpose(1, 0)
    return torch.softmax(torch.relu(Y), dim=0)


numpy_to_torch_dtype = {
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
}


def parse_edges_string(input_string):
    try:
        # Parse the string into a list of integer pairs
        raw_edges = ast.literal_eval(input_string)

        # Check if the parsed object is a list
        if not isinstance(raw_edges, list):
            return None, "The input string does not represent a list."

        edges = []
        for edge in raw_edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(x, int) for x in edge)
            ):
                return (
                    None,
                    f"Each edge must be a list of two integers. Invalid edge: {edge}",
                )
            edges.append(edge)

        return edges, None
    except (SyntaxError, ValueError) as e:
        return None, f"Error parsing string: {e}"


def validate_edges(edges, n_nodes):
    for edge in edges:
        src, target = edge
        if not (0 <= src < n_nodes) or not (0 <= target < n_nodes):
            return f"Edge ({src}, {target}) is out of bounds. Must be in range [0, {n_nodes - 1}]."

    # check incoming edges
    incoming_edges_count = [0] * n_nodes
    for _, target in edges:
        incoming_edges_count[target] += 1

    for idx in range(n_nodes):
        if incoming_edges_count[idx] == 0:
            return f"Index {idx} has no incoming edges."
    return None


def parse_edges(input_string, n_nodes):
    parsed_edges, parse_error = parse_edges_string(input_string)
    if parse_error:
        return None, parse_error
    else:
        # Validate the edges
        validation_error = validate_edges(parsed_edges, n_nodes)
        if validation_error is not None:
            return None, validation_error
        else:
            return parsed_edges, None


# Print adjacency matrix
def print_graph(edges, n_nodes):
    adjacency_matrix = [[0] * n_nodes for _ in range(n_nodes)]

    for src, tgt in edges:
        adjacency_matrix[src][tgt] = 1

    for row in adjacency_matrix:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision for node features",
    )
    parser.add_argument(
        "--node_features",
        type=str,
        default="[1.0,3.0,5.0,7.0]",
        help="List of node features as a comma-separated list. e.g. [1.0,2.0,3.0].",
    )
    parser.add_argument(
        "--edges",
        type=str,
        default="[[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]]",
        help="Pairs of source->target directed edges. Every node must have at least one incoming edge. e.g. [[0,1],[1,0]].",
    )
    parser.add_argument(
        "--num_classes", type=int, default=3, help="Number of classes in the classifier"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate result with reference"
    )
    parser.add_argument("--seed", type=int, help="Seed to use for weights generation")

    args = parser.parse_args()

    if args.seed is not None:
        print("Setting seed to:", args.seed)
        torch.manual_seed(args.seed)
    else:
        print("Setting seed to:", torch.seed())

    precision = trt.float32 if args.precision == "fp32" else trt.float16
    n_classes = args.num_classes

    numpy_precision = trt.nptype(precision)
    torch_precision = numpy_to_torch_dtype[numpy_precision]

    if args.num_classes < 1:
        parser.print_help()
        log.error("num_classes must be a positive integer")
        sys.exit(1)

    try:
        float_list = ast.literal_eval(args.node_features)
        if not isinstance(float_list, list):
            parser.print_help()
            log.error("The node_features string does not represent a list")
            sys.exit(1)

        # Check if all elements in the list are floats/ints
        if not all(isinstance(x, (float, int)) for x in float_list):
            parser.print_help()
            log.error("The node_features list must contain only numbers")
            sys.exit(1)
    except (SyntaxError, ValueError) as e:
        parser.print_help()
        log.error(f"The node_features string could not be parsed as a list: {e}")
        sys.exit(1)

    node_features = torch.tensor(float_list, dtype=torch_precision, device="cuda").view(
        -1, 1
    )

    n_nodes = node_features.shape[0]

    parsed_edges, parse_error = parse_edges(args.edges, n_nodes)
    if parse_error:
        parser.print_help()
        log.error(parse_error)
        sys.exit(1)

    edges = torch.tensor(parsed_edges, device="cuda", dtype=torch.int64)

    print()
    print("Adjacency matrix for graph:")
    print_graph(edges, n_nodes)
    print()

    target = torch.zeros_like(node_features, device="cuda")

    input_x = target.clone()
    input_src = node_features[edges[:, 0]].flatten()
    input_idx = edges[:, 1].contiguous() + 1

    W = torch.randn((n_classes, 1), dtype=torch_precision, device="cuda")

    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = ScatterAddPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    builder, network = create_network(strongly_typed=True)
    input_x_T = network.add_input(name="X", dtype=precision, shape=input_x.shape)
    input_src_T = network.add_input(name="src", dtype=precision, shape=input_src.shape)
    input_idx_T = network.add_input(name="idx", dtype=trt.int64, shape=input_idx.shape)
    w_T = network.add_input(name="W", dtype=precision, shape=W.shape)
    out = network.add_plugin_v3(
        [input_x_T, input_src_T, input_idx_T], [], ScatterAddPlugin()
    )
    cast_layer = network.add_cast(out.get_output(1), precision)
    div_layer = network.add_elementwise(
        out.get_output(0),
        cast_layer.get_output(0),
        op=trt.ElementWiseOperation.FLOOR_DIV,
    )
    matmul_layer = network.add_matrix_multiply(
        w_T,
        trt.MatrixOperation.NONE,
        div_layer.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    relu_layer = network.add_activation(
        matmul_layer.get_output(0), type=trt.ActivationType.RELU
    )
    softmax_layer = network.add_softmax(relu_layer.get_output(0))
    softmax_layer.get_output(0).name = "softmax"
    network.mark_output(tensor=softmax_layer.get_output(0))
    build_engine = engine_from_network(
        (builder, network),
        CreateConfig(
            preview_features=[trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03],
        ),
    )

    with TrtRunner(build_engine, "trt_runner") as runner:
        outputs = runner.infer(
            {"X": input_x, "src": input_src, "idx": input_idx, "W": W},
            copy_outputs_to_host=False,
        )

        print()
        print("Classifier output:")
        print(outputs["softmax"])
        print()

        if args.validate:
            tref = torch_ref(node_features, edges, W, torch_precision)
            if torch.allclose(outputs["softmax"], tref, 1e-2):
                print("Validation against reference successful!")
            else:
                print("Validation against reference failed!")
