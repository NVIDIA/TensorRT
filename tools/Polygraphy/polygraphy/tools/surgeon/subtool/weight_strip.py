#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.sparse import SparsityPruner
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool
from polygraphy.tools.args import ModelArgs, OnnxLoadArgs, OnnxSaveArgs

gs = mod.lazy_import("onnx_graphsurgeon")
np = mod.lazy_import("numpy")
onnx = mod.lazy_import("onnx")

class WeightStripperArgs(BaseArgs):
    """
    Weight Stripping: weight stripping
    """
    def add_parser_args_impl(self):
        self.group.add_argument(
            "--exclude-list",
            help="Path to text file containing a list of initializers to skip",
            default=None,
            required=False
        )
    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            exclude_list (str): Path to text file containing a list of initializers to skip.
        """
        self.exclude_list = args_util.get(args, "exclude_list")

    def get_exclude_list(self):
        if not self.exclude_list:
            return set()
        with open(self.exclude_list) as fp:
            lines = [line.rstrip() for line in fp]
            return set(lines)

def get_patterns():
    """
    Base Patterns contain single ops: Conv, ConvTranspose, Gemm, Gather, MatMul, Slice

    Q/DQ Patterns contain sequences with Q/DQ ops followed by a few base patterns:
        [QuantizeLinear, DequantizeLinear, Conv]
        [QuantizeLinear, DequantizeLinear, ConvTranspose]
        [QuantizeLinear, DequantizeLinear, Gemm]
        [QuantizeLinear, DequantizeLinear, MatMul]

    Transpose Patterns contain sequences with the Transpose op followed  by a few base patterns:
        [Transpose, Conv]
        [Transpose, ConvTranspose]
        [Transpose, Gemm]
        [Transpose, MatMul]
    """
    base_patterns = []

    # dictionary storing the index of the input the Producer output can be linked to
    input_positions = {
        'Conv': [0],
        'ConvTranspose': [0],
        'Gemm': [0, 1, 2],
        'MatMul': [0, 1],
    }

    # Conv with Weight input
    conv_w = gs.GraphPattern()
    in_0 = conv_w.variable()
    w = conv_w.variable()
    conv_w_out = conv_w.add("conv_w", "Conv", inputs=[in_0, w])
    conv_w.set_output_tensors([conv_w_out])
    base_patterns.append(conv_w)

    # Conv with Weight and Bias inputs
    conv_w_b = gs.GraphPattern()
    in_0 = conv_w_b.variable()
    w = conv_w_b.variable()
    b = conv_w_b.variable()
    conv_w_b_out = conv_w_b.add("conv_w_b", "Conv", inputs=[in_0, w, b])
    conv_w_b.set_output_tensors([conv_w_b_out])
    base_patterns.append(conv_w_b)

    # ConvTranspose with Weight input
    convt_w = gs.GraphPattern()
    in_0 = convt_w.variable()
    w = convt_w.variable()
    convt_w_out = convt_w.add("convt_w", "ConvTranspose", inputs=[in_0, w])
    convt_w.set_output_tensors([convt_w_out])
    base_patterns.append(convt_w)

    # ConvTranspose with Weight and Bias inputs
    convt_w_b = gs.GraphPattern()
    in_0 = convt_w_b.variable()
    w = convt_w_b.variable()
    b = convt_w_b.variable()
    convt_w_b_out = convt_w_b.add("convt_w_b", "ConvTranspose", inputs=[in_0, w, b])
    convt_w_b.set_output_tensors([convt_w_b_out])
    base_patterns.append(convt_w_b)

    # Gemm with A and B inputs
    gemm_1 = gs.GraphPattern()
    in_0 = gemm_1.variable()
    in_1 = gemm_1.variable()
    gemm_1_out = gemm_1.add("gemm_1", "Gemm", inputs=[in_0, in_1])
    gemm_1.set_output_tensors([gemm_1_out])
    base_patterns.append(gemm_1)

    # Gemm with A, B and C inputs
    gemm_2 = gs.GraphPattern()
    in_0 = gemm_2.variable()
    in_1 = gemm_2.variable()
    in_2 = gemm_2.variable()
    gemm_2_out = gemm_2.add("gemm_2", "Gemm", inputs=[in_0, in_1, in_2])
    gemm_2.set_output_tensors([gemm_2_out])
    base_patterns.append(gemm_2)

    # MatMul
    matmul = gs.GraphPattern()
    in_0 = matmul.variable()
    in_1 = matmul.variable()
    matmul_out = matmul.add("matmul", "MatMul", inputs=[in_0, in_1])
    matmul.set_output_tensors([matmul_out])
    base_patterns.append(matmul)

    # Q/DQ patterns
    # QuantizeLinear with y_scale input
    q_1 = gs.GraphPattern()
    in_0 = q_1.variable()
    y_scale = q_1.variable()
    q_1_out = q_1.add("q_1", "QuantizeLinear", inputs=[in_0, y_scale])
    q_1.set_output_tensors([q_1_out])

    # QuantizeLinear with y_scale and y_zero_point inputs
    q_2 = gs.GraphPattern()
    in_0 = q_2.variable()
    y_scale = q_2.variable()
    y_zero_point = q_2.variable()
    q_2_out = q_2.add("q_2", "QuantizeLinear", inputs=[in_0, y_scale, y_zero_point])
    q_2.set_output_tensors([q_2_out])

    # DequantizeLinear with x_scale input
    dq_1 = gs.GraphPattern()
    in_0 = dq_1.variable()
    x_scale = dq_1.variable()
    dq_1_out = dq_1.add("dq_1", "DequantizeLinear", inputs=[in_0, x_scale])
    dq_1.set_output_tensors([dq_1_out])

    # QuantizeLinear with y_scale and y_zero_point inputs
    dq_2 = gs.GraphPattern()
    in_0 = dq_2.variable()
    x_scale = dq_2.variable()
    x_zero_point = dq_2.variable()
    dq_2_out = dq_2.add("dq_2", "DequantizeLinear", inputs=[in_0, x_scale, x_zero_point])
    dq_2.set_output_tensors([dq_2_out])

    qdq_patterns = []
    for op in base_patterns:
        # all base patterns contain a single node
        op_type = next(iter(op.nodes.values())).op
        for input_pos in input_positions[op_type]:
            for q in [q_1, q_2]:
                for dq in [dq_1, dq_2]:
                    curr_pattern = gs.GraphPattern()

                    q_inps = [curr_pattern.variable() for _ in range(len(q.input_tensors))]
                    q_out = curr_pattern.add("Q", q, inputs=q_inps)

                    dq_inps = [curr_pattern.variable() for _ in range(len(dq.input_tensors) - 1)]
                    dq_out = curr_pattern.add("DQ", dq, inputs=[q_out] + dq_inps)

                    # in case of Gemm with 2 inputs, skip the case where output of dq node is the 3rd input of Gemm
                    if len(op.input_tensors) <= input_pos:
                        continue
                    op_inps = [curr_pattern.variable() for _ in range(len(op.input_tensors))]
                    op_inps[input_pos] = dq_out
                    out = curr_pattern.add("base_op", op, inputs=op_inps)
                    curr_pattern.set_output_tensors([out])
                    qdq_patterns.append(curr_pattern)

    # Transpose patterns
    transpose_patterns = []
    transpose = gs.GraphPattern()
    in_0 = transpose.variable()
    transpose_out = transpose.add("transpose", "Transpose", inputs=[in_0])
    transpose.set_output_tensors([transpose_out])

    for op in base_patterns:
        # all base patterns contain a single node
        op_type = next(iter(op.nodes.values())).op
        for input_pos in input_positions[op_type]:
            curr_pattern = gs.GraphPattern()

            t_inps = [curr_pattern.variable() for _ in range(len(transpose.input_tensors))]
            t_out = curr_pattern.add("t", transpose, inputs=t_inps)

            # in case of Gemm with 2 inputs, skip the case where output of transpose node is the 3rd input of Gemm
            if len(op.input_tensors) <= input_pos:
                continue
            op_inps = [curr_pattern.variable() for _ in range(len(op.input_tensors))]
            op_inps[input_pos] = t_out
            out = curr_pattern.add("base_op", op, inputs=op_inps)
            curr_pattern.set_output_tensors([out])
            transpose_patterns.append(curr_pattern)

    # Gather
    gather = gs.GraphPattern()
    in_0 = gather.variable()
    indices = gather.variable()
    gather_out = gather.add("gather", "Gather", inputs=[in_0, indices])
    gather.set_output_tensors([gather_out])
    base_patterns.append(gather)

    # Slice without no optional inputs
    slice_0 = gs.GraphPattern()
    in_0 = slice_0.variable()
    starts = slice_0.variable()
    ends = slice_0.variable()
    slice_0_out = slice_0.add("slice_0", "Slice", inputs=[in_0, starts, ends])
    slice_0.set_output_tensors([slice_0_out])
    base_patterns.append(slice_0)

    # Slice with axes inputs
    slice_1 = gs.GraphPattern()
    in_0 = slice_1.variable()
    starts = slice_1.variable()
    ends = slice_1.variable()
    axes = slice_1.variable()
    slice_1_out = slice_1.add("slice_1", "Slice", inputs=[in_0, starts, ends, axes])
    slice_1.set_output_tensors([slice_1_out])
    base_patterns.append(slice_1)

    # Slice with steps inputs
    slice_2 = gs.GraphPattern()
    in_0 = slice_2.variable()
    starts = slice_2.variable()
    ends = slice_2.variable()
    steps = slice_2.variable()
    slice_2_out = slice_2.add("slice_2", "Slice", inputs=[in_0, starts, ends, steps])
    slice_2.set_output_tensors([slice_2_out])
    base_patterns.append(slice_2)

    # Slice with axes and steps inputs
    slice_3 = gs.GraphPattern()
    in_0 = slice_3.variable()
    starts = slice_3.variable()
    ends = slice_3.variable()
    axes = slice_3.variable()
    steps = slice_3.variable()
    slice_3_out = slice_3.add("slice_3", "Slice", inputs=[in_0, starts, ends, axes, steps])
    slice_3.set_output_tensors([slice_3_out])
    base_patterns.append(slice_3)

    return base_patterns + qdq_patterns + transpose_patterns
    
def get_size_thresholds():
    """
    Strip the initializers of the ops only if the size threshold has been crossed
    """
    return {
        'Conv': 1,
        'ConvTranspose': 1,
        'Gather': 1024,
        'Gemm': 1,
        'Plugin': 1024,
        'Slice': 1024,
    }

def get_inputs_to_strip():
    """
    Restrict the stripping of initializers of the ops to the input index specified
    """
    return {
        'QuantizeLinear': set([0]),
        'Slice': set([0]),
    }

class WeightStripper(BaseSurgeonSubtool):
    """
    Strip weights from the provided ONNX model
    """
    def __init__(self):
        super().__init__("weight-strip")

    def show_start_end_logging_impl(self, args):
        return True

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, input_shapes_opt_name=False, required_model_type="onnx"),
            OnnxLoadArgs(allow_shape_inference=False, outputs_opt_prefix=False, allow_from_tf=False),
            OnnxSaveArgs(allow_shape_inference=False, output_opt_required=True),
            WeightStripperArgs()
        ]

    def __skip(self, node, inp, inp_index):
        """
        Skip stripping the input based on pre-defined heuristics

        The function also modifies exclude_list if a matching input is found
        """
        # restrict stripping of certain op inputs
        if node.op in self.inputs_to_strip and inp_index not in self.inputs_to_strip[node.op]:
            return True
        # Skip inputs that are not initializers
        if not isinstance(inp, gs.Constant):
            return True
        # Skip initializers with uint8 dtype
        if inp.dtype == np.uint8:
            return True
        # Skip initializers specified in user defined skip list
        if inp.name in self.exclude_list:
            self.exclude_list.remove(inp.name)
            return True
        # Heuristic to strip based on size
        if node.op in self.size_thresholds and inp.values.size < self.size_thresholds[node.op]:
            return True
        
        return False

    def __get_matching_subgraph_inputs(self, graph):
        """
        Use GraphPattern to find matching patterns in the graph
        """
        for pattern in self.patterns:
            subgraphs = pattern.match_all(graph)
            for subgraph in subgraphs:
                # the first node in the matched subgraph contains the initializer to strip
                curr_node = next(iter(subgraph.values()))
                while curr_node._get_node() is None:
                    curr_node = next(iter(curr_node.values()))
                onnx_node = curr_node.onnx_node
                for inp_index, inp in enumerate(onnx_node.inputs):
                    if not self.__skip(onnx_node, inp, inp_index):
                        self.initializers_to_strip.add(inp.name)

    def __get_plugin_inputs(self, nodes):
        """
        Identify Plugin inputs to strip
        """
        for node in nodes:
            # If plugin found
            if not onnx.defs.has(node.op):
                for inp_index, inp in enumerate(node.inputs):
                    if not self.__skip(node, inp, inp_index):
                        G_LOGGER.verbose(f"Stripping initializer {inp.name} to the {node.op} op.")
                        self.initializers_to_strip.add(inp.name)
    
    def __get_sparse_tensors(self, model):
        """
        Identify sparse tensors in the model
        """
        sparsity_checker = SparsityPruner(model)
        sparsity_checker.check()
        sparse_tensors = sparsity_checker.sparse_tensors
        return sparse_tensors

    def run_impl_surgeon(self, args):
        def strip_weights(model):
            G_LOGGER.start(f"Beginning weight stripping...")
            G_LOGGER.warning(f"The model is expected to be constant folded to successfully capture all weights eligible for stripping")
            graph = gs.import_onnx(model)
            # check model sparsity
            G_LOGGER.info("Querying Sparse Initializers in the model")
            sparse_initializers = self.__get_sparse_tensors(model)

            # Call PatternMatcher to populate initializers_to_strip
            self.__get_matching_subgraph_inputs(graph)
            self.__get_plugin_inputs(graph.nodes)

            # Strip initializers identified by the PatternMatcher and Plugin Identifier
            num_stripped = 0
            for initializer in model.graph.initializer:
                if initializer.name in self.initializers_to_strip:
                    G_LOGGER.verbose(f"Stripping initializer {initializer.name}")
                    # Erase initializer data
                    initializer.raw_data = b""

                    # Check sparsity
                    sparse_str = "SPARSE_2_4" if initializer.name in sparse_initializers else ""
                    
                    # Update initializer doc_string
                    initializer.doc_string = '/'.join(["TRT_WEIGHTLESS", sparse_str])
                    num_stripped += 1

            if self.exclude_list:
                G_LOGGER.warning(f"The following weights provided by the user to skip stripping were not found in the model: {self.exclude_list}.")
            assert num_stripped == len(self.initializers_to_strip)

            if num_stripped:
                model.doc_string = '-'.join(filter(None, [model.doc_string, "TRT_WEIGHTLESS"]))
            G_LOGGER.finish(f"Finished stripping {num_stripped} weights")

            return model

        # Initialize patterns
        self.patterns = get_patterns()
        self.size_thresholds = get_size_thresholds()
        self.inputs_to_strip = get_inputs_to_strip()
        self.initializers_to_strip = set()

        # load model
        model = super().load_model()

        self.exclude_list = self.arg_groups[WeightStripperArgs].get_exclude_list()
        stripped_model = strip_weights(model)
        super().save_model(stripped_model)
