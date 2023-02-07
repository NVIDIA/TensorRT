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
from polygraphy.common.interface import TypedDict
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import OnnxInferShapesArgs, OnnxLoadArgs, ModelArgs, OnnxSaveArgs
from polygraphy.tools.base import Tool

common_backend = mod.lazy_import("polygraphy.backend.common")
gs = mod.lazy_import("onnx_graphsurgeon")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
trt = mod.lazy_import("tensorrt")
trt_backend = mod.lazy_import("polygraphy.backend.trt")
trt_util = mod.lazy_import("polygraphy.backend.trt.util")
util = mod.lazy_import("polygraphy.util")


class UnsupportedNodeDict(TypedDict(lambda: str, lambda: dict)):
    """
    An ordered dictionary that maps ops to error(s) encountered by TensorRT
    while trying to parse them, and the range of node indices for the subgraphs
    where these errors were encountered.

    More specifically, it is an ``OrderedDict[str, Dict[str, List[Tuple[int]]]]``.
    """

    def add(self, op, err_string, node_range):
        """
        Add a single entry for a single error in a subgraph.

        Multiple node ranges may apply to a single op/error combination.

        Args:
            op (str): The name of the op that was unsupported.
            err_string (str): The error encountered.
            node_range (Union[Tuple[int], int]):
                    The start (inclusive) and end (exclusive) node indices of the subgraph
        """
        if op not in self:
            self[op] = {}

        if err_string not in self[op]:
            self[op][err_string] = []

        self[op][err_string].append(node_range)


def supports_model(path):
    """
    Invokes the ONNX parser's `supports_model` on the specified model.

    Args:
        path (str): The path to the ONNX model.

    Returns:
        Tuple[bool, SubgraphCollection, parser]:
                (1) Whether the model is supported.
                (2) A List[Tuple[List[int], bool]] mapping groups of node indices to a boolean
                    indicating whether they are supported.
                (3) The TensorRT ONNX parser instance.
    """
    _, network = trt_backend.create_network()
    parser = trt.OnnxParser(network, trt_backend.get_trt_logger())

    try:
        parser.supports_model
    except AttributeError:
        trt_util.fail_unavailable("supports_model in tensorrt.OnnxParser")

    supported, nodelists = parser.supports_model(common_backend.bytes_from_path(path), path)
    return supported, nodelists, parser


def save_subgraph(onnx_save_args, graph, start, end, prefix="", use_tmp_file=False):
    """
    Extracts a subgraph from the main graph and saves it to disk.

    Args:
        graph (onnx_graphsurgeon.Graph): The parent/main graph.
        start (int): The (inclusive) index of the start node.
        end (int): The (exclusive) index of the end node.
        prefix (str): The prefix for the model file name.
        use_tmp_file (bool):
                Whether the subgraph should be written to a temporary file instead of the output directory.

    Returns:
        str: The full path to the ONNX model of the subgraph.
    """
    subgraph_nodes = graph.nodes[start:end]
    out_dict = {out.name: out for node in subgraph_nodes for out in node.outputs}
    in_dict = {inp.name: inp for node in subgraph_nodes for inp in node.inputs}

    # Guess graph inputs/outputs by checking all output tensor names against all input tensor names, and vice-versa.
    subgraph_inputs = onnx_util.meta_from_gs_tensors([in_dict[k] for k in in_dict if k not in out_dict])
    subgraph_outputs = onnx_util.meta_from_gs_tensors([out_dict[k] for k in out_dict if k not in in_dict])

    subgraph = gs.export_onnx(onnx_backend.extract_subgraph(graph, subgraph_inputs, subgraph_outputs))

    if use_tmp_file:
        path = util.NamedTemporaryFile(prefix=prefix, suffix=".onnx").name
    else:
        # end is exclusive, so subtract one to make the model names friendlier.
        path = os.path.join(onnx_save_args.path, f"{prefix}_subgraph-nodes-{start}-{end - 1}.onnx")
    onnx_save_args.save_onnx(subgraph, path)
    return path


def gen_results_summary(final_unsupported):
    """
    Generates a results summary given the final unsupported nodes dictionary.

    Args:
        final_unsupported (UnsupportedNodeDict):
                The unsupported ops and corresponding errors and node index ranges.

    Returns:
        str: A summary of all the unsupported ops in model, along with reasons and node index ranges.
    """
    op_width = max(map(len, list(final_unsupported.keys()) + ["Operator "]))
    reason_width = max(len(reason) for node_index_map in final_unsupported.values() for reason in node_index_map.keys())

    summary = "===== Summary =====\n"

    header = f"{'Operator':{op_width}}| {'Count':7} | {'Reason':{reason_width}} | Nodes\n"
    summary += header + "-" * len(header) + "\n"

    for op, node_index_map in final_unsupported.items():
        for reason, node_indices in node_index_map.items():
            summary += f"{op:{op_width}}| {len(node_indices):7} | {reason:{reason_width}} | {node_indices}\n"
    return summary


class Capability(Tool):
    """
    Determine the capability of TensorRT to run an ONNX graph. Graph will be paritioned into supported and unsupported subgraphs.
    """

    def __init__(self):
        super().__init__("capability")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, input_shapes_opt_name=False, required_model_type="onnx"),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(outputs_opt_prefix=False),
            OnnxSaveArgs(output_default_path="polygraphy_capability_dumps", allow_multiple_models=True),
        ]

    def run_impl(self, args):
        supported, nodelists, _ = supports_model(self.arg_groups[ModelArgs].path)
        if supported:
            G_LOGGER.info("Graph is fully supported by TensorRT; Will not generate subgraphs.")
            return

        parent_graph = onnx_backend.gs_from_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())

        def partition(nodelists, offset):
            """
            Partitions a set of subgraphs into supported and unsupported subgraphs.

            Args:
                nodelists (List[Tuple[List[int], bool]]):
                        A list that maps node indices to a boolean indicating whether they
                        are supported by TensorRT.

            Returns:
                List[List[int]]:
                        A list of subgraphs supported by TensorRT, each described by a list of node indices.
            """
            supported_subgraphs = []
            for (node_indices, supported) in nodelists:
                if supported:
                    supported_subgraphs.append([index + offset for index in node_indices])
                    continue

                start = node_indices[0] + offset
                end = node_indices[-1] + offset + 1
                subgraph_path = save_subgraph(
                    self.arg_groups[OnnxSaveArgs],
                    parent_graph,
                    start,
                    end,
                    prefix="intermediate_",
                    use_tmp_file=True,
                )
                _, new_nodelists, _ = supports_model(subgraph_path)
                # Recursively partition each unsupported subgraph.
                supported_subgraphs += partition(new_nodelists, start)

            return supported_subgraphs

        supported_subgraphs = partition(nodelists, offset=0)
        unsupported_node_dict = UnsupportedNodeDict()

        def save_unsupported_graph(start, end):
            """
            Saves an unsupported subgraph, determines the error reason and adds it
            to unsupported_node_dict

            Args:
                start (int): The (inclusive) index of the start node.
                end (int): The (exclusive) index of the end node.
            """
            subgraph_path = save_subgraph(self.arg_groups[OnnxSaveArgs], parent_graph, start, end, "unsupported")
            _, _, parser = supports_model(subgraph_path)

            err_string = (
                " | ".join([str(parser.get_error(err_idx)) for err_idx in range(parser.num_errors)]) or "UNKNOWN ERROR"
            )
            unsupported_node_dict.add(parent_graph.nodes[start].op, err_string, [start, end])

        # Log errors for all the unsupported graphs between supported subgraphs.
        for index, subg_node_idxs in enumerate(supported_subgraphs):
            save_subgraph(
                self.arg_groups[OnnxSaveArgs],
                parent_graph,
                subg_node_idxs[0],
                subg_node_idxs[-1] + 1,
                "supported",
            )

            if index == 0 and subg_node_idxs[0] != 0:
                save_unsupported_graph(0, subg_node_idxs[0])

            if index == len(supported_subgraphs) - 1 and supported_subgraphs[-1][-1] != len(parent_graph.nodes) - 1:
                save_unsupported_graph(subg_node_idxs[-1] + 1, len(parent_graph.nodes))

            if index < len(supported_subgraphs) - 1:
                next_subg_node_idxs = supported_subgraphs[index + 1]
                save_unsupported_graph(subg_node_idxs[-1] + 1, next_subg_node_idxs[0])

        summary = gen_results_summary(unsupported_node_dict)

        G_LOGGER.info(summary)
        util.save_file(
            summary, os.path.join(self.arg_groups[OnnxSaveArgs].path, "results.txt"), "w", description="results"
        )
