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
"""
Single canonical text formatter for all model types.

``str_from_graph_data`` is the only function that converts a ``GraphData`` object
to a human-readable string.  Backend utils (``onnx/util.py``, ``trt/util.py``)
delegate their ``str_from_*`` functions here after extracting a ``GraphData`` via
``extractors.py``.
"""
from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    TensorAttrValue,
    TensorInfo,
)

np = mod.lazy_import("numpy")

# Section-name prefixes per model type.
_SECTION_PREFIX = {
    "onnx": "Graph",
    "trt_network": "Network",
    "trt_engine": "Engine",
}
# Node-label prefix per model type (used in util.str_from_layer).
_NODE_PREFIX = {
    "onnx": "Node",
    "trt_network": "Layer",
    "trt_engine": "Layer",
}


def _build_tensor_metadata(tensor_infos, prefix_initializers=True):
    """Reconstruct a ``TensorMetadata`` from a list of ``TensorInfo`` objects."""
    meta = TensorMetadata()
    for ti in tensor_infos:
        if prefix_initializers and ti.is_initializer:
            display_name = f"Initializer | {ti.name}"
        else:
            display_name = ti.name
        meta.add(
            name=display_name, dtype=ti.dtype, shape=ti.shape, docstring=ti.docstring
        )
    return meta


def _node_names_and_meta(tensor_infos, prefix_initializers=False):
    """Return (display_names, TensorMetadata) for a list of TensorInfo objects."""
    if prefix_initializers:
        names = [
            f"Initializer | {ti.name}" if ti.is_initializer else ti.name
            for ti in tensor_infos
        ]
    else:
        names = [ti.name for ti in tensor_infos]
    return names, _build_tensor_metadata(tensor_infos)


def _format_attr_value(
    val, show_layers, show_attrs, show_weights, indent_level, is_trt
):
    """
    Format a single node attribute value into a display string.

    Returns ``None`` if the attribute should be skipped (e.g. numpy array when
    ``show_weights`` is False in TRT mode).
    """
    # TRT numpy-array weight: only show if show_weights is requested.
    if is_trt:
        try:
            if isinstance(val, np.ndarray) and not show_weights:
                return None
        except Exception:
            pass

    if isinstance(val, GraphData):
        # ONNX GRAPH-type subgraph attribute.
        return "\n" + str_from_graph_data(
            val,
            show_layers=show_layers,
            show_attrs=show_attrs,
            show_weights=show_weights,
            indent_level=indent_level + 2,
        )
    if isinstance(val, TensorAttrValue):
        s = f"Tensor: [dtype={val.dtype}, shape={val.shape}]"
        if show_weights and val.values is not None:
            s += " | Values:\n" + util.indent_block(str(val.values))
        return s
    return val  # Returned as-is; f-string coerces to str.


def str_from_graph_data(
    graph_data,
    show_layers=False,
    show_attrs=False,
    show_weights=False,
    indent_level=0,
):
    """
    Convert a ``GraphData`` into a human-readable string.

    This is the single canonical text formatter used by all backends.

    Args:
        graph_data (GraphData): The graph to format.
        show_layers (bool): Display per-layer/node information.
        show_attrs (bool): Display per-layer attributes (requires show_layers).
        show_weights (bool): Display weight values.
        indent_level (int): Number of indentation levels to apply.

    Returns:
        str
    """
    mt = graph_data.model_type
    is_trt = mt in ("trt_network", "trt_engine")
    section_prefix = (
        "Subgraph" if graph_data.is_subgraph else _SECTION_PREFIX.get(mt, "Graph")
    )
    node_prefix = _NODE_PREFIX.get(mt, "Node")

    out = ""

    # ------------------------------------------------------------------ header
    if graph_data.title:
        out += graph_data.title + "\n\n"

    # ------------------------------------------------------------ doc string
    if show_attrs and graph_data.doc_string:
        out += f"---- Docstring ----\n{graph_data.doc_string}\n\n"

    # ------------------------------------------------------------ inputs
    input_meta = _build_tensor_metadata(graph_data.graph_inputs)
    out += f"---- {len(input_meta)} {section_prefix} Input(s) ----\n{input_meta}\n\n"

    # ------------------------------------------------------------ outputs
    output_meta = _build_tensor_metadata(graph_data.graph_outputs)
    out += f"---- {len(output_meta)} {section_prefix} Output(s) ----\n{output_meta}\n\n"

    # ------------------------------------------------------------ TRT engine extras
    if mt == "trt_engine":
        out += f"---- Memory ----\nDevice Memory: {graph_data.device_memory_bytes} bytes\n\n"

        num_profiles = len(graph_data.profiles)
        out += (
            f"---- {num_profiles} Profile(s) "
            f"({graph_data.num_io_tensors} Tensor(s) Each) ----\n"
        )
        for profile in graph_data.profiles:
            out += f"- Profile: {profile.index}\n"
            max_width = (
                max((len(ti.name) for ti in profile.tensor_infos), default=0) + 8
            )
            for ti in profile.tensor_infos:
                io_label = " (Input)" if ti.is_input else "(Output)"
                out += util.indent_block(
                    f"Tensor: {ti.name:<{max_width}} {io_label}, Index: {ti.tensor_index}"
                )
                if ti.is_input:
                    out += (
                        f" | Shapes: min={ti.min_shape}, "
                        f"opt={ti.opt_shape}, max={ti.max_shape}\n"
                    )
                else:
                    out += f" | Shape: {ti.shape}\n"
            out += "\n"

    # ------------------------------------------------ initializers (ONNX only)
    if mt == "onnx":
        init_count = len(graph_data.initializers)
        out += f"---- {init_count} Initializer(s) ----\n"
        if show_weights:
            for ti in graph_data.initializers:
                out += (
                    f"Initializer | {ti.name} "
                    f"[dtype={ti.dtype}, shape={ti.shape}] | Values:\n"
                    f"{util.indent_block(str(ti.values))}\n\n"
                )
            if not graph_data.initializers:
                out += "{}\n\n"
        elif show_layers:
            # In the initializer section, don't prefix names with "Initializer | "
            # since the section header already makes their role clear.
            init_meta = _build_tensor_metadata(
                graph_data.initializers, prefix_initializers=False
            )
            out += str(init_meta) + "\n\n"
        else:
            out += "\n"

    # ------------------------------------------------------------ nodes/layers
    if mt == "trt_engine":
        # TRT engine: per-profile node lists.
        num_layers = graph_data.num_layers
        num_profiles_to_show = len(graph_data.per_profile_nodes)
        per_profile = num_profiles_to_show > 1
        profile_label = " Per Profile" if per_profile else ""
        out += f"---- {num_layers} Layer(s){profile_label} ----\n"

        if show_layers:
            for profile_idx, profile_nodes in enumerate(graph_data.per_profile_nodes):
                indent = 0
                if per_profile:
                    out += f"- Profile: {profile_idx}\n"
                    indent = 1
                for index, node in enumerate(profile_nodes):
                    input_names, input_meta = _node_names_and_meta(node.inputs)
                    output_names, output_meta = _node_names_and_meta(node.outputs)
                    out += (
                        util.indent_block(
                            util.str_from_layer(
                                node_prefix,
                                index,
                                node.name,
                                node.op_type,
                                input_names,
                                input_meta,
                                output_names,
                                output_meta,
                            ),
                            indent,
                        )
                        + "\n"
                    )
                    if show_attrs:
                        out += (
                            util.indent_block("---- Attributes ----", indent + 1) + "\n"
                        )
                        out += (
                            util.indent_block(f"Origin = {node.origin}", indent + 1)
                            + "\n"
                        )
                        out += (
                            util.indent_block(f"Tactic = {node.tactic}", indent + 1)
                            + "\n"
                        )
                    out += "\n"
    else:
        # ONNX / TRT network: flat node list.
        node_count = len(graph_data.nodes)
        out += f"---- {node_count} {node_prefix}(s) ----\n"

        if show_layers:
            for index, node in enumerate(graph_data.nodes):
                input_names, input_meta = _node_names_and_meta(
                    node.inputs, prefix_initializers=True
                )
                output_names, output_meta = _node_names_and_meta(
                    node.outputs, prefix_initializers=True
                )

                out += util.str_from_layer(
                    node_prefix,
                    index,
                    node.name,
                    node.op_type,
                    input_names,
                    input_meta,
                    output_names,
                    output_meta,
                )

                if show_attrs and node.attrs:
                    out += util.indent_block("---- Attributes ----") + "\n"
                    for key, val in node.attrs.items():
                        rendered = _format_attr_value(
                            val,
                            show_layers,
                            show_attrs,
                            show_weights,
                            indent_level,
                            is_trt,
                        )
                        if rendered is None:
                            continue
                        attr_prefix = f"{node.name}." if node.name else ""
                        out += (
                            util.indent_block(f"{attr_prefix}{key} = {rendered}") + "\n"
                        )
                out += "\n"

    return util.indent_block(out, indent_level)
