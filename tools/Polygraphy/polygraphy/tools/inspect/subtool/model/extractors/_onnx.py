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
"""ONNX-specific GraphData extractor."""
from collections import OrderedDict

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    NodeInfo,
    TensorAttrValue,
    TensorInfo,
)

onnx = mod.lazy_import("onnx")

# ONNX attribute names whose INT value encodes a TensorProto.DataType.
_ONNX_DTYPE_ATTR_NAMES = {"to", "dtype", "target_type"}
# onnx.TensorProto.DataType.Name() returns "FLOAT" / "INT32" etc.
# Map the non-obvious ones to NumPy-style names; others are just lowercased.
_ONNX_DTYPE_NAME_OVERRIDES = {"FLOAT": "float32", "DOUBLE": "float64"}

# Maps ONNX attribute type name -> proto field accessor name.
_ONNX_PYTHON_ATTR_MAPPING = {
    "FLOAT": "f",
    "INT": "i",
    "STRING": "s",
    "TENSOR": "t",
    "GRAPH": "g",
    "FLOATS": "floats",
    "INTS": "ints",
    "STRINGS": "strings",
}


def _onnx_dtype_int_to_str(val):
    """Convert an ONNX TensorProto.DataType integer to a human-readable string."""
    try:
        name = onnx.TensorProto.DataType.Name(val)
    except (ValueError, KeyError):
        return val
    return _ONNX_DTYPE_NAME_OVERRIDES.get(name, name.lower())


def graph_data_from_onnx(model, show_weights=False):
    """
    Build a ``GraphData`` from an ONNX ``ModelProto``.

    Args:
        model: An ``onnx.ModelProto``.
        show_weights (bool):
            When True, eagerly load initializer and TENSOR-attribute values so
            that ``str_from_graph_data`` and the visual viewer can display them.

    Returns:
        GraphData
    """
    default_opset = "Unknown"
    other_opsets = {}
    for info in model.opset_import:
        if not info.domain:
            default_opset = info.version
        else:
            other_opsets[info.domain] = info.version

    title = f"Name: {model.graph.name} | ONNX Opset: {default_opset}"
    if other_opsets:
        title += f" | Other Opsets: {other_opsets}"

    return _graph_data_from_onnx_graph(
        model.graph,
        parent_tensors={},
        is_subgraph=False,
        title=title,
        show_weights=show_weights,
    )


def _graph_data_from_onnx_graph(
    graph, parent_tensors, is_subgraph, title, show_weights
):
    """Recursively extract a single ONNX graph (or subgraph) into a ``GraphData``."""
    from polygraphy.backend.onnx.util import (
        get_dtype,
        get_input_metadata,
        get_output_metadata,
        get_shape,
        get_tensor_metadata,
        get_values,
    )
    from polygraphy.tools.inspect.subtool.model.extractors import (
        _build_edges,
        _meta_to_tensor_infos,
    )

    input_metadata = get_input_metadata(graph)
    output_metadata = get_output_metadata(graph)
    initializer_metadata = get_tensor_metadata(graph.initializer)

    # Merge all available tensor metadata for dtype/shape lookups during node processing.
    tensors = dict(parent_tensors)
    tensors.update(get_tensor_metadata(graph.value_info))
    tensors.update(initializer_metadata)
    tensors.update(input_metadata)
    tensors.update(output_metadata)

    graph_inputs = _meta_to_tensor_infos(input_metadata)
    graph_outputs = _meta_to_tensor_infos(output_metadata)

    # ---- Initializers --------------------------------------------------
    initializers = [
        TensorInfo(
            name=init.name,
            dtype=get_dtype(init),
            shape=get_shape(init),
            is_initializer=True,
            values=get_values(init) if show_weights else None,
        )
        for init in graph.initializer
    ]

    # Build a name→values map so we can attach weight values to node inputs.
    init_values_map = {init.name: init.values for init in initializers}

    # ---- ONNX attribute helpers ----------------------------------------
    _ATTR_TYPE_MAPPING = dict(
        zip(
            onnx.AttributeProto.AttributeType.values(),
            onnx.AttributeProto.AttributeType.keys(),
        )
    )

    def _process_attr(attr):
        """Return a Python value for an ONNX attribute proto, or None to skip."""
        if attr.type not in _ATTR_TYPE_MAPPING:
            G_LOGGER.warning(
                f"Attribute type {attr.type} was not recognized. Skipping attribute."
            )
            return None
        attr_str = _ATTR_TYPE_MAPPING[attr.type]
        if attr_str not in _ONNX_PYTHON_ATTR_MAPPING:
            G_LOGGER.warning(
                f"Attribute of type {attr_str} is currently unsupported. Skipping attribute."
            )
            return None
        raw = getattr(attr, _ONNX_PYTHON_ATTR_MAPPING[attr_str])
        if attr_str == "STRING":
            return raw.decode()
        if attr_str == "TENSOR":
            return TensorAttrValue(
                dtype=get_dtype(raw),
                shape=get_shape(raw),
                values=get_values(raw) if show_weights else None,
            )
        if attr_str == "GRAPH":
            return _graph_data_from_onnx_graph(
                raw,
                parent_tensors=tensors,
                is_subgraph=True,
                title="",
                show_weights=show_weights,
            )
        if attr_str in ("FLOATS", "INTS"):
            return list(raw)
        if attr_str == "STRINGS":
            return [s.decode() for s in raw]
        # For INT attributes that encode a data type, convert to a readable name.
        if attr_str == "INT" and attr.name in _ONNX_DTYPE_ATTR_NAMES:
            return _onnx_dtype_int_to_str(raw)
        return raw  # FLOAT or INT

    # ---- Nodes ---------------------------------------------------------
    def _tensor_infos_for_names(names):
        """Look up dtype/shape for a list of tensor names from the merged tensors dict."""
        result = []
        for name in names:
            meta = tensors.get(name)
            is_init = name in initializer_metadata
            result.append(
                TensorInfo(
                    name=name,
                    dtype=meta.dtype if meta is not None else None,
                    shape=(
                        list(meta.shape)
                        if (meta is not None and meta.shape is not None)
                        else None
                    ),
                    is_initializer=is_init,
                    values=init_values_map.get(name) if is_init else None,
                )
            )
        return result

    nodes = []
    for index, node in enumerate(graph.node):
        attrs = OrderedDict()
        for attr in node.attribute:
            val = _process_attr(attr)
            if val is not None:
                attrs[attr.name] = val

        nodes.append(
            NodeInfo(
                node_id=f"node_{index}",
                name=node.name,
                op_type=node.op_type,
                inputs=_tensor_infos_for_names(node.input),
                outputs=_tensor_infos_for_names(node.output),
                attrs=attrs,
            )
        )

    return GraphData(
        title=title,
        model_type="onnx",
        is_subgraph=is_subgraph,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=nodes,
        edges=_build_edges(nodes, graph_inputs, graph_outputs),
        initializers=initializers,
        doc_string=graph.doc_string or None,
    )
