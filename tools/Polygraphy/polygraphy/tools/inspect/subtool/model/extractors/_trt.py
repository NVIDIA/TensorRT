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
"""TensorRT-specific GraphData extractors (network and engine)."""
import json
from collections import OrderedDict

from polygraphy import mod
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    NodeInfo,
    ProfileInfo,
    ProfileTensorInfo,
    TensorInfo,
)

trt = mod.lazy_import("tensorrt")

# Maps TRT JSON format strings to DataType values.
# Keys are checked via ``key in format_string.upper()``, so order matters:
# more-specific keys must come before shorter ones that are substrings of them
# (e.g. "INT32" before "INT" to avoid INT32 matching "INT" first).
_TRT_DTYPE_MAP = {
    "BFLOAT16": DataType.BFLOAT16,
    "FLOAT8E4M3": DataType.FLOAT8E4M3FN,
    "FLOAT8E5M2": DataType.FLOAT8E5M2,
    "FLOAT4": DataType.FLOAT4,
    "FLOAT": DataType.FLOAT32,
    "HALF": DataType.FLOAT16,
    "FP32": DataType.FLOAT32,
    "FP16": DataType.FLOAT16,
    "INT64": DataType.INT64,
    "INT32": DataType.INT32,
    "INT16": DataType.INT16,
    "INT8": DataType.INT8,
    "INT4": DataType.INT4,
    "UINT8": DataType.UINT8,
    "BOOL": DataType.BOOL,
}


def graph_data_from_trt_network(network, show_weights=False):
    """
    Build a ``GraphData`` from a TensorRT ``INetworkDefinition``.

    Args:
        network: A ``trt.INetworkDefinition``.
        show_weights (bool): When True, include numpy-array weight attributes.

    Returns:
        GraphData
    """
    from polygraphy.backend.trt.util import (
        get_layer_attribute_names,
        get_layer_class_mapping,
        get_layer_input_names_meta,
        get_layer_output_names_meta,
        get_network_input_names_meta,
        get_network_output_names_meta,
    )
    from polygraphy.tools.inspect.subtool.model.extractors import (
        _build_edges,
        _meta_to_tensor_infos,
    )

    implicit = (
        hasattr(network, "has_implicit_batch_dimension")
        and network.has_implicit_batch_dimension
    )
    strongly_typed = hasattr(network, "get_flag") and network.get_flag(
        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
    )
    batch_str = "Implicit" if implicit else "Explicit"
    typed_str = " Strongly Typed" if strongly_typed else ""
    title = f"Name: {network.name} | {batch_str} Batch{typed_str} Network"

    _, input_meta = get_network_input_names_meta(network)
    _, output_meta = get_network_output_names_meta(network)

    graph_inputs = _meta_to_tensor_infos(input_meta)
    graph_outputs = _meta_to_tensor_infos(output_meta)

    LAYER_TYPE_CLASS_MAPPING = get_layer_class_mapping()

    def _layer_tensor_infos(names, meta):
        return [
            TensorInfo(
                name=n,
                dtype=meta[n].dtype,
                shape=list(meta[n].shape) if meta[n].shape is not None else None,
            )
            for n in names
        ]

    nodes = []
    for index, layer in enumerate(network):
        if layer.type in LAYER_TYPE_CLASS_MAPPING:
            layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

        input_names, input_meta_layer = get_layer_input_names_meta(layer)
        output_names, output_meta_layer = get_layer_output_names_meta(layer)

        attrs = OrderedDict()
        for attr_name in get_layer_attribute_names(layer):
            with G_LOGGER.verbosity():
                try:
                    val = getattr(layer, attr_name)
                except Exception as err:
                    val = f"<Error: could not retrieve attribute '{attr_name}': {err}>"
            attrs[attr_name] = val

        nodes.append(
            NodeInfo(
                node_id=f"node_{index}",
                name=layer.name,
                op_type=str(layer.type),
                inputs=_layer_tensor_infos(input_names, input_meta_layer),
                outputs=_layer_tensor_infos(output_names, output_meta_layer),
                attrs=attrs,
            )
        )

    return GraphData(
        title=title,
        model_type="trt_network",
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=nodes,
        edges=_build_edges(nodes, graph_inputs, graph_outputs),
        num_layers=network.num_layers,
    )


def graph_data_from_trt_engine(
    engine, context, show_weights=False, combine_tensor_info=None
):
    """
    Build a ``GraphData`` from a TensorRT ``ICudaEngine``.

    Args:
        engine: A ``trt.ICudaEngine``.
        context: The active execution context.
        show_weights (bool): Unused for engines; kept for API consistency.
        combine_tensor_info (str): Optional path to a tensor JSON file.

    Returns:
        GraphData
    """
    from polygraphy.backend.trt.util import (
        TensorInfo as TrtTensorInfo,
        get_metadata_from_engine,
    )
    from polygraphy.tools.inspect.subtool.model.extractors import (
        _build_edges,
        _meta_to_tensor_infos,
    )

    implicit = (
        hasattr(engine, "has_implicit_batch_dimension")
        and engine.has_implicit_batch_dimension
    )
    refittable = "Refittable " if engine.refittable else ""
    batch_str = "Implicit" if implicit else "Explicit"
    title = f"Name: {engine.name} | {refittable}{batch_str} Batch Engine"

    num_io_tensors = engine.num_io_tensors
    input_meta = get_metadata_from_engine(engine, context, mode=trt.TensorIOMode.INPUT)
    output_meta = get_metadata_from_engine(
        engine, context, mode=trt.TensorIOMode.OUTPUT
    )

    graph_inputs = _meta_to_tensor_infos(input_meta)
    graph_outputs = _meta_to_tensor_infos(output_meta)

    # ---- Profiles ------------------------------------------------------
    profiles = []
    for profile_index in range(engine.num_optimization_profiles):
        tensor_infos = []
        for idx in range(num_io_tensors):
            name = engine.get_tensor_name(idx)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                min_s, opt_s, max_s = engine.get_tensor_profile_shape(
                    name, profile_index
                )
                tensor_infos.append(
                    ProfileTensorInfo(
                        name=name,
                        tensor_index=idx,
                        is_input=True,
                        min_shape=tuple(min_s),
                        opt_shape=tuple(opt_s),
                        max_shape=tuple(max_s),
                    )
                )
            else:
                tensor_infos.append(
                    ProfileTensorInfo(
                        name=name,
                        tensor_index=idx,
                        is_input=False,
                        shape=tuple(engine.get_tensor_shape(name)),
                    )
                )
        profiles.append(ProfileInfo(index=profile_index, tensor_infos=tensor_infos))

    # ---- Per-profile layers --------------------------------------------
    layers_per_profile = engine.num_layers // max(engine.num_optimization_profiles, 1)
    per_profile_nodes = []

    try:
        inspector = engine.create_engine_inspector()
    except AttributeError:
        G_LOGGER.warning(
            f"IEngineInspector is not available in TensorRT {trt.__version__}; "
            "layer information will be omitted."
        )
        inspector = None

    if inspector is not None:
        inspector.execution_context = context
        tensor_stats_helper = TrtTensorInfo(combine_tensor_info)

        def _infos_from_inspector_key(layer_info, key):
            """Convert one inspector JSON tensor list (Inputs or Outputs) to TensorInfo objects."""
            result = []
            for elem in layer_info.get(key) or []:
                # TensorRT 11 split the combined "Format/Datatype" field into
                # separate "Datatype" and "Format" keys.
                if "Format/Datatype" in elem:
                    fmt_dtype = elem["Format/Datatype"]
                else:
                    fmt_dtype = " ".join(
                        elem[k] for k in ("Datatype", "Format") if elem.get(k)
                    )
                dtype = next(
                    (v for k, v in _TRT_DTYPE_MAP.items() if k in fmt_dtype.upper()),
                    None,
                )
                stats = tensor_stats_helper.get_tensor_statistics(elem["Name"])
                docstring = (
                    f"Format: {fmt_dtype}"
                    if fmt_dtype and "N/A" not in fmt_dtype
                    else ""
                ) + stats
                result.append(
                    TensorInfo(
                        name=elem["Name"],
                        dtype=dtype,
                        shape=elem["Dimensions"],
                        docstring=docstring or None,
                    )
                )
            return result

        # TRT 10+ does not repeat layer info per profile.
        num_profiles_to_extract = (
            1
            if mod.version(trt.__version__) >= mod.version("10")
            else engine.num_optimization_profiles
        )

        for profile_idx in range(num_profiles_to_extract):
            profile_nodes = []
            offset = profile_idx * layers_per_profile

            for index in range(layers_per_profile):
                layer_info = json.loads(
                    inspector.get_layer_information(
                        offset + index, trt.LayerInformationFormat.JSON
                    )
                )

                if engine.profiling_verbosity == trt.ProfilingVerbosity.DETAILED:
                    name = layer_info.get("Name", "Unknown")
                    op = layer_info.get("LayerType", "Unknown")
                    origin = layer_info.get("Origin", "Unknown")
                    tactic = layer_info.get(
                        "TacticValue", layer_info.get("TacticName", "Unknown")
                    )
                    input_tis = _infos_from_inspector_key(layer_info, "Inputs")
                    output_tis = _infos_from_inspector_key(layer_info, "Outputs")
                else:
                    G_LOGGER.warning(
                        f"Engine profiling verbosity: {engine.profiling_verbosity}. "
                        "Some layer info may be missing. Set higher verbosity for more detail.",
                        mode=LogMode.ONCE,
                    )
                    name, op, origin, tactic = (
                        str(layer_info),
                        "Unknown",
                        "Unknown",
                        "Unknown",
                    )
                    input_tis, output_tis = [], []

                profile_nodes.append(
                    NodeInfo(
                        node_id=f"p{profile_idx}_node_{index}",
                        name=name,
                        op_type=op,
                        inputs=input_tis,
                        outputs=output_tis,
                        attrs=OrderedDict(),
                        origin=origin,
                        tactic=tactic,
                    )
                )
            per_profile_nodes.append(profile_nodes)

    first_nodes = per_profile_nodes[0] if per_profile_nodes else []

    return GraphData(
        title=title,
        model_type="trt_engine",
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=first_nodes,
        edges=_build_edges(first_nodes, graph_inputs, graph_outputs),
        device_memory_bytes=(
            # device_memory_size was renamed to device_memory_size_v2 in TRT 11.
            engine.device_memory_size_v2
            if hasattr(engine, "device_memory_size_v2")
            else engine.device_memory_size
        ),
        num_io_tensors=num_io_tensors,
        profiles=profiles,
        per_profile_nodes=per_profile_nodes,
        num_layers=layers_per_profile,
    )
