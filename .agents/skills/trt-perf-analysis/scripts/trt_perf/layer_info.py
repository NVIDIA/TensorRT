# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process TensorRT layer-info JSON and write visualization targets."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STANDARD_LAYER_KEYS = {
    "Name",
    "LayerType",
    "Inputs",
    "Constants",
    "Outputs",
    "TacticName",
    "StreamId",
    "Metadata",
    "_subgraph",
}


@dataclass(frozen=True)
class ProcessedInitializer:
    name: str
    desc: dict[str, Any]


@dataclass(frozen=True)
class ProcessedNode:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class ProcessedLayerInfo:
    source_path: Path
    layers: list[dict[str, Any]]
    nodes: list[ProcessedNode]
    graph_inputs: list[str]
    graph_outputs: list[str]
    graph_value_info: list[str]
    value_descriptors: dict[str, dict[str, Any]]
    initializers: list[ProcessedInitializer]
    layer_type_counts: Counter[str]
    node_op_type_counts: Counter[str]
    extra_layer_fields: list[str]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    module_name: str
    default_suffix: str
    description: str
    aliases: tuple[str, ...] = ()


TARGET_SPECS = (
    TargetSpec(
        name="html",
        module_name="trt_perf.layer_info_html",
        default_suffix=".html",
        description="Standalone HTML layer graph summary",
    ),
)

TARGETS_BY_NAME: dict[str, TargetSpec] = {}
for target_spec in TARGET_SPECS:
    TARGETS_BY_NAME[target_spec.name] = target_spec
    for alias in target_spec.aliases:
        TARGETS_BY_NAME[alias] = target_spec


class NameRegistry:
    """Keep target value names valid under SSA while preserving TRT names."""

    def __init__(self) -> None:
        self.used: set[str] = set()

    def reserve(self, preferred: Any, fallback: str) -> str:
        base = str(preferred).strip() if preferred is not None else ""
        if not base:
            base = fallback

        candidate = base
        suffix = 1
        while candidate in self.used:
            candidate = f"{base}__trt_dup{suffix}"
            suffix += 1
        self.used.add(candidate)
        return candidate


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_layers(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        layers = payload
    elif isinstance(payload, dict):
        for key in ("Layers", "layers", "LayerInfo", "layerInfo"):
            if isinstance(payload.get(key), list):
                layers = payload[key]
                break
        else:
            raise ValueError("JSON object does not contain a layer list")
    else:
        raise ValueError("top-level JSON value must be a list or object")

    bad_indexes = [idx for idx, layer in enumerate(layers) if not isinstance(layer, dict)]
    if bad_indexes:
        preview = ", ".join(str(idx) for idx in bad_indexes[:5])
        raise ValueError(f"layer entries must be objects; invalid indexes: {preview}")

    return layers


def as_descriptor_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def tensor_name(desc: dict[str, Any], fallback: str) -> str:
    name = desc.get("Name")
    if name is None or str(name).strip() == "":
        return fallback
    return str(name)


def safe_dim_param(text: Any, fallback: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_")
    if not value:
        value = fallback
    if value[0].isdigit():
        value = f"d_{value}"
    return value


def tensor_shape(desc: dict[str, Any], tensor: str) -> list[int | str] | None:
    dims = desc.get("Dimensions")
    if dims is None:
        return None
    if not isinstance(dims, list):
        return None

    shape: list[int | str] = []
    prefix = safe_dim_param(tensor, "tensor")
    for idx, dim in enumerate(dims):
        if isinstance(dim, bool):
            shape.append(int(dim))
        elif isinstance(dim, int) and dim >= 0:
            shape.append(dim)
        else:
            shape.append(f"{prefix}_dim{idx}")
    return shape


def sanitize_op_type(layer_type: Any) -> str:
    text = str(layer_type).strip() if layer_type is not None else ""
    op_type = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not op_type:
        op_type = "Layer"
    if not re.match(r"[A-Za-z_]", op_type[0]):
        op_type = f"_{op_type}"
    return op_type[:200]


KGEN_TOKEN_MAP = {
    "and": "And",
    "add": "Add",
    "cast": "Cast",
    "div": "Div",
    "erf": "Erf",
    "gath": "Gather",
    "gemm": "Gemm",
    "mean": "Mean",
    "mha": "MHA",
    "move": "Move",
    "mul": "Mul",
    "resh": "Reshape",
    "sele": "Select",
    "slic": "Slice",
    "sqrt": "Sqrt",
    "sub": "Sub",
    "tanh": "Tanh",
    "tran": "Transpose",
}


def strip_kgen_signature_noise(text: str) -> str:
    text = re.sub(r"_0x[0-9A-Fa-f]+$", "", text)
    text = re.sub(r"_myl\d+(?:_\d+)?$", "", text)
    text = re.sub(r"^__?myl_", "", text)
    return text.strip("_")


def split_kgen_signature(signature: str) -> list[str]:
    if not signature:
        return []
    if "_" in signature:
        return [part for part in signature.split("_") if part]
    return re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", signature)


def normalize_kgen_token(token: str) -> str:
    lowered = token.lower()
    if lowered in KGEN_TOKEN_MAP:
        return KGEN_TOKEN_MAP[lowered]
    if lowered.startswith("v") and lowered[1:].isdigit():
        return lowered.upper()
    return token[:1].upper() + token[1:]


def kgen_details(layer: dict[str, Any]) -> tuple[str, list[str], str]:
    candidates = [str(layer.get("TacticName", "")), layer_name(layer, 0)]
    for candidate in candidates:
        signature = strip_kgen_signature_noise(candidate)
        tokens = [normalize_kgen_token(token) for token in split_kgen_signature(signature)]
        if tokens:
            return signature, tokens, f"KGEN_{'_'.join(tokens)}"
    return "", [], "kgen"


def node_op_type(layer: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_layer_type = layer.get("LayerType", "Layer")
    if str(raw_layer_type).lower() != "kgen":
        return sanitize_op_type(raw_layer_type), {}

    signature, tokens, op_type = kgen_details(layer)
    attrs: dict[str, Any] = {}
    if signature:
        attrs["trt_kgen_signature"] = signature
    if tokens:
        attrs["trt_kgen_ops_json"] = compact_json(tokens)
        attrs["trt_kgen_op_count"] = len(tokens)
    return sanitize_op_type(op_type), attrs


def layer_name(layer: dict[str, Any], index: int) -> str:
    raw = layer.get("Name")
    if raw is None or str(raw).strip() == "":
        return f"trt_layer_{index}"
    return str(raw)


def int_attr(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def process_layers(layers: list[dict[str, Any]], source_path: Path) -> ProcessedLayerInfo:
    registry = NameRegistry()
    external_by_original: dict[str, str] = {}
    latest_by_original: dict[str, str] = {}
    descriptors_by_value: dict[str, dict[str, Any]] = {}
    initializers_by_original: dict[str, str] = {}
    initializers: list[ProcessedInitializer] = []
    produced_values: list[str] = []
    consumed_values: set[str] = set()
    nodes: list[ProcessedNode] = []

    def map_external(desc: dict[str, Any], fallback: str) -> str:
        original = tensor_name(desc, fallback)
        if original not in external_by_original:
            value_name = registry.reserve(original, fallback)
            external_by_original[original] = value_name
            descriptors_by_value[value_name] = desc
        return external_by_original[original]

    def map_input(desc: dict[str, Any], fallback: str) -> str:
        original = tensor_name(desc, fallback)
        value_name = latest_by_original.get(original)
        if value_name is None:
            value_name = map_external(desc, fallback)
        descriptors_by_value.setdefault(value_name, desc)
        consumed_values.add(value_name)
        return value_name

    def map_constant(desc: dict[str, Any], fallback: str) -> str:
        original = tensor_name(desc, fallback)
        if original not in initializers_by_original:
            value_name = registry.reserve(original, fallback)
            initializers_by_original[original] = value_name
            descriptors_by_value[value_name] = desc
            initializers.append(ProcessedInitializer(value_name, desc))
        value_name = initializers_by_original[original]
        descriptors_by_value.setdefault(value_name, desc)
        consumed_values.add(value_name)
        return value_name

    def map_output(desc: dict[str, Any], fallback: str) -> str:
        original = tensor_name(desc, fallback)
        value_name = registry.reserve(original, fallback)
        latest_by_original[original] = value_name
        descriptors_by_value[value_name] = desc
        produced_values.append(value_name)
        return value_name

    layer_type_counts: Counter[str] = Counter()
    node_op_type_counts: Counter[str] = Counter()
    unmapped_layer_fields: set[str] = set()

    for index, layer in enumerate(layers):
        inputs = as_descriptor_list(layer.get("Inputs"))
        constants = as_descriptor_list(layer.get("Constants"))
        outputs = as_descriptor_list(layer.get("Outputs"))

        node_inputs = [
            map_input(desc, f"trt_layer_{index}_input_{pos}")
            for pos, desc in enumerate(inputs)
        ]
        node_inputs.extend(
            map_constant(desc, f"trt_layer_{index}_constant_{pos}")
            for pos, desc in enumerate(constants)
        )
        node_outputs = [
            map_output(desc, f"trt_layer_{index}_output_{pos}")
            for pos, desc in enumerate(outputs)
        ]

        raw_layer_type = layer.get("LayerType", "Layer")
        layer_type_counts[str(raw_layer_type)] += 1
        op_type, op_type_attrs = node_op_type(layer)
        node_op_type_counts[op_type] += 1
        extra = {key: value for key, value in layer.items() if key not in STANDARD_LAYER_KEYS}
        unmapped_layer_fields.update(extra)

        attrs: dict[str, Any] = {
            "trt_layer_index": index,
            "trt_layer_type": str(raw_layer_type),
            "trt_input_count": len(inputs),
            "trt_constant_count": len(constants),
            "trt_output_count": len(outputs),
            "trt_inputs_json": compact_json(inputs),
            "trt_constants_json": compact_json(constants),
            "trt_outputs_json": compact_json(outputs),
        }
        attrs.update(op_type_attrs)

        if "TacticName" in layer:
            attrs["trt_tactic_name"] = str(layer.get("TacticName", ""))
        if "StreamId" in layer:
            attrs["trt_stream_id"] = int_attr(layer.get("StreamId"))
        if "_subgraph" in layer:
            attrs["trt_subgraph"] = int_attr(layer.get("_subgraph"))
        if "Metadata" in layer:
            attrs["trt_metadata"] = str(layer.get("Metadata", ""))
        if extra:
            attrs["trt_extra_json"] = compact_json(extra)

        nodes.append(
            ProcessedNode(
                name=layer_name(layer, index),
                op_type=op_type,
                inputs=node_inputs,
                outputs=node_outputs,
                attrs=attrs,
            )
        )

    terminal_values = [value_name for value_name in produced_values if value_name not in consumed_values]
    if not terminal_values and produced_values:
        terminal_values = [produced_values[-1]]

    graph_output_names = set(terminal_values)
    graph_input_names = set(external_by_original.values())
    graph_value_info = [
        value_name
        for value_name in descriptors_by_value
        if value_name not in graph_input_names and value_name not in graph_output_names
    ]

    return ProcessedLayerInfo(
        source_path=source_path,
        layers=layers,
        nodes=nodes,
        graph_inputs=list(external_by_original.values()),
        graph_outputs=terminal_values,
        graph_value_info=graph_value_info,
        value_descriptors=descriptors_by_value,
        initializers=initializers,
        layer_type_counts=layer_type_counts,
        node_op_type_counts=node_op_type_counts,
        extra_layer_fields=sorted(unmapped_layer_fields),
    )


def load_and_process_layer_info(path: Path) -> ProcessedLayerInfo:
    return process_layers(load_layers(path), path)


def resolve_target_spec(target: str) -> TargetSpec:
    try:
        return TARGETS_BY_NAME[target]
    except KeyError as exc:
        valid = ", ".join(sorted(TARGETS_BY_NAME))
        raise ValueError(f"unknown target {target!r}; valid targets: {valid}") from exc


def default_output_path(input_path: Path, target: str = "html") -> Path:
    spec = resolve_target_spec(target)
    return input_path.with_name(f"{input_path.stem}{spec.default_suffix}")


def target_names_help() -> str:
    return ", ".join(spec.name for spec in TARGET_SPECS)


def load_target_module(spec: TargetSpec) -> Any:
    return importlib.import_module(spec.module_name)


def write_target(processed: ProcessedLayerInfo, spec: TargetSpec, output_path: Path) -> None:
    module = load_target_module(spec)
    writer = getattr(module, "write", None)
    if writer is None:
        raise NotImplementedError(
            f"target {spec.name!r} is not implemented yet; "
            f"expected write(processed, output_path) in {spec.module_name}"
        )
    writer(processed, output_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process TensorRT layer-info JSON and write one or more visualization targets."
    )
    parser.add_argument("input", type=Path, nargs="?", help="Path to TensorRT layer-info JSON")
    parser.add_argument(
        "-t",
        "--target",
        action="append",
        dest="targets",
        metavar="TARGET",
        help=f"Output target; repeat for multiple outputs. Default: html. Available: {target_names_help()}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Only valid when writing one target.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for target-specific default filenames.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List supported target names and exit.",
    )
    return parser.parse_args(argv)


def list_targets() -> None:
    seen: set[str] = set()
    for spec in TARGET_SPECS:
        aliases = f" (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
        print(f"{spec.name}{aliases}: {spec.description}")
        seen.add(spec.name)

    alias_only = sorted(name for name, spec in TARGETS_BY_NAME.items() if spec.name not in seen)
    if alias_only:
        print(f"aliases: {', '.join(alias_only)}")


def resolve_requested_targets(targets: list[str] | None) -> list[TargetSpec]:
    requested = targets or ["html"]
    specs: list[TargetSpec] = []
    seen: set[str] = set()
    for target in requested:
        spec = resolve_target_spec(target)
        if spec.name in seen:
            continue
        specs.append(spec)
        seen.add(spec.name)
    return specs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_targets:
        list_targets()
        return 0
    if args.input is None:
        print("error: input is required unless --list-targets is used", file=sys.stderr)
        return 1

    try:
        target_specs = resolve_requested_targets(args.targets)
        if args.output is not None and len(target_specs) != 1:
            raise ValueError("--output can only be used when writing one target")

        processed = load_and_process_layer_info(args.input)
        written: list[Path] = []
        for spec in target_specs:
            if args.output is not None:
                output_path = args.output
            elif args.output_dir is not None:
                output_path = args.output_dir / f"{args.input.stem}{spec.default_suffix}"
            else:
                output_path = default_output_path(args.input, spec.name)

            write_target(processed, spec, output_path)
            written.append(output_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for output_path in written:
        print(f"wrote {output_path}")
    return 0
