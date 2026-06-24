# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract and analyze TensorRT layer/profile performance data."""

from __future__ import annotations

import collections
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


Number = (int, float)
DYNAMIC_TOKENS = (
    "shape",
    "cast_hvar",
    "iota",
    "slic",
    "conc",
    "eql",
    "gtr",
    "orand",
    "repl",
    "sele",
)
MISC_LAYER_TYPES = {
    "reshape",
    "shape_call",
    "signal",
    "wait",
    "signal_and_wait",
    "signal_wait",
}
TENSOR_CORE_TOKENS = ("xmma", "tensor", "tc", "cublaslt", "mma")
MHA_TOKENS = (
    "kgen_mha",
    "fmha",
    "multi_head_attention",
    "multiheadattention",
    "multi-head attention",
    "multi head attention",
    "fused_multihead_attention",
    "masked_multihead_attention",
)
MODEL_NAME_STOPWORDS = {
    "analysis",
    "analyze",
    "aot",
    "backend",
    "backends",
    "benchmark",
    "benchmarks",
    "cache",
    "cached",
    "case",
    "checkpoint",
    "checkpoints",
    "ckpt",
    "component",
    "components",
    "data",
    "dataset",
    "datasets",
    "decode",
    "decoder",
    "default",
    "dev",
    "development",
    "download",
    "downloads",
    "encode",
    "encoder",
    "engine",
    "engines",
    "example",
    "examples",
    "export",
    "exports",
    "full",
    "input",
    "inputs",
    "json",
    "layer",
    "layers",
    "model",
    "models",
    "no",
    "none",
    "onnx",
    "only",
    "output",
    "outputs",
    "perf",
    "performance",
    "plan",
    "plans",
    "profile",
    "profiles",
    "report",
    "reports",
    "result",
    "results",
    "run",
    "runs",
    "sample",
    "samples",
    "temp",
    "tensorrt",
    "test",
    "tests",
    "tmp",
    "torch",
    "trt",
}
CONFIG_MODEL_NAME_KEYS = (
    "_name_or_path",
    "name_or_path",
    "model_name",
    "model_id",
    "model",
    "pretrained_model_name_or_path",
    "base_model_name_or_path",
)
MODEL_NAME_CONFIG_FILENAMES = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "generation_config.json",
)
MODEL_NAME_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
SHORT_MODEL_NAME_TOKENS = {"t5", "gpt", "vit"}
MODEL_NAME_PARENT_HINT_DIRS = {"encode", "encoder", "decode", "decoder"}
MODEL_COMPONENT_STOPWORDS = (MODEL_NAME_STOPWORDS - {"encode", "encoder", "decode", "decoder"}) | {
    "component",
    "components",
}
MODEL_COMPONENT_ALIASES = {
    "decode": "decoder",
    "encode": "encoder",
}
MODEL_COMPONENT_HINT_TOKENS = {
    "controlnet",
    "decoder",
    "denoiser",
    "encoder",
    "transformer",
    "unet",
    "vae",
}


class ModelNameGuess(NamedTuple):
    name: str
    confidence: str
    evidence: str


class DataError(Exception):
    pass


def is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(float(value))


def short_name(value: str, width: int = 96) -> str:
    if len(value) <= width:
        return value
    left = max(12, width // 2 - 3)
    right = max(12, width - left - 3)
    return value[:left] + "..." + value[-right:]


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f}%"


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value < 0.01:
        return f"{value:.4f} ms"
    return f"{value:.3f} ms"


def add_limited(messages: List[str], message: str, limit: int = 40) -> None:
    if len(messages) < limit:
        messages.append(message)
    elif len(messages) == limit:
        messages.append("Further validation messages suppressed.")


def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise DataError(f"Unable to read {path}: {exc}") from exc


def suffix_for(path: str, prefix: str) -> str:
    base = os.path.basename(path)
    match = re.match(rf"^{prefix}s?[_-]?(.*)\.json$", base, flags=re.IGNORECASE)
    if match:
        return match.group(1) or "default"
    return os.path.splitext(base)[0]


def discover_backends(folder: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    layers: Dict[str, str] = {}
    profiles: Dict[str, str] = {}

    try:
        names = sorted(os.listdir(folder))
    except OSError as exc:
        raise DataError(f"Unable to list folder {folder}: {exc}") from exc

    for name in names:
        if not name.lower().endswith(".json"):
            continue
        layer_match = re.match(r"^layers?[_-]?(.*)\.json$", name, flags=re.IGNORECASE)
        profile_match = re.match(r"^profiles?[_-]?(.*)\.json$", name, flags=re.IGNORECASE)
        if layer_match:
            layers[layer_match.group(1) or "default"] = os.path.join(folder, name)
        elif profile_match:
            profiles[profile_match.group(1) or "default"] = os.path.join(folder, name)

    backends = [
        (key, layers.get(key), profiles.get(key))
        for key in sorted(set(layers) | set(profiles))
    ]
    if not backends:
        raise DataError(f"No layers_*.json or profile_*.json files found in {folder}.")
    return backends


def explicit_backends(data_specs: Sequence[Sequence[str]]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    backends: List[Tuple[str, Optional[str], Optional[str]]] = []
    for spec in data_specs:
        if len(spec) not in (1, 2):
            raise DataError("--data expects one layer-info path and optional one profile JSON path")
        layer_spec = spec[0]
        profile_spec = spec[1] if len(spec) == 2 else None
        if not layer_spec:
            raise DataError("--data must start with a layer-info JSON path")
        if profile_spec == "":
            raise DataError("--data profile JSON path must be non-empty")

        layer_path = os.path.abspath(layer_spec)
        profile_path = os.path.abspath(profile_spec) if profile_spec is not None else None
        backends.append((suffix_for(layer_path, "layer"), layer_path, profile_path))
    return backends


def common_source_folder(source_paths: Sequence[str]) -> Optional[str]:
    if not source_paths:
        return None
    folders = [os.path.dirname(path) for path in source_paths]
    try:
        return os.path.commonpath(folders)
    except ValueError:
        return folders[0]


def validate_tensor(
    tensor: Any,
    layer_name: str,
    field_name: str,
    index: int,
    errors: List[str],
) -> None:
    if not isinstance(tensor, dict):
        add_limited(errors, f"Layer `{layer_name}` {field_name}[{index}] is not an object.")
        return
    name = tensor.get("Name")
    if not isinstance(name, str):
        add_limited(errors, f"Layer `{layer_name}` {field_name}[{index}] is missing string `Name`.")
    dims = tensor.get("Dimensions")
    if not isinstance(dims, list) or not all(is_number(dim) for dim in dims):
        add_limited(errors, f"Layer `{layer_name}` tensor `{name}` has invalid `Dimensions`.")
    dtype = tensor.get("Format/Datatype")
    if dtype is not None and not isinstance(dtype, str):
        add_limited(errors, f"Layer `{layer_name}` tensor `{name}` has non-string `Format/Datatype`.")


def validate_layers(data: Any, errors: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        add_limited(errors, "Layer-info JSON root must be a list.")
        return []
    if not data:
        add_limited(errors, "Layer-info JSON must contain at least one layer.")
        return []

    names: collections.Counter[str] = collections.Counter()
    valid_layers: List[Dict[str, Any]] = []

    for idx, layer in enumerate(data):
        if not isinstance(layer, dict):
            add_limited(errors, f"Layer record {idx} is not an object.")
            continue
        name = layer.get("Name")
        if not isinstance(name, str) or not name:
            add_limited(errors, f"Layer record {idx} is missing non-empty string `Name`.")
            name = f"<invalid-layer-{idx}>"
        names[name] += 1
        if not isinstance(layer.get("LayerType"), str):
            add_limited(errors, f"Layer `{name}` is missing string `LayerType`.")
        for field_name in ("Inputs", "Outputs"):
            tensors = layer.get(field_name)
            if not isinstance(tensors, list):
                add_limited(errors, f"Layer `{name}` is missing list `{field_name}`.")
                continue
            for tensor_idx, tensor in enumerate(tensors):
                validate_tensor(tensor, name, field_name, tensor_idx, errors)
        valid_layers.append(layer)

    duplicates = [name for name, count in names.items() if count > 1]
    if duplicates:
        add_limited(errors, "Duplicate layer names: " + ", ".join(short_name(name, 64) for name in duplicates[:8]))

    return valid_layers


def has_detailed_tensor_metadata(tensor: Any) -> bool:
    if not isinstance(tensor, dict):
        return False
    detailed_keys = {
        "Datatype",
        "Dimensions",
        "Format",
        "Format/Datatype",
        "StrideOrder",
        "Strides",
    }
    return any(key in tensor for key in detailed_keys)


def has_detailed_layer_metadata(layer: Any) -> bool:
    if not isinstance(layer, dict):
        return False
    detailed_layer_keys = {"Constants", "Metadata", "StreamId", "TacticName"}
    if any(key in layer for key in detailed_layer_keys):
        return True
    for field_name in ("Inputs", "Outputs", "Constants"):
        tensors = layer.get(field_name)
        if isinstance(tensors, list) and any(has_detailed_tensor_metadata(tensor) for tensor in tensors):
            return True
    return False


def has_detailed_layer_info(data: Any) -> bool:
    """Detect TensorRT inspector output generated with detailed profiling verbosity."""
    return isinstance(data, list) and any(has_detailed_layer_metadata(layer) for layer in data)


def message(level: str, text: str) -> Dict[str, str]:
    return {"level": level, "message": text}


def backend_validation(
    status: str,
    analysis_mode: str,
    errors: Sequence[str],
    warnings: Sequence[str],
    layer_path: Optional[str],
    profile_path: Optional[str],
    report_available: bool,
) -> Dict[str, Any]:
    return {
        "status": status,
        "analysis_mode": analysis_mode,
        "messages": [message("error", item) for item in errors] + [message("warning", item) for item in warnings],
        "source_paths": {
            "layer": layer_path,
            "profile": profile_path,
        },
        "report_available": report_available,
    }


def top_level_validation(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    passed_count = sum(1 for result in results if result.get("validation", {}).get("status") == "passed")
    failed_count = len(results) - passed_count
    if failed_count == 0:
        status = "passed"
    elif passed_count > 0:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "backend_count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "warnings": [],
    }


def validate_profile(data: Any, errors: List[str]) -> Tuple[List[Dict[str, Any]], Optional[int], str]:
    if not isinstance(data, list):
        add_limited(errors, "Profile JSON root must be a list.")
        return [], None, "none"
    if not data:
        add_limited(errors, "Profile JSON must contain at least one record.")
        return [], None, "none"

    count: Optional[int] = None
    count_source = "none"
    entries = data
    if isinstance(data[0], dict) and set(data[0].keys()) == {"count"}:
        raw_count = data[0].get("count")
        if isinstance(raw_count, int) and raw_count > 0:
            count = raw_count
            count_source = "profile header"
        else:
            add_limited(errors, "Profile count header must be a positive integer.")
        entries = data[1:]

    if not entries:
        add_limited(errors, "Profile JSON contains no layer timing records.")
        return [], count, count_source

    names: collections.Counter[str] = collections.Counter()
    valid_entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            add_limited(errors, f"Profile record {idx} is not an object.")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            add_limited(errors, f"Profile record {idx} is missing non-empty string `name`.")
            name = f"<invalid-profile-{idx}>"
        names[name] += 1
        for field_name in ("timeMs", "averageMs", "medianMs", "percentage"):
            if not is_number(entry.get(field_name)):
                add_limited(errors, f"Profile record `{short_name(name, 72)}` has invalid `{field_name}`.")
        valid_entries.append(entry)

    duplicates = [name for name, duplicate_count in names.items() if duplicate_count > 1]
    if duplicates:
        add_limited(errors, "Duplicate profile names: " + ", ".join(short_name(name, 64) for name in duplicates[:8]))

    return valid_entries, count, count_source


def validate_names(layers: Sequence[Dict[str, Any]], profile: Sequence[Dict[str, Any]], errors: List[str]) -> None:
    layer_names = {layer.get("Name") for layer in layers if isinstance(layer.get("Name"), str)}
    profile_names = {entry.get("name") for entry in profile if isinstance(entry.get("name"), str)}
    missing_profile = sorted(layer_names - profile_names)
    extra_profile = sorted(profile_names - layer_names)
    if missing_profile:
        sample = ", ".join(short_name(name, 72) for name in missing_profile[:6])
        add_limited(errors, f"{len(missing_profile)} layer names are missing from profile JSON: {sample}")
    if extra_profile:
        sample = ", ".join(short_name(name, 72) for name in extra_profile[:6])
        add_limited(errors, f"{len(extra_profile)} profile names are missing from layer-info JSON: {sample}")


def tensor_name(tensor: Any) -> Optional[str]:
    if isinstance(tensor, dict) and isinstance(tensor.get("Name"), str) and tensor.get("Name"):
        return tensor["Name"]
    return None


def validate_dag(layers: Sequence[Dict[str, Any]], errors: List[str]) -> Dict[str, Any]:
    producer: Dict[str, int] = {}
    duplicate_outputs: List[str] = []
    for idx, layer in enumerate(layers):
        for tensor in layer.get("Outputs", []) if isinstance(layer.get("Outputs"), list) else []:
            name = tensor_name(tensor)
            if not name:
                continue
            if name in producer:
                duplicate_outputs.append(name)
            else:
                producer[name] = idx

    if duplicate_outputs:
        sample = ", ".join(short_name(name, 72) for name in duplicate_outputs[:6])
        add_limited(errors, f"Duplicate tensor producers make the graph ambiguous: {sample}")

    adjacency = [set() for _ in layers]
    indegree = [0 for _ in layers]
    external_inputs = set()
    consumed_tensors = set()

    for consumer_idx, layer in enumerate(layers):
        for tensor in layer.get("Inputs", []) if isinstance(layer.get("Inputs"), list) else []:
            name = tensor_name(tensor)
            if not name:
                continue
            producer_idx = producer.get(name)
            if producer_idx is None:
                external_inputs.add(name)
                continue
            consumed_tensors.add(name)
            if producer_idx == consumer_idx:
                add_limited(errors, f"Layer `{short_name(layer.get('Name', '<unknown>'), 72)}` consumes its own output `{name}`.")
                continue
            if consumer_idx not in adjacency[producer_idx]:
                adjacency[producer_idx].add(consumer_idx)
                indegree[consumer_idx] += 1

    queue = collections.deque(idx for idx, degree in enumerate(indegree) if degree == 0)
    visited = 0
    while queue:
        idx = queue.popleft()
        visited += 1
        for next_idx in adjacency[idx]:
            indegree[next_idx] -= 1
            if indegree[next_idx] == 0:
                queue.append(next_idx)

    if visited != len(layers):
        unresolved = [
            short_name(str(layers[idx].get("Name", idx)), 72)
            for idx, degree in enumerate(indegree)
            if degree > 0
        ][:8]
        add_limited(errors, "Layer graph is not a DAG; unresolved nodes: " + ", ".join(unresolved))

    output_tensors = set(producer)
    graph_outputs = output_tensors - consumed_tensors
    edges = sum(len(next_nodes) for next_nodes in adjacency)
    return {
        "edge_count": edges,
        "external_inputs": sorted(external_inputs),
        "graph_outputs": sorted(graph_outputs),
    }


def infer_count(profile: Sequence[Dict[str, Any]], header_count: Optional[int]) -> Tuple[Optional[int], str, List[str]]:
    warnings: List[str] = []
    if header_count is not None:
        source = "profile header"
    else:
        source = "timeMs/averageMs ratio"

    ratios: List[float] = []
    for entry in profile:
        time_ms = entry.get("timeMs")
        average_ms = entry.get("averageMs")
        if is_number(time_ms) and is_number(average_ms) and float(average_ms) > 0:
            ratios.append(float(time_ms) / float(average_ms))

    inferred: Optional[int] = None
    if ratios:
        median_ratio = sorted(ratios)[len(ratios) // 2]
        rounded = int(round(median_ratio))
        if rounded > 0:
            max_delta = max(abs(ratio - rounded) for ratio in ratios)
            if max_delta <= max(0.05 * rounded, 0.5):
                inferred = rounded
            else:
                warnings.append("Profile timeMs/averageMs ratios are not consistent enough to infer iteration count.")

    if header_count is not None:
        if inferred is not None and abs(inferred - header_count) > max(1, int(0.05 * header_count)):
            warnings.append(f"Profile count header ({header_count}) disagrees with inferred ratio ({inferred}).")
        return header_count, "profile header", warnings

    if inferred is not None:
        return inferred, source, warnings
    return None, "none", warnings


def validate_timing(profile: Sequence[Dict[str, Any]], count: Optional[int], warnings: List[str]) -> Dict[str, float]:
    total_time = sum(float(entry["timeMs"]) for entry in profile if is_number(entry.get("timeMs")))
    total_average = sum(float(entry["averageMs"]) for entry in profile if is_number(entry.get("averageMs")))
    total_percentage = sum(float(entry["percentage"]) for entry in profile if is_number(entry.get("percentage")))

    if abs(total_percentage - 100.0) > 2.0:
        warnings.append(f"Profile percentages sum to {total_percentage:.2f}%, not approximately 100%.")

    if count and total_average > 0:
        expected_time = total_average * count
        if abs(expected_time - total_time) > max(0.01, 0.02 * max(total_time, expected_time)):
            warnings.append(
                "Sum(timeMs) does not match sum(averageMs) * count "
                f"({total_time:.4f} vs {expected_time:.4f})."
            )

    return {
        "total_time_ms": total_time,
        "average_inference_ms": total_average if total_average else (total_time / count if count else 0.0),
        "total_percentage": total_percentage,
    }


def load_optional_config(folder: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not folder:
        return None, None
    candidates = [
        os.path.join(folder, "model_hub", "config.json"),
        os.path.join(folder, "config.json"),
    ]
    try:
        for name in sorted(os.listdir(folder)):
            candidate = os.path.join(folder, name, "config.json")
            if candidate not in candidates:
                candidates.append(candidate)
    except OSError:
        pass

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            data = load_json(path)
        except DataError:
            continue
        if isinstance(data, dict):
            return data, path
    return None, None


def normalized_name_tokens(value: str) -> List[str]:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return [
        token
        for token in re.split(r"[^a-z0-9]+", ascii_value)
        if token
    ]


def model_name_tokens(value: str) -> List[str]:
    return [token for token in normalized_name_tokens(value) if token not in MODEL_NAME_STOPWORDS]


def has_model_name_signal(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if all(re.fullmatch(r"v?\d+", token) for token in tokens):
        return False

    alpha_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
    if not alpha_tokens:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        return len(token) >= 3 or token in SHORT_MODEL_NAME_TOKENS
    return any(len(token) >= 2 for token in alpha_tokens)


def clean_model_name_segment(value: str) -> Optional[str]:
    tokens = model_name_tokens(value)
    if not has_model_name_signal(tokens):
        return None
    return "-".join(tokens[:16])


def clean_model_name(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    segments = [segment.strip() for segment in re.split(r"[\\/]+", ascii_value) if segment.strip()]
    if not segments:
        segments = [ascii_value]

    for segment in reversed(segments):
        cleaned = clean_model_name_segment(segment)
        if cleaned:
            return cleaned
    return None


def model_name_config_paths(folder: Optional[str]) -> List[str]:
    if not folder:
        return []

    candidates: List[str] = []
    for filename in MODEL_NAME_CONFIG_FILENAMES:
        candidates.append(os.path.join(folder, filename))
        candidates.append(os.path.join(folder, "model_hub", filename))

    try:
        for name in sorted(os.listdir(folder)):
            child = os.path.join(folder, name)
            if not os.path.isdir(child):
                continue
            for filename in MODEL_NAME_CONFIG_FILENAMES:
                candidates.append(os.path.join(child, filename))
    except OSError:
        pass

    unique: List[str] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def infer_model_name_from_configs(folder: Optional[str]) -> List[ModelNameGuess]:
    guesses: List[ModelNameGuess] = []
    for path in model_name_config_paths(folder):
        if not os.path.isfile(path):
            continue
        try:
            data = load_json(path)
        except DataError:
            continue
        if not isinstance(data, dict):
            continue
        for key in CONFIG_MODEL_NAME_KEYS:
            cleaned = clean_model_name(data.get(key))
            if cleaned:
                guesses.append(ModelNameGuess(cleaned, "high", f"config `{os.path.basename(path)}` field `{key}`"))
                break
    return guesses


def infer_model_name_from_folder(folder: Optional[str]) -> List[ModelNameGuess]:
    if not folder:
        return []

    parts = [part for part in os.path.abspath(folder).split(os.sep) if part]
    guesses: List[ModelNameGuess] = []
    candidate_parts = [parts[-1]] if parts else []
    if parts and parts[-1].lower() in MODEL_NAME_PARENT_HINT_DIRS and len(parts) >= 2:
        candidate_parts.append(parts[-2])

    for part in candidate_parts:
        cleaned = clean_model_name_segment(part)
        if cleaned:
            guesses.append(ModelNameGuess(cleaned, "medium", f"directory name `{part}`"))
            break
    return guesses


def infer_model_name_from_filenames(source_paths: Optional[Sequence[str]]) -> List[ModelNameGuess]:
    guesses: List[ModelNameGuess] = []
    for path in source_paths or []:
        base = os.path.splitext(os.path.basename(path))[0]
        stripped = re.sub(r"^(layers?|profiles?)[_-]*", "", base, flags=re.IGNORECASE)
        cleaned = clean_model_name(stripped)
        if cleaned:
            guesses.append(ModelNameGuess(cleaned, "medium", f"input file `{os.path.basename(path)}`"))
    return guesses


def empty_model_name_metadata() -> Dict[str, Any]:
    return {"name": None, "name_confidence": "low", "name_evidence": []}


def infer_model_name_metadata(
    folder: Optional[str],
    explicit_name: Optional[str] = None,
    source_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    guesses: List[ModelNameGuess] = []
    explicit_cleaned = clean_model_name(explicit_name)
    if explicit_cleaned:
        guesses.append(ModelNameGuess(explicit_cleaned, "high", "explicit model name"))

    guesses.extend(infer_model_name_from_configs(folder))
    guesses.extend(infer_model_name_from_folder(folder))
    guesses.extend(infer_model_name_from_filenames(source_paths))

    if not guesses:
        return empty_model_name_metadata()

    best = max(guesses, key=lambda item: MODEL_NAME_CONFIDENCE_RANK[item.confidence])
    evidence = [guess.evidence for guess in guesses if guess.name == best.name]
    return {
        "name": best.name,
        "name_confidence": best.confidence,
        "name_evidence": evidence,
    }


def clean_model_component_segment(value: str, model_name: Optional[str] = None) -> Optional[str]:
    tokens = [
        MODEL_COMPONENT_ALIASES.get(token, token)
        for token in normalized_name_tokens(value)
        if token not in MODEL_COMPONENT_STOPWORDS
    ]
    if not tokens:
        return None

    model_tokens = set(normalized_name_tokens(model_name or ""))
    if model_tokens:
        without_model = [token for token in tokens if token not in model_tokens]
        if any(token in MODEL_COMPONENT_HINT_TOKENS for token in without_model):
            tokens = without_model

    if not any(token in MODEL_COMPONENT_HINT_TOKENS for token in tokens):
        return None
    if all(token.isdigit() for token in tokens):
        return None
    return "-".join(tokens[:8])


def append_unique_component(components: List[str], component: Optional[str]) -> None:
    if component and component not in components:
        components.append(component)


def infer_model_components(
    folder: Optional[str],
    source_paths: Optional[Sequence[str]],
    model_name: Optional[str],
) -> List[str]:
    components: List[str] = []
    if folder:
        folder_name = os.path.basename(os.path.abspath(folder))
        append_unique_component(components, clean_model_component_segment(folder_name, model_name))

    for path in source_paths or []:
        base = os.path.splitext(os.path.basename(path))[0]
        stripped = re.sub(r"^(layers?|profiles?)[_-]*", "", base, flags=re.IGNORECASE)
        append_unique_component(components, clean_model_component_segment(stripped, model_name))

    return components


def collect_shapes(layers: Sequence[Dict[str, Any]]) -> List[Tuple[float, ...]]:
    shapes: List[Tuple[float, ...]] = []
    for layer in layers:
        for field_name in ("Inputs", "Outputs"):
            for tensor in layer.get(field_name, []) if isinstance(layer.get(field_name), list) else []:
                dims = tensor.get("Dimensions") if isinstance(tensor, dict) else None
                if isinstance(dims, list) and all(is_number(dim) for dim in dims):
                    shapes.append(tuple(float(dim) for dim in dims))
    return shapes


def external_input_shapes(layers: Sequence[Dict[str, Any]], external_inputs: Iterable[str]) -> Dict[str, Tuple[float, ...]]:
    external = set(external_inputs)
    result: Dict[str, Tuple[float, ...]] = {}
    for layer in layers:
        for tensor in layer.get("Inputs", []) if isinstance(layer.get("Inputs"), list) else []:
            name = tensor_name(tensor)
            dims = tensor.get("Dimensions") if isinstance(tensor, dict) else None
            if name in external and isinstance(dims, list) and all(is_number(dim) for dim in dims):
                result[name] = tuple(float(dim) for dim in dims)
    return result


def infer_model_info(
    layers: Sequence[Dict[str, Any]],
    graph: Dict[str, Any],
    folder: Optional[str],
    model_name_metadata: Optional[Dict[str, Any]] = None,
    model_components: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    config, config_path = load_optional_config(folder)
    name_metadata = model_name_metadata or empty_model_name_metadata()
    names_blob = "\n".join(
        str(layer.get("Name", "")) + "\n" + str(layer.get("Metadata", ""))
        for layer in layers
    ).lower()

    category = "unknown DL model"
    confidence = "low"
    evidence: List[str] = []

    if config:
        model_type = str(config.get("model_type", "")).lower()
        architectures = " ".join(str(item).lower() for item in config.get("architectures", []))
        if model_type in {"bert", "roberta", "distilbert", "albert", "mpnet"} or "bert" in architectures:
            category = "Transformer encoder / sentence embedding"
            confidence = "high"
            evidence.append(f"Hugging Face config model_type={config.get('model_type')}")
        elif any(token in model_type for token in ("gpt", "llama", "mistral", "gemma", "decoder")):
            category = "Transformer decoder / LLM"
            confidence = "high"
            evidence.append(f"Hugging Face config model_type={config.get('model_type')}")
        elif "vit" in model_type:
            category = "Vision transformer"
            confidence = "high"
            evidence.append(f"Hugging Face config model_type={config.get('model_type')}")

    if confidence == "low":
        if "input_ids" in names_blob and ("encoder.layer" in names_blob or "attention" in names_blob):
            category = "Transformer encoder / sentence embedding"
            confidence = "medium"
            evidence.append("Layer names include input_ids and attention/encoder markers")
        elif "scaled_dot_product_attention" in names_blob or "attention" in names_blob:
            category = "Transformer model"
            confidence = "medium"
            evidence.append("Layer names include attention markers")
        elif "conv" in names_blob:
            category = "CNN / convolutional vision model"
            confidence = "medium"
            evidence.append("Layer names include convolution markers")

    shapes = collect_shapes(layers)
    last_dims = collections.Counter(
        int(shape[-1])
        for shape in shapes
        if shape and float(shape[-1]).is_integer() and abs(int(shape[-1])) > 1
    )
    hidden_size = None
    intermediate_size = None
    num_layers = None
    num_heads = None
    max_position_embeddings = None

    if config:
        hidden_size = config.get("hidden_size")
        intermediate_size = config.get("intermediate_size")
        num_layers = config.get("num_hidden_layers")
        num_heads = config.get("num_attention_heads")
        max_position_embeddings = config.get("max_position_embeddings")

    if hidden_size is None and last_dims:
        hidden_size = last_dims.most_common(1)[0][0]
    if intermediate_size is None and hidden_size and last_dims:
        larger = [dim for dim, count in last_dims.items() if dim > int(hidden_size) and count >= 2]
        if larger:
            intermediate_size = max(larger)

    if num_layers is None:
        encoder_indices = [int(match) for match in re.findall(r"encoder\.layer\.(\d+)", names_blob)]
        if encoder_indices:
            num_layers = max(encoder_indices) + 1

    input_shapes = external_input_shapes(layers, graph.get("external_inputs", []))
    sequence_lengths = []
    for name, shape in input_shapes.items():
        if any(token in name for token in ("input_ids", "attention_mask", "token_type_ids")) and len(shape) >= 2:
            sequence_lengths.append(int(shape[-1]))
    sequence_length = max(set(sequence_lengths), key=sequence_lengths.count) if sequence_lengths else None

    dynamic_dims = any(any(dim < 0 for dim in shape) for shape in shapes)

    return {
        "name": name_metadata.get("name"),
        "name_confidence": name_metadata.get("name_confidence", "low"),
        "name_evidence": name_metadata.get("name_evidence", []),
        "components": list(model_components or []),
        "category": category,
        "confidence": confidence,
        "evidence": evidence,
        "config_path": config_path,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_position_embeddings": max_position_embeddings,
        "sequence_length": sequence_length,
        "dynamic_dims": dynamic_dims,
        "input_shapes": input_shapes,
    }


def engine_hints(layers: Sequence[Dict[str, Any]], folder: Optional[str]) -> Dict[str, Any]:
    engine_files: List[str] = []
    if folder:
        try:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith((".trt", ".engine", ".plan")):
                    engine_files.append(name)
        except OSError:
            pass

    subgraphs = sorted(
        {
            layer.get("_subgraph")
            for layer in layers
            if isinstance(layer.get("_subgraph"), (int, str))
        },
        key=str,
    )
    streams = sorted(
        {
            layer.get("StreamId")
            for layer in layers
            if isinstance(layer.get("StreamId"), (int, str))
        },
        key=str,
    )

    inferred_count = None
    source = "none"
    if engine_files:
        inferred_count = len(engine_files)
        source = "engine files in folder"
    elif subgraphs:
        inferred_count = len(subgraphs)
        source = "_subgraph ids"
    elif streams:
        inferred_count = len(streams)
        source = "StreamId values"

    return {
        "engine_files": engine_files,
        "subgraphs": subgraphs,
        "streams": streams,
        "inferred_count": inferred_count,
        "source": source,
    }


def raw_layer_type(layer: Dict[str, Any]) -> str:
    value = layer.get("LayerType")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def combined_layer_text(layer: Dict[str, Any]) -> str:
    return " ".join(
        str(layer.get(field, ""))
        for field in ("Name", "LayerType", "TacticName", "Metadata")
    ).lower()


def looks_like_mha(layer: Dict[str, Any]) -> bool:
    combined = combined_layer_text(layer)
    if any(token in combined for token in MHA_TOKENS):
        return True
    return bool(re.search(r"(^|[^a-z0-9])mha([^a-z0-9]|$)", combined))


def infer_layer_type(layer: Dict[str, Any]) -> str:
    layer_type = raw_layer_type(layer)
    normalized = re.sub(r"[\s-]+", "_", layer_type.strip().lower())
    if normalized in MISC_LAYER_TYPES:
        return "misc"
    if layer_type.lower() == "kgen" and looks_like_mha(layer):
        return "kgen_mha"
    return layer_type


def tags_for(layer: Dict[str, Any]) -> List[str]:
    combined = combined_layer_text(layer)
    lower = str(layer.get("Name", "")).lower()
    layer_type = raw_layer_type(layer).lower()
    tags: List[str] = []

    if layer_type == "kgen":
        tags.append("kgen")
    if layer_type in {"gemm", "matrix_multiply"} or "matmul" in combined or "matrix_multiply" in combined:
        tags.append("matmul/gemm")
    if "attention" in combined or "fmha" in combined or "flash" in combined or looks_like_mha(layer):
        tags.append("attention")
    if "scaled_dot_product_attention" in combined:
        tags.append("sdpa")
    if any(token in lower for token in DYNAMIC_TOKENS) or layer_type == "shape_call":
        tags.append("shape/cast/dynamic")
    if "mean" in lower and "sqrt" in lower and "div" in lower:
        tags.append("layernorm/reduction")
    if "erf" in lower or "gelu" in lower:
        tags.append("activation")
    return tags or ["other"]


def is_qkv_fused(layer: Dict[str, Any]) -> bool:
    name = str(layer.get("Name", "")).lower()
    metadata = str(layer.get("Metadata", "")).lower()
    combined = name + " " + metadata
    has_named_qkv = all(token in combined for token in ("query", "key", "value"))
    has_three_matmuls = combined.count("matmul") >= 3 or combined.count("matrix_multiply") >= 3
    has_plus_fusion = str(layer.get("Name", "")).count("+") >= 2
    return has_named_qkv or (has_three_matmuls and has_plus_fusion)


def is_obvious_attention_fusion(layer: Dict[str, Any]) -> bool:
    combined = (str(layer.get("Name", "")) + " " + str(layer.get("Metadata", ""))).lower()
    return any(token in combined for token in ("scaled_dot_product_attention", "fmha", "flash_attention", "flashattention"))


def combine_records(layers: Sequence[Dict[str, Any]], profile: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    profile_by_name = {entry["name"]: entry for entry in profile}
    records: List[Dict[str, Any]] = []
    for layer in layers:
        entry = profile_by_name.get(layer.get("Name"))
        if not entry:
            continue
        raw_type = raw_layer_type(layer)
        records.append(
            {
                "name": layer.get("Name"),
                "layer_type": infer_layer_type(layer),
                "raw_layer_type": raw_type,
                "tactic": layer.get("TacticName", ""),
                "time_ms": float(entry.get("timeMs", 0.0)),
                "avg_ms": float(entry.get("averageMs", 0.0)),
                "median_ms": float(entry.get("medianMs", 0.0)),
                "percentage": float(entry.get("percentage", 0.0)),
                "tags": tags_for(layer),
                "layer": layer,
            }
        )
    return records


def aggregate(records: Sequence[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
    totals: Dict[str, Dict[str, Any]] = {}
    for record in records:
        keys = record[key_name] if isinstance(record[key_name], list) else [record[key_name]]
        for key in keys:
            bucket = totals.setdefault(key, {"name": key, "count": 0, "time_ms": 0.0, "percentage": 0.0})
            bucket["count"] += 1
            bucket["time_ms"] += record["time_ms"]
            bucket["percentage"] += record["percentage"]
    return sorted(totals.values(), key=lambda item: item["percentage"], reverse=True)


def add_issue(
    issues: List[Dict[str, Any]],
    score: float,
    title: str,
    evidence: str,
    next_check: str,
    confidence: str = "medium",
) -> None:
    issues.append(
        {
            "score": score,
            "title": title,
            "evidence": evidence,
            "next_check": next_check,
            "confidence": confidence,
        }
    )


def find_issues(
    records: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
    engines: Dict[str, Any],
    type_breakdown: Sequence[Dict[str, Any]],
    tag_breakdown: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    positives: List[str] = []
    by_type = {item["name"]: item for item in type_breakdown}
    by_tag = {item["name"]: item for item in tag_breakdown}
    top_layers = sorted(records, key=lambda item: item["percentage"], reverse=True)

    if top_layers:
        top = top_layers[0]
        if top["percentage"] >= 10.0:
            add_issue(
                issues,
                top["percentage"] + 30.0,
                "Single-layer hotspot",
                f"`{short_name(top['name'], 96)}` consumes {fmt_pct(top['percentage'])} ({fmt_ms(top['avg_ms'])} average).",
                "Inspect this layer's tactic, precision, shape, and whether neighboring ops can be fused.",
                "high",
            )
        elif top["percentage"] >= 5.0:
            add_issue(
                issues,
                top["percentage"] + 12.0,
                "Moderate layer hotspot",
                f"The largest layer is `{short_name(top['name'], 96)}` at {fmt_pct(top['percentage'])}.",
                "Check whether this layer is expected for the model architecture before deeper tuning.",
                "medium",
            )
        else:
            positives.append(f"No single layer exceeds 5%; the largest layer is {fmt_pct(top['percentage'])}.")

    kgen = by_type.get("kgen")
    if kgen and kgen["percentage"] >= 25.0:
        add_issue(
            issues,
            kgen["percentage"] + 8.0,
            "Pointwise/reduction kernels are a large share",
            f"`kgen` layers account for {fmt_pct(kgen['percentage'])} across {kgen['count']} profiled layers.",
            "Review whether residual, layer norm, activation, mask, and reshape-heavy regions are fusing as intended.",
            "medium",
        )

    dynamic = by_tag.get("shape/cast/dynamic")
    if dynamic and dynamic["percentage"] >= 3.0:
        add_issue(
            issues,
            dynamic["percentage"] + 15.0,
            "Dynamic shape/cast overhead is visible",
            f"Shape/cast/dynamic-tagged layers account for {fmt_pct(dynamic['percentage'])} across {dynamic['count']} tagged layers.",
            "Check whether static shapes or stronger shape specialization can reduce shape/cast kernels.",
            "medium",
        )

    small_records = [record for record in records if record["avg_ms"] < 0.005]
    small_pct = sum(record["percentage"] for record in small_records)
    if len(small_records) >= 20 and small_pct >= 20.0:
        add_issue(
            issues,
            small_pct + 5.0,
            "Many tiny kernels add up",
            f"{len(small_records)} layers below 0.005 ms average account for {fmt_pct(small_pct)}.",
            "Prioritize fusion/partitioning checks over tuning isolated micro-kernels.",
            "medium",
        )

    is_transformer = "transformer" in str(model.get("category", "")).lower()
    block_count = model.get("num_layers")
    qkv_fused_count = sum(1 for record in records if is_qkv_fused(record["layer"]))
    if is_transformer and isinstance(block_count, int) and block_count > 0:
        if qkv_fused_count < block_count:
            add_issue(
                issues,
                30.0 + 3.0 * (block_count - qkv_fused_count),
                "Q/K/V projection fusion may be incomplete",
                f"Detected {qkv_fused_count} QKV-fused projection layers for {block_count} transformer blocks.",
                "Confirm whether query/key/value projections are fused per block in the exported graph and TRT engine.",
                "medium",
            )
        else:
            positives.append(f"Detected {qkv_fused_count} QKV-fused projection layers for {block_count} transformer blocks.")

        attention_fused_count = sum(1 for record in records if is_obvious_attention_fusion(record["layer"]))
        if attention_fused_count < block_count:
            add_issue(
                issues,
                18.0 + 2.0 * (block_count - attention_fused_count),
                "Fused attention is not obvious from layer names",
                f"Detected {attention_fused_count} obvious SDPA/FMHA-style layers for {block_count} transformer blocks.",
                "Verify whether attention is decomposed into matmul/softmax kernels or hidden behind generated kernel names.",
                "low",
            )
        else:
            positives.append(f"Detected {attention_fused_count} obvious fused attention layers for {block_count} transformer blocks.")

    engine_count = engines.get("inferred_count")
    if isinstance(engine_count, int):
        if engine_count > 1:
            add_issue(
                issues,
                35.0,
                "Multiple engine/partition hints",
                f"Inferred {engine_count} engines from {engines.get('source')}.",
                "Investigate graph breaks or unsupported ops that force partitioning.",
                "medium",
            )
        elif engine_count == 1:
            positives.append(f"Engine hints point to one engine ({engines.get('source')}).")

    gemm_bad: List[Dict[str, Any]] = []
    for record in records:
        if "matmul/gemm" not in record["tags"]:
            continue
        tactic = str(record.get("tactic", "")).lower()
        if tactic and not any(token in tactic for token in TENSOR_CORE_TOKENS):
            gemm_bad.append(record)
    gemm_bad_pct = sum(record["percentage"] for record in gemm_bad)
    if gemm_bad_pct >= 5.0:
        add_issue(
            issues,
            gemm_bad_pct + 10.0,
            "Some GEMM tactics do not look Tensor Core based",
            f"{len(gemm_bad)} GEMM-like layers with non-obvious Tensor Core tactics account for {fmt_pct(gemm_bad_pct)}.",
            "Inspect precision settings and tactic choices for these GEMMs.",
            "medium",
        )

    issues.sort(key=lambda item: item["score"], reverse=True)
    return issues, positives


def failure_result(
    label: str,
    errors: Sequence[str],
    warnings: Sequence[str],
    layer_path: Optional[str],
    profile_path: Optional[str],
    layer_count: Optional[int] = None,
    profile_count: Optional[int] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "label": label,
        "validation": backend_validation("failed", "unusable", errors, warnings, layer_path, profile_path, False),
    }
    if layer_count is not None:
        result["layer_count"] = layer_count
    if profile_count is not None:
        result["profile_count"] = profile_count
    return result


def analyze_backend(
    label: str,
    layer_path: Optional[str],
    profile_path: Optional[str],
    folder: Optional[str],
    model_name_metadata: Optional[Dict[str, Any]] = None,
    model_components: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not layer_path:
        errors.append(
            "Full TensorRT layer info is missing. Provide layers_*.json generated with detailed profiling verbosity."
        )
        return failure_result(label, errors, warnings, layer_path, profile_path)

    try:
        layer_data = load_json(layer_path)
    except DataError as exc:
        return failure_result(label, [str(exc)], warnings, layer_path, profile_path)

    layers = validate_layers(layer_data, errors)
    if not has_detailed_layer_info(layer_data):
        add_limited(
            errors,
            "Full TensorRT layer info is missing. Generate layer JSON with TensorRT detailed profiling verbosity.",
        )
    graph = validate_dag(layers, errors) if layers else {"edge_count": 0, "external_inputs": [], "graph_outputs": []}

    if not profile_path:
        if errors:
            return failure_result(label, errors, warnings, layer_path, profile_path, len(layers))
        model = infer_model_info(layers, graph, folder, model_name_metadata, model_components)
        engines = engine_hints(layers, folder)
        return {
            "label": label,
            "validation": backend_validation("passed", "layer_only", errors, warnings, layer_path, profile_path, True),
            "layers": layers,
            "layer_count": len(layers),
            "graph": graph,
            "model": model,
            "engines": engines,
        }

    try:
        profile_data = load_json(profile_path)
    except DataError as exc:
        errors.append(str(exc))
        return failure_result(label, errors, warnings, layer_path, profile_path, len(layers))

    profile, header_count, count_source = validate_profile(profile_data, errors)
    if not errors:
        validate_names(layers, profile, errors)

    if errors:
        return failure_result(label, errors, warnings, layer_path, profile_path, len(layers), len(profile))

    count, count_source, count_warnings = infer_count(profile, header_count)
    warnings.extend(count_warnings)
    timing = validate_timing(profile, count, warnings)
    model = infer_model_info(layers, graph, folder, model_name_metadata, model_components)
    engines = engine_hints(layers, folder)
    records = combine_records(layers, profile)
    type_breakdown = aggregate(records, "layer_type")
    tag_breakdown = aggregate(records, "tags")
    issues, positives = find_issues(records, model, engines, type_breakdown, tag_breakdown)

    return {
        "label": label,
        "validation": backend_validation("passed", "layer_profile", errors, warnings, layer_path, profile_path, True),
        "layers": layers,
        "profile": profile,
        "layer_count": len(layers),
        "profile_count": len(profile),
        "graph": graph,
        "count": count,
        "count_source": count_source,
        "timing": timing,
        "model": model,
        "engines": engines,
        "records": records,
        "type_breakdown": type_breakdown,
        "tag_breakdown": tag_breakdown,
        "issues": issues,
        "positives": positives,
    }


def extract_perf_data(
    path: Optional[str],
    data_specs: Optional[Sequence[Sequence[str]]],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve inputs, validate them, and return structured analysis data."""
    folder: Optional[str] = None

    if data_specs:
        if path:
            raise DataError("Use either a folder path or explicit --data inputs, not both.")
        backends = explicit_backends(data_specs)
        source_paths = [
            source_path
            for _, layer_path, profile_path in backends
            for source_path in (layer_path, profile_path)
            if source_path
        ]
        folder = common_source_folder(source_paths)
    else:
        if not path:
            raise DataError("Provide a folder path, or use --data.")
        input_path = os.path.abspath(path)
        if os.path.isdir(input_path):
            folder = input_path
            backends = discover_backends(input_path)
        else:
            raise DataError("A single file input requires --data.")
        source_paths = [
            source_path
            for _, layer_path, profile_path in backends
            for source_path in (layer_path, profile_path)
            if source_path
        ]
    model_name_metadata = infer_model_name_metadata(folder, model_name, source_paths)
    model_components = infer_model_components(folder, source_paths, model_name_metadata["name"])
    results = [
        analyze_backend(label, layer_path, profile_path, folder, model_name_metadata, model_components)
        for label, layer_path, profile_path in backends
    ]
    return {
        "schema_version": "1.0",
        "folder": folder,
        "model_name": model_name_metadata["name"],
        "model_name_confidence": model_name_metadata["name_confidence"],
        "model_name_evidence": model_name_metadata["name_evidence"],
        "model_components": model_components,
        "validation": top_level_validation(results),
        "results": results,
    }
