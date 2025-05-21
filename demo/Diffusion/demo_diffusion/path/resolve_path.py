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

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List

import tensorrt as trt

from demo_diffusion import pipeline
from demo_diffusion.path import dd_path

ARTIFACT_CACHE_DIRECTORY = os.path.join(os.getcwd(), "artifacts_cache")


def resolve_path(
    model_names: List[str],
    args: argparse.Namespace,
    pipeline_type: pipeline.PIPELINE_TYPE,
    pipeline_uid: str,
) -> dd_path.DDPath:
    """Resolve all paths and store them in a newly constructed dd_path.DDPath object.

    Args:
        model_names (List[str]): List of model names.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dd_path.DDPath: Path object containing all the resolved paths.
    """
    path = dd_path.DDPath()
    model_name_to_model_uri = {
        model_name: _resolve_model_uri(model_name, args, pipeline_type, pipeline_uid) for model_name in model_names
    }

    _resolve_default_path(model_name_to_model_uri, args, path)
    _resolve_custom_path(args, path)

    path.create_directory()

    return path


def _resolve_model_uri(
    model_name: str, args: argparse.Namespace, pipeline_type: pipeline.PIPELINE_TYPE, pipeline_uid: str
) -> str:
    """Resolve and return the model URI.

    The model URI is a partial path that uniquely identifies the model. It is used to construct various model paths like
    artifact cache path, checkpoint path, etc.
    """
    # Lora unique ID represents the lora configuration.
    if args.lora_path and args.lora_weight:
        lora_config_uid = "-".join(
            sorted(
                [
                    f"{hashlib.sha256(lora_path.encode()).hexdigest()}-{lora_weight}-{args.lora_scale}"
                    for lora_path, lora_weight in zip(args.lora_path, args.lora_weight)
                    if args.lora_path
                ]
            )
        )
    else:
        lora_config_uid = ""

    # Quantization config unique ID represents the quantization configuration.
    def _is_quantized() -> bool:
        """Return True if model is quantized, False if otherwise.

        When quantization flags are set in `args`, only a subset of the models are actually quantized.
        """
        is_unet = model_name == "unet"
        is_unetxl_base = pipeline_type.is_sd_xl_base() and model_name == "unetxl"
        is_flux_transformer = args.version.startswith("flux.1") and model_name == "transformer"

        if args.int8:
            return is_unet or is_unetxl_base
        elif args.fp8:
            return is_unet or is_unetxl_base or is_flux_transformer
        elif args.fp4:
            return is_flux_transformer
        else:
            return False

    if _is_quantized():
        if args.int8 or args.fp8:
            quantization_config_uid = (
                f"{'int8' if args.int8 else 'fp8'}.l{args.quantization_level}.bs2"
                f".c{args.calibration_size}.p{args.quantization_percentile}.a{args.quantization_alpha}"
            )
        else:
            quantization_config_uid = "fp4"
    else:
        quantization_config_uid = ""

    # Model unique ID represents the model name and its configuration. It is unique under the same pipeline.
    model_uid = "_".join([s for s in [model_name, lora_config_uid, quantization_config_uid] if s])

    # Model URI is the concatenation of pipeline unique ID and model unique ID.
    model_uri = os.path.join(pipeline_uid, model_uid)

    return model_uri


def _resolve_default_path(
    model_name_to_model_uri: Dict[str, str], args: argparse.Namespace, path: dd_path.DDPath
) -> None:
    """Resolve the default paths.

    Args:
        model_name_to_model_uri (Dict[str, str]): Dictionary of model name to model URI.
        args (argparse.Namespace): Parsed arguments.
        path (dd_path.DDPath): Path object. This object is modified in-place to store all resolved default paths.
    """
    for model_name, model_uri in model_name_to_model_uri.items():
        path.model_name_to_optimized_onnx_path[model_name] = os.path.join(
            args.onnx_dir, model_uri, "model_optimized.onnx"
        )
        path.model_name_to_engine_path[model_name] = os.path.join(
            args.engine_dir, model_uri, f"engine_trt{trt.__version__}.plan"
        )

        # Resolve artifact paths.
        artifact_dir = os.path.join(ARTIFACT_CACHE_DIRECTORY, model_uri)

        path.model_name_to_unoptimized_onnx_path[model_name] = os.path.join(artifact_dir, "model_unoptimized.onnx")
        path.model_name_to_weights_map_path[model_name] = os.path.join(artifact_dir, "weights_map.json")
        path.model_name_to_refit_weights_path[model_name] = os.path.join(artifact_dir, "refit_weights.json")
        path.model_name_to_quantized_model_state_dict_path[model_name] = os.path.join(
            artifact_dir, "quantized_model_state_dict.json"
        )


def _resolve_custom_path(args: argparse.Namespace, path: dd_path.DDPath) -> None:
    """Resolve the custom paths.

    If a different path already exists in `path`, it will be overridden.

    Args:
        args (argparse.Namespace): Parsed arguments.
        path (dd_path.DDPath): Path object. This object is modified in-place to store or override all resolved paths.
    """
    # Resolve and override custom ONNX paths.
    if args.custom_onnx_paths:
        for model_name, optimized_onnx_path in args.custom_onnx_paths.items():
            path.model_name_to_optimized_onnx_path[model_name] = optimized_onnx_path

    # Resolve and override custom engine paths.
    if args.custom_engine_paths:
        for model_name, engine_path in args.custom_engine_paths.items():
            path.model_name_to_engine_path[model_name] = engine_path
