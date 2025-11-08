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
from typing import Any, Dict, Tuple

import torch

# Define valid optimization levels for TensorRT engine build
VALID_OPTIMIZATION_LEVELS = list(range(6))


def parse_key_value_pairs(string: str) -> Dict[str, str]:
    """Parse a string of comma-separated key-value pairs into a dictionary.

    Args:
        string (str): A string of comma-separated key-value pairs.

    Returns:
        Dict[str, str]: Parsed dictionary of key-value pairs.

    Example:
        >>> parse_key_value_pairs("key1:value1,key2:value2")
        {"key1": "value1", "key2": "value2"}
    """
    parsed = {}

    for key_value_pair in string.split(","):
        if not key_value_pair:
            continue

        key_value_pair = key_value_pair.split(":")
        if len(key_value_pair) != 2:
            raise argparse.ArgumentTypeError(f"Invalid key-value pair: {key_value_pair}. Must have length 2.")
        key, value = key_value_pair
        parsed[key] = value

    return parsed


def add_arguments(parser):
    # Stable Diffusion configuration
    parser.add_argument(
        "--version",
        type=str,
        default="1.5",
        choices=(
            "1.4",
            "1.5",
            "dreamshaper-7",
            "2.0-base",
            "2.0",
            "2.1-base",
            "2.1",
            "xl-1.0",
            "xl-turbo",
            "svd-xt-1.1",
            "sd3",
            "3.5-medium",
            "3.5-large",
            "cascade",
            "flux.1-dev",
            "flux.1-schnell",
            "flux.1-dev-canny",
            "flux.1-dev-depth",
            "flux.1-kontext-dev",
            "cosmos-predict2-2b-text2image",
            "cosmos-predict2-14b-text2image",
            "cosmos-predict2-2b-video2world",
            "cosmos-predict2-14b-video2world",
        ),
        help="Version of Stable Diffusion",
    )
    parser.add_argument("prompt", nargs="*", help="Text prompt(s) to guide image generation")
    parser.add_argument(
        "--negative-prompt", nargs="*", default=[""], help="The negative prompt(s) to guide the image generation."
    )
    parser.add_argument("--batch-size", type=int, default=1, choices=[1, 2, 4], help="Batch size (repeat prompt)")
    parser.add_argument(
        "--batch-count", type=int, default=1, help="Number of images to generate in sequence, one at a time."
    )
    parser.add_argument("--height", type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument("--denoising-steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=("DDIM", "DDPM", "EulerA", "Euler", "LCM", "LMSD", "PNDM", "UniPC", "DDPMWuerstchen", "FlowMatchEuler"),
        help="Scheduler for diffusion process",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Value of classifier-free guidance scale (must be greater than 1)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Controls how much to influence the outputs with the LoRA parameters. (must between 0 and 1)",
    )
    parser.add_argument(
        "--lora-weight",
        type=float,
        nargs="+",
        default=None,
        help="The LoRA adapter(s) weights to use with the UNet. (must between 0 and 1)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        nargs="+",
        default=None,
        help="Path to LoRA adaptor. Ex: 'latent-consistency/lcm-lora-sdv1-5'",
    )
    parser.add_argument("--bf16", action="store_true", help="Run pipeline in BFloat16 precision")

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=19,
        choices=range(7, 20),
        help="Select ONNX opset version to target for exported models",
    )
    parser.add_argument("--onnx-dir", default="onnx", help="Output directory for ONNX export")
    parser.add_argument(
        "--custom-onnx-paths",
        type=parse_key_value_pairs,
        help=(
            "[FLUX, Stable Diffusion 3.5-large, Cosmos only] Custom override paths to pre-exported ONNX model files. These ONNX models are directly used to "
            "build TRT engines without further optimization on the ONNX graphs. Paths should be a comma-separated list "
            "of <model_name>:<path> pairs. For example: "
            "--custom-onnx-paths=transformer:/path/to/transformer.onnx,vae:/path/to/vae.onnx. Call "
            "<PipelineClass>.get_model_names(...) for the list of supported model names."
        ),
    )
    parser.add_argument(
        "--onnx-export-only",
        action="store_true",
        help="If set, only performs the export of models to ONNX, skipping engine build and inference.",
    )
    parser.add_argument(
        "--download-onnx-models",
        action="store_true",
        help=("[FLUX and Stable Diffusion 3.5-large only] Download pre-exported ONNX models"),
    )

    # Framework model ckpt
    parser.add_argument("--framework-model-dir", default="pytorch_model", help="Directory for HF saved models")

    # TensorRT engine build
    parser.add_argument("--engine-dir", default="engine", help="Output directory for TensorRT engines")
    parser.add_argument(
        "--custom-engine-paths",
        type=parse_key_value_pairs,
        help=(
            "[FLUX only] Custom override paths to pre-built engine files. Paths should be a comma-separated list of "
            "<model_name>:<path> pairs. For example: "
            "--custom-onnx-paths=transformer:/path/to/transformer.plan,vae:/path/to/vae.plan. Call "
            "<PipelineClass>.get_model_names(...) for the list of supported model names."
        ),
    )

    parser.add_argument(
        "--optimization-level",
        type=int,
        default=None,
        help=f"Set the builder optimization level to build the engine with. A higher level allows TensorRT to spend more building time for more optimization options. Must be one of {VALID_OPTIMIZATION_LEVELS}.",
    )
    parser.add_argument(
        "--build-static-batch", action="store_true", help="Build TensorRT engines with fixed batch size."
    )
    parser.add_argument(
        "--build-dynamic-shape", action="store_true", help="Build TensorRT engines with dynamic image shapes."
    )
    parser.add_argument(
        "--build-enable-refit", action="store_true", help="Enable Refit option in TensorRT engines during build."
    )
    parser.add_argument(
        "--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources."
    )
    parser.add_argument(
        "--timing-cache", default=None, type=str, help="Path to the precached timing measurements to accelerate build."
    )
    parser.add_argument("--ws", action="store_true", help="Build TensorRT engines with weight streaming enabled.")

    # Quantization configuration.
    parser.add_argument("--int8", action="store_true", help="Apply int8 quantization.")
    parser.add_argument("--fp8", action="store_true", help="Apply fp8 quantization.")
    parser.add_argument("--fp4", action="store_true", help="Apply fp4 quantization.")
    parser.add_argument(
        "--quantization-level",
        type=float,
        default=0.0,
        choices=[0.0, 1.0, 2.0, 2.5, 3.0, 4.0],
        help="int8/fp8 quantization level, 1: CNN, 2: CNN + FFN, 2.5: CNN + FFN + QKV, 3: CNN + Almost all Linear (Including FFN, QKV, Proj and others), 4: CNN + Almost all Linear + fMHA, 0: Default to 2.5 for int8 and 4.0 for fp8.",
    )
    parser.add_argument(
        "--quantization-percentile",
        type=float,
        default=1.0,
        help="Control quantization scaling factors (amax) collecting range, where the minimum amax in range(n_steps * percentile) will be collected. Recommendation: 1.0.",
    )
    parser.add_argument(
        "--quantization-alpha",
        type=float,
        default=0.8,
        help="The alpha parameter for SmoothQuant quantization used for linear layers. Recommendation: 0.8 for SDXL.",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=32,
        help="The number of steps to use for calibrating the model for quantization. Recommendation: 32, 64, 128 for SDXL",
    )

    # Inference
    parser.add_argument(
        "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance"
    )
    parser.add_argument("--use-cuda-graph", action="store_true", help="Enable cuda graph")
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling")
    parser.add_argument(
        "--torch-inference",
        default="",
        help="Run inference with PyTorch (using specified compilation mode) instead of TensorRT.",
    )
    parser.add_argument(
        "--torch-fallback",
        default=None,
        type=str,
        help="[FLUX only] Comma separated list of models to be inferenced using torch instead of TRT. For example --torch-fallback t5,transformer. If --torch-inference set, this parameter will be ignored.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="[FLUX only] Optimize for low VRAM usage, possibly at the expense of inference performance. Disabled by default.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results")
    parser.add_argument("--output-dir", default="output", help="Output directory for logs and image artifacts")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    return parser


def process_pipeline_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple]:
    """Validate parsed arguments and process argument values.

    Some argument values are resolved or overwritten during processing.

    Args:
        args (argparse.Namespace): Parsed argument. This is modified in-place.

    Returns:
        Dict[str, Any]: Keyword arguments for initializing a pipeline. This is only used in legacy pipelines that do not
            have factory methods `FromArgs` that construct the pipeline directly from the parsed argument.
        Dict[str, Any]: Keyword arguments for calling the `.load_engine` method of the pipeline.
        Tuple: Arguments for calling the `.run` method of the pipeline.
    """

    # GPU device info
    device_info = torch.cuda.get_device_properties(0)
    sm_version = device_info.major * 10 + device_info.minor

    is_flux = args.version.startswith("flux")
    is_sd35 = args.version.startswith("3.5")
    is_cosmos = args.version.startswith("cosmos")

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 8 but specified as: {args.image_height} and {args.width}."
        )

    # Handle batch size
    max_batch_size = 4
    if args.batch_size > max_batch_size:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed {max_batch_size}.")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(
            "Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`"
        )

    # TensorRT builder optimization level
    if args.optimization_level is None:
        # optimization level set to 3 for all Flux pipelines to reduce GPU memory usage
        if args.int8 or args.fp8 and not is_flux:
            args.optimization_level = 4
        else:
            args.optimization_level = 3

    if args.optimization_level not in VALID_OPTIMIZATION_LEVELS:
        raise ValueError(
            f"Optimization level {args.optimization_level} not valid.  Valid values are: {VALID_OPTIMIZATION_LEVELS}"
        )

    # Quantized pipeline
    # int8 support
    if args.int8 and not any(args.version.startswith(prefix) for prefix in ("xl", "1.4", "1.5", "2.1")):
        raise ValueError("int8 quantization is only supported for SDXL, SD1.4, SD1.5 and SD2.1 pipelines.")

    # fp8 support validation
    if args.fp8:
        # Check version compatibility
        supported_versions = ("xl", "1.4", "1.5", "2.1", "3.5-large")
        if not (any(args.version.startswith(prefix) for prefix in supported_versions) or is_flux):
            raise ValueError(
                "fp8 quantization is only supported for SDXL, SD1.4, SD1.5, SD2.1, SD3.5-large and FLUX pipelines."
            )

        # Check controlnet compatibility
        if getattr(args, "controlnet_type", None) is not None:
            if args.version not in ("xl-1.0", "3.5-large"):
                raise ValueError("fp8 controlnet quantization is only supported for SDXL and SD3.5-large.")
            if args.version == "3.5-large" and args.controlnet_type == "blur":
                raise ValueError("Blur controlnet type is not supported for SD3.5.")
        # Check for conflicting quantization
        if args.int8:
            raise ValueError("Cannot apply both int8 and fp8 quantization, please choose only one.")

        # Check GPU compute capability
        if sm_version < 89:
            raise ValueError(
                f"Cannot apply FP8 quantization for GPU with compute capability {sm_version / 10.0}. A minimum compute capability of 8.9 is required."
            )

        # Check SD3.5-large specific requirement
        if args.version == "3.5-large" and not args.download_onnx_models:
            raise ValueError(
                "Native FP8 quantization is not supported for SD3.5-large. Please pass --download-onnx-models."
            )

    # TensorRT ModelOpt quantization level
    if args.quantization_level == 0.0:
        def override_quant_level(level: float, dtype_str: str):
            args.quantization_level = level
            print(f"[W] The default quantization level has been set to {level} for {dtype_str}.")

        if args.fp8:
            # L4 fp8 fMHA on Hopper not yet enabled.
            if sm_version == 90 and is_flux:
                override_quant_level(3.0, "FP8")
            else:
                override_quant_level(3.0 if args.version in ("1.4", "1.5") else 4.0, "FP8")

        elif args.int8:
            override_quant_level(3.0, "INT8")

    if args.version.startswith("flux") and args.quantization_level == 3.0 and args.download_onnx_models:
        raise ValueError(
            "Transformer ONNX model for Quantization level 3 is not available for download. Please export the quantized Transformer model natively with the removal of --download-onnx-models."
        )
    if args.fp4:
        # FP4 precision is only supported for the Flux pipeline
        assert is_flux, "FP4 precision is only supported for the Flux pipeline"

    # Handle LoRA
    # FLUX canny and depth official LoRAs are not supported because they modify the transformer architecture, conflicting with refit
    if args.lora_path and not any(args.version.startswith(prefix) for prefix in ("1.5", "2.1", "xl", "flux.1-dev", "flux.1-schnell")):
        raise ValueError("LoRA adapter support is only supported for SD1.5, SD2.1, SDXL, FLUX.1-dev and FLUX.1-schnell pipelines")

    if args.lora_weight:
        for weight in (weight for weight in args.lora_weight if not 0 <= weight <= 1):
            raise ValueError(f"LoRA adapter weights must be between 0 and 1, provided {weight}")

    if not 0 <= args.lora_scale <= 1:
        raise ValueError(f"LoRA scale value must be between 0 and 1, provided {args.lora_scale}")

    # Force lora merge when fp8 or int8 is used with LoRA
    if args.build_enable_refit and args.lora_path and (args.int8 or args.fp8):
        raise ValueError(
            "Engine refit should not be enabled for quantized models with LoRA. ModelOpt recommends fusing the LoRA to the model before quantization. \
            See https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/quantization#lora"
        )

    # Torch-fallback and Torch-inference
    if args.torch_fallback and not args.torch_inference:
        assert (
            is_flux or is_sd35 or is_cosmos
        ), "PyTorch Fallback is only supported for Flux, Stable Diffusion 3.5 and Cosmos pipelines."
        args.torch_fallback = args.torch_fallback.split(",")

    if args.torch_fallback and args.torch_inference:
        print(
            "[W] All models will run in PyTorch when --torch-inference is set. Parameter --torch-fallback will be ignored."
        )
        args.torch_fallback = None

    # low-vram
    if args.low_vram:
        assert (
            is_flux or is_sd35 or is_cosmos
        ), "low-vram mode is only supported for Flux, Stable Diffusion 3.5 and Cosmos pipelines."

    # Pack arguments
    kwargs_init_pipeline = {
        "version": args.version,
        "max_batch_size": max_batch_size,
        "denoising_steps": args.denoising_steps,
        "scheduler": args.scheduler,
        "guidance_scale": args.guidance_scale,
        "output_dir": args.output_dir,
        "hf_token": args.hf_token,
        "verbose": args.verbose,
        "nvtx_profile": args.nvtx_profile,
        "use_cuda_graph": args.use_cuda_graph,
        "lora_scale": args.lora_scale,
        "lora_weight": args.lora_weight,
        "lora_path": args.lora_path,
        "framework_model_dir": args.framework_model_dir,
        "torch_inference": args.torch_inference,
    }

    kwargs_load_engine = {
        "onnx_opset": args.onnx_opset,
        "opt_batch_size": args.batch_size,
        "opt_image_height": args.height,
        "opt_image_width": args.width,
        "optimization_level": args.optimization_level,
        "static_batch": args.build_static_batch,
        "static_shape": not args.build_dynamic_shape,
        "enable_all_tactics": args.build_all_tactics,
        "enable_refit": args.build_enable_refit,
        "timing_cache": args.timing_cache,
        "int8": args.int8,
        "fp8": args.fp8,
        "fp4": args.fp4,
        "quantization_level": args.quantization_level,
        "quantization_percentile": args.quantization_percentile,
        "quantization_alpha": args.quantization_alpha,
        "calibration_size": args.calibration_size,
        "onnx_export_only": args.onnx_export_only,
        "download_onnx_models": args.download_onnx_models,
    }

    args_run_demo = (
        args.prompt,
        args.negative_prompt,
        args.height,
        args.width,
        args.batch_size,
        args.batch_count,
        args.num_warmup_runs,
        args.use_cuda_graph,
    )

    return kwargs_init_pipeline, kwargs_load_engine, args_run_demo
