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
"""
Functions for loading models.
"""
from __future__ import annotations

import gc
import glob
import os
import sys
from typing import List, Optional

import torch

import onnx


def onnx_graph_needs_external_data(onnx_graph: onnx.ModelProto) -> bool:
    """Return true if ONNX graph needs to store external data."""
    if sys.platform == "win32":
        # ByteSize is broken (wraps around) on Windows, so always assume external data is needed.
        return True
    else:
        TWO_GIGABYTES = 2147483648
        return onnx_graph.ByteSize() > TWO_GIGABYTES


def get_path(version: str, pipeline: "pipeline.DiffusionPipeline", controlnets: Optional[List[str]] = None) -> str:
    """Return the relative path to the model files directory."""
    if controlnets is not None:
        if version == "xl-1.0":
            return ["diffusers/controlnet-canny-sdxl-1.0"]
        if version == "3.5-large":
            return f"stabilityai/stable-diffusion-3.5-large-controlnet-{controlnets}"
        return ["lllyasviel/sd-controlnet-" + modality for modality in controlnets]

    if version in ("1.4", "1.5") and pipeline.is_inpaint():
        return "benjamin-paine/stable-diffusion-v1-5-inpainting"
    elif version == "1.4":
        return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        return "KiwiXR/stable-diffusion-v1-5"
    elif version == "dreamshaper-7":
        return "Lykon/dreamshaper-7"
    elif version in ("2.0-base", "2.0") and pipeline.is_inpaint():
        return "stabilityai/stable-diffusion-2-inpainting"
    elif version == "2.0-base":
        return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        return "stabilityai/stable-diffusion-2"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "xl-1.0" and pipeline.is_sd_xl_base():
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif version == "xl-1.0" and pipeline.is_sd_xl_refiner():
        return "stabilityai/stable-diffusion-xl-refiner-1.0"
    # TODO SDXL turbo with refiner
    elif version == "xl-turbo" and pipeline.is_sd_xl_base():
        return "stabilityai/sdxl-turbo"
    elif version == "sd3":
        return "stabilityai/stable-diffusion-3-medium"
    elif version == "3.5-medium":
        return "stabilityai/stable-diffusion-3.5-medium"
    elif version == "3.5-large":
        return "stabilityai/stable-diffusion-3.5-large"
    elif version == "svd-xt-1.1" and pipeline.is_img2vid():
        return "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    elif version == "cascade":
        if pipeline.is_cascade_decoder():
            return "stabilityai/stable-cascade"
        else:
            return "stabilityai/stable-cascade-prior"
    elif version == "flux.1-dev":
        return "black-forest-labs/FLUX.1-dev"
    elif version == "flux.1-schnell":
        return "black-forest-labs/FLUX.1-schnell"
    elif version == "flux.1-dev-canny":
        return "black-forest-labs/FLUX.1-Canny-dev"
    elif version == "flux.1-dev-depth":
        return "black-forest-labs/FLUX.1-Depth-dev"
    elif version == "flux.1-kontext-dev":
        return "black-forest-labs/FLUX.1-Kontext-dev"
    elif version == "cosmos-predict2-2b-text2image":
        return "nvidia/Cosmos-Predict2-2B-Text2Image"
    elif version == "cosmos-predict2-14b-text2image":
        return "nvidia/Cosmos-Predict2-14B-Text2Image"
    elif version == "cosmos-predict2-2b-video2world":
        return "nvidia/Cosmos-Predict2-2B-Video2World"
    elif version == "cosmos-predict2-14b-video2world":
        return "nvidia/Cosmos-Predict2-14B-Video2World"
    else:
        raise ValueError(f"Unsupported version {version} + pipeline {pipeline.name}")


# FIXME serialization not supported for torch.compile
def get_checkpoint_dir(framework_model_dir: str, version: str, pipeline: str, subfolder: str) -> str:
    """Return the path to the torch model checkpoint directory."""
    return os.path.join(framework_model_dir, version, pipeline, subfolder)


def is_model_cached(model_dir, model_opts, hf_safetensor, model_name="diffusion_pytorch_model") -> bool:
    """Return True if model was cached."""
    variant = "." + model_opts.get("variant") if "variant" in model_opts else ""
    suffix = ".safetensors" if hf_safetensor else ".bin"
    # WAR with * for larger models that are split into multiple smaller ckpt files
    model_file = model_name + variant + "*" + suffix
    return bool(glob.glob(os.path.join(model_dir, model_file)))


def unload_torch_model(model):
    if model:
        del model
        torch.cuda.empty_cache()
        gc.collect()
