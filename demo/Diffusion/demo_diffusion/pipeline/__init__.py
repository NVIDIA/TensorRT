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
import importlib
from typing import TYPE_CHECKING

# Expose public API while avoiding importing optional dependencies at module import time.
# Each attribute is imported on first access via __getattr__.

__all__ = [
    "DiffusionPipeline",
    "FluxPipeline",
    "FluxKontextPipeline",
    "StableCascadePipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusion35Pipeline",
    "StableDiffusionPipeline",
    "CosmosPipeline",
    "StableVideoDiffusionPipeline",
    "PIPELINE_TYPE",
]

_LAZY_ATTRS = {
    # Core/base
    "DiffusionPipeline": ("demo_diffusion.pipeline.diffusion_pipeline", "DiffusionPipeline"),
    "PIPELINE_TYPE": ("demo_diffusion.pipeline.type", "PIPELINE_TYPE"),
    # Stable Diffusion family
    "StableDiffusionPipeline": ("demo_diffusion.pipeline.stable_diffusion_pipeline", "StableDiffusionPipeline"),
    "StableDiffusion3Pipeline": ("demo_diffusion.pipeline.stable_diffusion_3_pipeline", "StableDiffusion3Pipeline"),
    "StableDiffusion35Pipeline": ("demo_diffusion.pipeline.stable_diffusion_35_pipeline", "StableDiffusion35Pipeline"),
    # Stable Cascade
    "StableCascadePipeline": ("demo_diffusion.pipeline.stable_cascade_pipeline", "StableCascadePipeline"),
    # Stable Video Diffusion
    "StableVideoDiffusionPipeline": ("demo_diffusion.pipeline.stable_video_diffusion_pipeline", "StableVideoDiffusionPipeline"),
    # Flux family (optional dependency: `flux`)
    "FluxPipeline": ("demo_diffusion.pipeline.flux_pipeline", "FluxPipeline"),
    "FluxKontextPipeline": ("demo_diffusion.pipeline.flux_pipeline", "FluxKontextPipeline"),
    # Cosmos (optional dependency: `flux`)
    "CosmosPipeline": ("demo_diffusion.pipeline.cosmos_pipeline", "CosmosPipeline"),
}

def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'demo_diffusion.pipeline' has no attribute {name!r}")

    module_path, attr_name = _LAZY_ATTRS[name]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        missing_pkg = e.name or "<unknown>"
        raise ModuleNotFoundError(
            f"Optional dependency '{missing_pkg}' is required for '{name}'. "
            "Install the appropriate extras/requirements for the selected pipeline "
            "(e.g., use the non-legacy requirements for Flux/Cosmos), or install the missing package."
        ) from e
    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(
            f"'{module_path}' does not export attribute '{attr_name}' (while resolving '{name}')."
        ) from e


if TYPE_CHECKING:
    from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline as DiffusionPipeline
    from demo_diffusion.pipeline.type import PIPELINE_TYPE as PIPELINE_TYPE
    from demo_diffusion.pipeline.stable_diffusion_pipeline import (
        StableDiffusionPipeline as StableDiffusionPipeline,
    )
    from demo_diffusion.pipeline.stable_diffusion_3_pipeline import (
        StableDiffusion3Pipeline as StableDiffusion3Pipeline,
    )
    from demo_diffusion.pipeline.stable_diffusion_35_pipeline import (
        StableDiffusion35Pipeline as StableDiffusion35Pipeline,
    )
    from demo_diffusion.pipeline.stable_cascade_pipeline import (
        StableCascadePipeline as StableCascadePipeline,
    )
    from demo_diffusion.pipeline.stable_video_diffusion_pipeline import (
        StableVideoDiffusionPipeline as StableVideoDiffusionPipeline,
    )
    # Optional pipelines
    from demo_diffusion.pipeline.flux_pipeline import (
        FluxPipeline as FluxPipeline,
        FluxKontextPipeline as FluxKontextPipeline,
    )
    from demo_diffusion.pipeline.cosmos_pipeline import CosmosPipeline as CosmosPipeline
