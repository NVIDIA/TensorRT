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

from demo_diffusion.model.base_model import BaseModel
from demo_diffusion.model.clip import (
    CLIPImageProcessorModel,
    CLIPModel,
    CLIPVisionWithProjModel,
    CLIPWithProjModel,
    SD3_CLIPGModel,
    SD3_CLIPLModel,
    SD3_T5XXLModel,
    get_clip_embedding_dim,
)
from demo_diffusion.model.controlnet import SD3ControlNet
from demo_diffusion.model.diffusion_transformer import (
    CosmosTransformerModel,
    FluxTransformerModel,
    SD3_MMDiTModel,
    SD3TransformerModel,
)
from demo_diffusion.model.gan import VQGANModel
from demo_diffusion.model.load import unload_torch_model
from demo_diffusion.model.lora import FLUXLoraLoader, SDLoraLoader, merge_loras
from demo_diffusion.model.scheduler import make_scheduler
from demo_diffusion.model.t5 import T5Model
from demo_diffusion.model.tokenizer import make_tokenizer
from demo_diffusion.model.unet import (
    UNet2DConditionControlNetModel,
    UNetCascadeModel,
    UNetModel,
    UNetTemporalModel,
    UNetXLModel,
    UNetXLModelControlNet,
)
from demo_diffusion.model.vae import (
    AutoencoderKLWanEncoderModel,
    AutoencoderKLWanModel,
    SD3_VAEDecoderModel,
    SD3_VAEEncoderModel,
    TorchVAEEncoder,
    VAEDecTemporalModel,
    VAEEncoderModel,
    VAEModel,
)

__all__ = [
    # base_model
    "BaseModel",
    # clip
    "get_clip_embedding_dim",
    "CLIPModel",
    "CLIPWithProjModel",
    "SD3_CLIPGModel",
    "SD3_CLIPLModel",
    "SD3_T5XXLModel",
    "CLIPVisionWithProjModel",
    "CLIPImageProcessorModel",
    "CosmosTransformerModel",
    # diffusion_transformer
    "SD3_MMDiTModel",
    "FluxTransformerModel",
    "SD3TransformerModel",
    "SD3ControlNet",
    # gan
    "VQGANModel",
    # lora
    "SDLoraLoader",
    "FLUXLoraLoader",
    "merge_loras",
    # scheduler
    "make_scheduler",
    # t5
    "T5Model",
    # tokenizer
    "make_tokenizer",
    # unet
    "UNetModel",
    "UNetXLModel",
    "UNetXLModelControlNet",
    "UNet2DConditionControlNetModel",
    "UNetTemporalModel",
    "UNetCascadeModel",
    # vae
    "VAEModel",
    "SD3_VAEDecoderModel",
    "VAEDecTemporalModel",
    "TorchVAEEncoder",
    "VAEEncoderModel",
    "SD3_VAEEncoderModel",
    "AutoencoderKLWanModel",
    "AutoencoderKLWanEncoderModel",
    # load
    "unload_torch_model",
]
