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

from abc import ABC
from diffusers.loaders import StableDiffusionLoraLoaderMixin, FluxLoraLoaderMixin


class LoraLoader(ABC):
    def __init__(self, paths, weights, scale):
        self.paths = paths
        self.weights = weights
        self.scale = scale


class SDLoraLoader(LoraLoader, StableDiffusionLoraLoaderMixin):
    def __init__(self, paths, weights, scale):
        super().__init__(paths, weights, scale)


class FLUXLoraLoader(LoraLoader, FluxLoraLoaderMixin):
    def __init__(self, paths, weights, scale):
        super().__init__(paths, weights, scale)


def merge_loras(model, lora_loader):
    paths, weights, scale = lora_loader.paths, lora_loader.weights, lora_loader.scale
    for i, path in enumerate(paths):
        print(f"[I] Loading LoRA: {path}, weight {weights[i]}")
        if isinstance(lora_loader, SDLoraLoader):
            state_dict, network_alphas = lora_loader.lora_state_dict(path, unet_config=model.config)
            lora_loader.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=model, adapter_name=path)
        elif isinstance(lora_loader, FLUXLoraLoader):
            state_dict, network_alphas = lora_loader.lora_state_dict(path, return_alphas=True)
            lora_loader.load_lora_into_transformer(state_dict, network_alphas=network_alphas, transformer=model, adapter_name=path)
        else:
            raise ValueError(f"Unsupported LoRA loader: {lora_loader}")

    model.set_adapters(paths, weights=weights)
    # NOTE: fuse_lora an experimental API in Diffusers
    model.fuse_lora(adapter_names=paths, lora_scale=scale)
    model.unload_lora()
    return model
