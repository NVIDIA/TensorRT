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

import torch
from cuda.bindings import runtime as cudart


class ModelMemoryManager:
    """
    Context manager for efficiently loading and unloading models to optimize VRAM usage.

    This class provides a context to temporarily load models into GPU memory for inference
    and automatically unload them afterward. It's especially useful in low VRAM environments
    where models need to be swapped in and out of GPU memory.

    Args:
        parent: The parent class instance that contains the model references and resources.
        model_names (list): List of model names to load and unload.
        low_vram (bool, optional): If True, enables VRAM optimization. If False, the context manager does nothing. Defaults to False.
    """

    def __init__(self, parent, model_names, low_vram=False):
        self.parent = parent
        self.model_names = model_names
        self.low_vram = low_vram

    def __enter__(self):
        if not self.low_vram:
            return
        for model_name in self.model_names:
            if not self.parent.torch_fallback[model_name]:
                # creating engine object (load from plan file)
                self.parent.engine[model_name].load()
                # allocate device memory
                _, shared_device_memory = cudart.cudaMalloc(self.parent.device_memory_sizes[model_name])
                self.parent.shared_device_memory = shared_device_memory
                # creating context
                self.parent.engine[model_name].activate(device_memory=self.parent.shared_device_memory)
                # creating input and output buffer
                self.parent.engine[model_name].allocate_buffers(
                    shape_dict=self.parent.shape_dicts[model_name], device=self.parent.device
                )
            else:
                print(f"[I] Reloading torch model {model_name} from cpu.")
                self.parent.torch_models[model_name] = self.parent.torch_models[model_name].to("cuda")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.low_vram:
            return
        for model_name in self.model_names:
            if not self.parent.torch_fallback[model_name]:
                self.parent.engine[model_name].deallocate_buffers()
                self.parent.engine[model_name].deactivate()
                self.parent.engine[model_name].unload()
                cudart.cudaFree(self.parent.shared_device_memory)
            else:
                print(f"[I] Offloading torch model {model_name} to cpu.")
                self.parent.torch_models[model_name] = self.parent.torch_models[model_name].to("cpu")
                torch.cuda.empty_cache()
