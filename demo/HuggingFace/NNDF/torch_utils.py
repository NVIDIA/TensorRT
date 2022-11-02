#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch utils used by demo folder."""

import inspect
from typing import Callable

# pytorch
import torch

# NNDF
from NNDF.logger import G_LOGGER

# Function Decorators #
def use_cuda(func: Callable):
    """
    Tries to send all parameters of a given function to cuda device if user supports it.
    Object must have a "to(device: str)" and maps to target device "cuda"
    Basically, uses torch implementation.

    Wrapped functions musts have keyword argument "use_cuda: bool" which enables
    or disables toggling of cuda.
    """

    def _send_args_to_device(caller_kwargs, device):
        new_kwargs = {}
        for k, v in caller_kwargs.items():
            if getattr(v, "to", False):
                new_kwargs[k] = v.to(device)
            else:
                new_kwargs[k] = v
        return new_kwargs

    def wrapper(*args, **kwargs):
        caller_kwargs = inspect.getcallargs(func, *args, **kwargs)
        assert (
            "use_cuda" in caller_kwargs
        ), "Function must have 'use_cuda' as a parameter."

        if caller_kwargs["use_cuda"]:
            new_kwargs = {}
            used_cuda = False
            if torch.cuda.is_available() and caller_kwargs["use_cuda"]:
                new_kwargs = _send_args_to_device(caller_kwargs, "cuda")
                used_cuda = True
            else:
                new_kwargs = _send_args_to_device(caller_kwargs, "cpu")

            try:
                return func(**new_kwargs)
            except RuntimeError as e:
                # If a device has cuda installed but no compatible kernels, cuda.is_available() will still return True.
                # This exception is necessary to catch remaining incompat errors.
                if used_cuda:
                    G_LOGGER.warning("Unable to execute program using cuda compatible device: {}".format(e))
                    G_LOGGER.warning("Retrying using CPU only.")
                    new_kwargs = _send_args_to_device(caller_kwargs, "cpu")
                    new_kwargs["use_cuda"] = False
                    cpu_result = func(**new_kwargs)
                    G_LOGGER.warning("Successfully obtained result using CPU.")
                    return cpu_result
                else:
                    raise e
        else:
            return func(**caller_kwargs)

    return wrapper

def expand_inputs_for_beam_search(
    tensor,
    expand_size: int = 1,
):
    """
    Interleave input tensor with `num_beams`, similar to HuggingFace's _expand_inputs_for_generation() in generation_utils.py
    """
    expanded_return_idx = (
        torch.arange(tensor.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1)
    )
    tensor = tensor.index_select(0, expanded_return_idx.to(tensor.device))

    return tensor
