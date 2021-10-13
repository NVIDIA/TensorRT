#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

# Function Decorators #
def use_cuda(func: Callable):
    """
    Tries to send all parameters of a given function to cuda device if user supports it.
    Object must have a "to(device: str)" and maps to target device "cuda"
    Basically, uses torch implementation.

    Wrapped functions musts have keyword argument "use_cuda: bool" which enables
    or disables toggling of cuda.
    """

    def wrapper(*args, **kwargs):
        caller_kwargs = inspect.getcallargs(func, *args, **kwargs)
        assert (
            "use_cuda" in caller_kwargs
        ), "Function must have 'use_cuda' as a parameter."

        if caller_kwargs["use_cuda"] and torch.cuda.is_available():
            new_kwargs = {}
            for k, v in caller_kwargs.items():
                if getattr(v, "to", False):
                    new_kwargs[k] = v.to("cuda")
                else:
                    new_kwargs[k] = v

            return func(**new_kwargs)
        else:
            return func(**caller_kwargs)

    return wrapper
