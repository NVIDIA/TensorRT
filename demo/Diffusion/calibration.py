#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ReduceOp
from utilities import PercentileAmaxes

from ammo.torch.quantization.model_calib import (
    enable_stats_collection,
    finish_stats_collection,
    max_calibrate,
)
from ammo.torch.quantization.utils import is_quantized_linear


def precentile_calib_mode(base_unet, quant_config={}):
    def compute_amax(self, all_reduce=True):
        """Return the absolute max of all tensors collected."""
        if (
            self._calib_amax is not None
            and all_reduce
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            tmp_amax = self._calib_amax.clone()
            dist.all_reduce(tmp_amax, op=ReduceOp.MAX)
            self._calib_amax.copy_(tmp_amax)
        if self._track_amax:
            up_lim = int(self._amaxs.total_step * self._amaxs.percentile)
            if up_lim <= 0:
                up_lim = 1
            amaxs_values = [self._amaxs.data[i] for i in range(0, up_lim)]
            act_amax = (
                torch.tensor(np.vstack(amaxs_values).min(axis=0))
                .float()
                .squeeze(0)
                .to(self._calib_amax.device)
                .to(self._calib_amax.dtype)
            )
            return act_amax
        return self._calib_amax

    for _, module in base_unet.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.input_quantizer._calibrator._track_amax = True
            module.input_quantizer._calibrator._amaxs = PercentileAmaxes(
                total_step=quant_config["base-step"], percentile=quant_config["percentile"]
            )
            module.input_quantizer._calibrator.compute_amax = types.MethodType(
                compute_amax, module.input_quantizer._calibrator
            )


@torch.no_grad()
def smoothquant(model, forward_loop=None):
    """
    Rewrite the original SmoothQuant method
    """
    assert forward_loop is not None, "forward_loop must be provided for smoothquant"
    max_calibrate(model, forward_loop)

    smoothed_modules = 0
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                print(f"Warning: {name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                print(f"Warning: only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                print(f"Warning: only per-channel smoothing is supported, skip {name}")
                continue

            alpha = 1.0
            if hasattr(module, "alpha"):
                alpha = module.alpha
            assert (
                module.input_quantizer._amax.numel() > 1
            ), f"Error: {name} has only one channel to smooth"

            # It is important to keep scaling math in fp32 to be numerically safe
            act_amax = module.input_quantizer.amax.float()

            act_device = act_amax.device

            # If model is split across devices, this tensor may be on wrong one
            act_amax = act_amax.to(module.weight.device)

            weight_scale = module.weight.abs().max(dim=0, keepdim=True)[0]
            scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()

            # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
            epsilon = 1.0 / (1 << 31)
            if act_amax.min() <= epsilon:
                zero_mask = act_amax <= epsilon
                scale_a[zero_mask] = 1
            inv_scale_a = 1.0 / scale_a
            inv_scale_a = inv_scale_a.squeeze()[None, :]

            # Use per-tensor quantization for activation, add a pre-quantization scale vector
            module.input_quantizer.pre_quant_scale = scale_a.to(module.weight.dtype).to(act_device)
            module.input_quantizer._axis = None
            delattr(module.input_quantizer, "_amax")
            module.input_quantizer.amax = torch.tensor(
                (act_amax * scale_a).max().item(),
                dtype=module.weight.dtype,
                device=module.weight.device,
            )

            # Multiply weight by inv_scale_a and recalibrate
            module.weight.detach().copy_(
                (module.weight.float() * inv_scale_a).to(module.weight.dtype)
            )

            enable_stats_collection(module.weight_quantizer)
            module.weight_quantizer(module.weight)
            finish_stats_collection(module.weight_quantizer)

            smoothed_modules += 1
    print(f"Smoothed {smoothed_modules} modules")


def calibrate(
    model: nn.Module,
    algorithm: Union[str, dict, None] = "max",
    forward_loop: Optional[Callable] = None,
) -> None:
    if algorithm is None:
        return

    if isinstance(algorithm, str):
        kwargs = {}
    elif isinstance(algorithm, dict):
        kwargs = algorithm.copy()
        algorithm = kwargs.pop("method")
    else:
        raise TypeError(f"Unsupported type for algorithm: {type(algorithm)}")

    if algorithm == "smoothquant":
        smoothquant(model, forward_loop)
    elif algorithm == "max":
        max_calibrate(model, forward_loop)
    else:
        raise ValueError(f"Unsupported calibration algorithm: {algorithm}")


def reg_alpha_qkv(base_unet, alpha):
    """
    Only apply alpha to QKV layers
    """
    for name, module in base_unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "to_q" in name or "to_k" in name or "to_v" in name:
                module.alpha = alpha

