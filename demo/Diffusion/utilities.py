#
# Copyright 2022 The HuggingFace Inc. team.
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

from collections import OrderedDict
from typing import List
from copy import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import math
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
import random
from scipy import integrate
import tensorrt as trt
import torch
import requests
from io import BytesIO
from cuda import cudart
from enum import Enum, auto

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class PIPELINE_TYPE(Enum):
    TXT2IMG = auto()
    IMG2IMG = auto()
    INPAINT = auto()
    SD_XL_BASE = auto()
    SD_XL_REFINER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_inpaint(self):
        return self == self.INPAINT

    def is_sd_xl_base(self):
        return self == self.SD_XL_BASE

    def is_sd_xl_refiner(self):
        return self == self.SD_XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()

class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTKERNEL"] = node.name+"_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTBIAS"] = node.name+"_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name
        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name+"_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name+"_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None


        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name+"_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name+"_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, update_output_names=None):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors


class LMSDiscreteScheduler():
    def __init__(
        self,
        device = 'cuda',
        beta_start = 0.00085,
        beta_end = 0.012,
        num_train_timesteps = 1000,
        steps_offset = 0,
        prediction_type = 'epsilon'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.order = 4

        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = (torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        self.device = device
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type

    def set_timesteps(self, steps):
        self.num_inference_steps = steps

        timesteps = np.linspace(0, self.num_train_timesteps - 1, steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=self.device)

        # Move all timesteps to correct device beforehand
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device).float()
        self.derivatives = []

    def scale_model_input(self, sample: torch.FloatTensor, idx, *args, **kwargs) -> torch.FloatTensor:
        return sample * self.latent_scales[idx]

    def configure(self):
        order = self.order
        self.lms_coeffs = []
        self.latent_scales = [1./((sigma**2 + 1) ** 0.5) for sigma in self.sigmas]

        def get_lms_coefficient(order, t, current_order):
            """
            Compute a linear multistep coefficient.
            """
            def lms_derivative(tau):
                prod = 1.0
                for k in range(order):
                    if current_order == k:
                        continue
                    prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
                return prod
            integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]
            return integrated_coeff

        for step_index in range(self.num_inference_steps):
            order = min(step_index + 1, order)
            self.lms_coeffs.append([get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)])

    def step(self, output, latents, idx, timestep):
        # compute the previous noisy sample x_t -> x_t-1
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        sigma = self.sigmas[idx]
        if self.prediction_type == "epsilon":
            pred_original_sample = latents - sigma * output
        elif self.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = output * (-sigma / (sigma**2 + 1) ** 0.5) + (latents / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        # 2. Convert to an ODE derivative
        derivative = (latents - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > self.order:
            self.derivatives.pop(0)
        # 3. Compute previous sample based on the derivatives path
        prev_sample = latents + sum(
            coeff * derivative for coeff, derivative in zip(self.lms_coeffs[idx], reversed(self.derivatives))
        )

        return prev_sample

    def add_noise(self, init_latents, noise, idx, latent_timestep):
        sigma = self.sigmas[idx]

        noisy_latents = init_latents + noise * sigma
        return noisy_latents

class DDIMScheduler():
    def __init__(
        self,
        device='cuda',
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 1,
        prediction_type: str = "epsilon",
    ):
        # this schedule is very specific to the latent diffusion model.
        betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.steps_offset = steps_offset
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.device = device

    def configure(self):
        variance = np.zeros(self.num_inference_steps, dtype=np.float32)
        for idx, timestep in enumerate(self.timesteps):
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
            variance[idx] = self._get_variance(timestep, prev_timestep)
        self.variance = torch.from_numpy(variance).to(self.device)

        timesteps = self.timesteps.long().cpu()
        self.alphas_cumprod = self.alphas_cumprod[timesteps].to(self.device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(self.device)

    def scale_model_input(self, sample: torch.FloatTensor, idx, *args, **kwargs) -> torch.FloatTensor:
        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(self.device)
        self.timesteps += self.steps_offset

    def step(self, model_output, sample, idx, timestep,
             eta: float = 0.0,
             use_clipped_model_output: bool = False,
             generator=None,
             variance_noise: torch.FloatTensor = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        prev_idx = idx + 1
        alpha_prod_t = self.alphas_cumprod[idx]
        alpha_prod_t_prev = self.alphas_cumprod[prev_idx] if prev_idx < self.num_inference_steps  else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.variance[idx]
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                )
            variance = variance ** (0.5) * eta * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self, init_latents, noise, idx, latent_timestep):
        sqrt_alpha_prod = self.alphas_cumprod[idx] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[idx]) ** 0.5
        noisy_latents = sqrt_alpha_prod * init_latents + sqrt_one_minus_alpha_prod * noise

        return noisy_latents


class EulerAncestralDiscreteScheduler():
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device = 'cuda',
        steps_offset = 0,
        prediction_type = "epsilon"
    ):
        # this schedule is very specific to the latent diffusion model.
        betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type

    def scale_model_input(
        self, sample: torch.FloatTensor, idx, timestep, *args, **kwargs
    ) -> torch.FloatTensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=self.device)
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device)

    def configure(self):
        dts = np.zeros(self.num_inference_steps, dtype=np.float32)
        sigmas_up = np.zeros(self.num_inference_steps, dtype=np.float32)
        for idx, timestep in enumerate(self.timesteps):
            step_index = (self.timesteps == timestep).nonzero().item()
            sigma = self.sigmas[step_index]

            sigma_from = self.sigmas[step_index]
            sigma_to = self.sigmas[step_index + 1]
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
            dt = sigma_down - sigma
            dts[idx] = dt
            sigmas_up[idx] = sigma_up

        self.dts = torch.from_numpy(dts).to(self.device)
        self.sigmas_up = torch.from_numpy(sigmas_up).to(self.device)

    def step(
        self, model_output, sample, idx, timestep,
        generator = None,
    ):
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_up = self.sigmas_up[idx]

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = self.dts[idx]

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = torch.randn(model_output.shape, dtype=model_output.dtype, device=device, generator=generator).to(
            device
        )

        prev_sample = prev_sample + noise * sigma_up

        return prev_sample

    def add_noise(
        self, original_samples, noise, idx, timestep=None):
        step_index = (self.timesteps == timestep).nonzero().item()
        noisy_samples = original_samples + noise * self.sigmas[step_index]
        return noisy_samples


class DPMScheduler():
    def __init__(
        self,
        beta_start = 0.00085,
        beta_end = 0.012,
        num_train_timesteps = 1000,
        solver_order = 2,
        predict_epsilon = True,
        thresholding = False,
        dynamic_thresholding_ratio = 0.995,
        sample_max_value = 1.0,
        algorithm_type = "dpmsolver++",
        solver_type = "midpoint",
        lower_order_final = True,
        device = 'cuda',
        steps_offset = 0,
        prediction_type = 'epsilon'
    ):
        # this schedule is very specific to the latent diffusion model.
        self.betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )

        self.device = device
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.steps_offset = steps_offset

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.algorithm_type = algorithm_type
        self.predict_epsilon = predict_epsilon
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.lower_order_final = lower_order_final
        self.prediction_type = prediction_type

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++"]:
            raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")
        if solver_type not in ["midpoint", "heun"]:
            raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        # setable values
        self.num_inference_steps = None
        self.solver_order = solver_order
        self.num_train_timesteps = num_train_timesteps
        self.solver_type = solver_type

        self.first_order_first_coef = []
        self.first_order_second_coef = []

        self.second_order_first_coef = []
        self.second_order_second_coef = []
        self.second_order_third_coef = []

        self.third_order_first_coef = []
        self.third_order_second_coef = []
        self.third_order_third_coef = []
        self.third_order_fourth_coef = []

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        return sample

    def configure(self):
        lower_order_nums = 0
        for step_index in range(self.num_inference_steps):
            step_idx = step_index
            timestep = self.timesteps[step_idx]

            prev_timestep = 0 if step_idx == len(self.timesteps) - 1 else self.timesteps[step_idx + 1]

            self.dpm_solver_first_order_coefs_precompute(timestep, prev_timestep)

            timestep_list = [self.timesteps[step_index - 1], timestep]
            self.multistep_dpm_solver_second_order_coefs_precompute(timestep_list, prev_timestep)

            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            self.multistep_dpm_solver_third_order_coefs_precompute(timestep_list, prev_timestep)

            if lower_order_nums < self.solver_order:
                lower_order_nums += 1

    def dpm_solver_first_order_coefs_precompute(self, timestep, prev_timestep):
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            self.first_order_first_coef.append(sigma_t / sigma_s)
            self.first_order_second_coef.append(alpha_t * (torch.exp(-h) - 1.0))
        elif self.algorithm_type == "dpmsolver":
            self.first_order_first_coef.append(alpha_t / alpha_s)
            self.first_order_second_coef.append(sigma_t * (torch.exp(h) - 1.0))

    def multistep_dpm_solver_second_order_coefs_precompute(self, timestep_list, prev_timestep):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h = lambda_t - lambda_s0
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                self.second_order_first_coef.append(sigma_t / sigma_s0)
                self.second_order_second_coef.append((alpha_t * (torch.exp(-h) - 1.0)))
                self.second_order_third_coef.append(0.5 * (alpha_t * (torch.exp(-h) - 1.0)))
            elif self.solver_type == "heun":
                self.second_order_first_coef.append(sigma_t / sigma_s0)
                self.second_order_second_coef.append((alpha_t * (torch.exp(-h) - 1.0)))
                self.second_order_third_coef.append(alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0))
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                self.second_order_first_coef.append(alpha_t / alpha_s0)
                self.second_order_second_coef.append((sigma_t * (torch.exp(h) - 1.0)))
                self.second_order_third_coef.append(0.5 * (sigma_t * (torch.exp(h) - 1.0)))
            elif self.solver_type == "heun":
                self.second_order_first_coef.append(alpha_t / alpha_s0)
                self.second_order_second_coef.append((sigma_t * (torch.exp(h) - 1.0)))
                self.second_order_third_coef.append((sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)))

    def multistep_dpm_solver_third_order_coefs_precompute(self, timestep_list, prev_timestep):
        t, s0 = prev_timestep, timestep_list[-1]
        lambda_t, lambda_s0 = (
            self.lambda_t[t],
            self.lambda_t[s0]
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h = lambda_t - lambda_s0
        if self.algorithm_type == "dpmsolver++":
            self.third_order_first_coef.append(sigma_t / sigma_s0)
            self.third_order_second_coef.append(alpha_t * (torch.exp(-h) - 1.0))
            self.third_order_third_coef.append(alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0))
            self.third_order_fourth_coef.append(alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5))
        elif self.algorithm_type == "dpmsolver":
            self.third_order_first_coef.append(alpha_t / alpha_s0)
            self.third_order_second_coef.append(sigma_t * (torch.exp(h) - 1.0))
            self.third_order_third_coef.append(sigma_t * ((torch.exp(h) - 1.0) / h - 1.0))
            self.third_order_fourth_coef.append(sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5))

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
            .round()[::-1][:-1]
            .copy()
            .astype(np.int32)
        )
        self.timesteps = torch.from_numpy(timesteps).to(self.device)
        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0

    def convert_model_output(
        self, model_output, timestep, sample
    ):
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.algorithm_type == "dpmsolver++":
            if self.prediction_type == "epsilon":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or"
                    " `v_prediction` for the DPMScheduler."
                )

            if self.thresholding:
                # Dynamic thresholding in https://arxiv.org/abs/2205.11487
                dynamic_max_val = torch.quantile(
                    torch.abs(x0_pred).reshape((x0_pred.shape[0], -1)), self.dynamic_thresholding_ratio, dim=1
                )
                dynamic_max_val = torch.maximum(
                    dynamic_max_val,
                    self.sample_max_value * torch.ones_like(dynamic_max_val).to(dynamic_max_val.device),
                )[(...,) + (None,) * (x0_pred.ndim - 1)]
                x0_pred = torch.clamp(x0_pred, -dynamic_max_val, dynamic_max_val) / dynamic_max_val
            return x0_pred
        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.algorithm_type == "dpmsolver":
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon` or"
                    " `v_prediction` for the DPMScheduler."
                )

    def dpm_solver_first_order_update(
        self,
        idx,
        model_output,
        sample
    ):
        first_coef = self.first_order_first_coef[idx]
        second_coef = self.first_order_second_coef[idx]

        if self.algorithm_type == "dpmsolver++":
            x_t = first_coef * sample - second_coef * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = first_coef * sample - second_coef * model_output
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        idx,
        model_output_list,
        timestep_list,
        prev_timestep,
        sample
    ):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        first_coef = self.second_order_first_coef[idx]
        second_coef = self.second_order_second_coef[idx]
        third_coef = self.second_order_third_coef[idx]

        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    first_coef * sample
                    - second_coef * D0
                    - third_coef * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    first_coef * sample
                    - second_coef * D0
                    + third_coef * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    first_coef * sample
                    - second_coef * D0
                    - third_coef * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    first_coef * sample
                    - second_coef * D0
                    - third_coef * D1
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        idx,
        model_output_list,
        timestep_list,
        prev_timestep,
        sample
    ):
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        first_coef = self.third_order_first_coef[idx]
        second_coef = self.third_order_second_coef[idx]
        third_coef = self.third_order_third_coef[idx]
        fourth_coef = self.third_order_fourth_coef[idx]

        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                first_coef * sample
                - second_coef * D0
                + third_coef * D1
                - fourth_coef * D2
            )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                first_coef * sample
                - second_coef * D0
                - third_coef * D1
                - fourth_coef * D2
            )
        return x_t

    def step(self, output, latents, step_index, timestep):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.lower_order_final and len(self.timesteps) < 15
        )

        output = self.convert_model_output(output, timestep, latents)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = output

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(step_index, output, latents)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                step_index, self.model_outputs, timestep_list, prev_timestep, latents
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                step_index, self.model_outputs, timestep_list, prev_timestep, latents
            )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        return prev_sample

    def add_noise(self, init_latents, noise, idx, latent_timestep):
        self.alphas_cumprod = self.alphas_cumprod.to(device=init_latents.device, dtype=init_latents.dtype)
        timestep = latent_timestep.to(init_latents.device).long()

        sqrt_alpha_prod = self.alphas_cumprod[timestep] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timestep]) ** 0.5
        noisy_latents = sqrt_alpha_prod * init_latents + sqrt_one_minus_alpha_prod * noise

        return noisy_latents


class PNDMScheduler():
    def __init__(
        self,
        device = 'cuda',
        beta_start = 0.00085,
        beta_end = 0.012,
        num_train_timesteps = 1000,
        steps_offset: int = 0,
        prediction_type = 'epsilon'
    ):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.pndm_order = 4

        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = (torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device=self.device)
        self.final_alpha_cumprod = self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        self.steps_offset = steps_offset

        # running values
        self.counter = 0
        self.cur_sample = None
        self.ets = []
        self.prediction_type = prediction_type

    def set_timesteps(self, steps):
        self.num_inference_steps = steps

        self.step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        timesteps = (np.arange(0, self.num_inference_steps) * self.step_ratio).round()
        timesteps += self.steps_offset

        # for some models like stable diffusion the prk steps can/should be skipped to produce better results
        plms_timesteps = np.concatenate([timesteps[:-1], timesteps[-2:-1], timesteps[-1:]])[::-1].copy()
        self.timesteps = torch.from_numpy(plms_timesteps).to(self.device)

        # reset running values
        self.counter = 0
        self.cur_sample = None
        self.ets = []

    def scale_model_input(self, sample: torch.FloatTensor, idx, *args, **kwargs) -> torch.FloatTensor:
        return sample

    def configure(self):
        self.alphas_cumprod_prev = torch.roll(self.alphas_cumprod, shifts=self.step_ratio)
        self.alphas_cumprod_prev[:self.step_ratio] = self.final_alpha_cumprod
        self.sample_coeff = (self.alphas_cumprod_prev / self.alphas_cumprod) ** (0.5)

        self.beta_cumprod = 1 - self.alphas_cumprod
        self.beta_cumprod_prev = 1 - self.alphas_cumprod_prev
        self.model_output_denom_coeff = self.alphas_cumprod * (self.beta_cumprod_prev) ** (0.5) + (
            self.alphas_cumprod * self.beta_cumprod * self.alphas_cumprod_prev) ** (0.5)

        timesteps = self.timesteps.cpu().long()

        self.alphas_cumprod = self.alphas_cumprod[timesteps]
        self.beta_cumprod = self.beta_cumprod[timesteps]
        self.alphas_cumprod_prev = self.alphas_cumprod_prev[timesteps]
        self.sample_coeff = self.sample_coeff[timesteps]
        self.model_output_denom_coeff = self.model_output_denom_coeff[timesteps]

    def step(self, output, sample, idx, timestep):
        # step_plms: propagate the sample with the linear multi-step method. This has one forward pass with multiple
        # times to approximate the solution.

        # prev_timestep = timestep - self.step_ratio

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(output)
        # else:
        #     prev_timestep = timestep
        #     timestep = timestep + self.step_ratio

        if len(self.ets) == 1 and self.counter == 0:
            output = output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            output = (output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        if self.prediction_type == "v_prediction":
            output = (self.alphas_cumprod[idx]**0.5) * output + (self.beta_cumprod[idx]**0.5) * sample
        elif self.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon` or `v_prediction`"
            )

        prev_sample = (
            self.sample_coeff[idx] * sample - (self.alphas_cumprod_prev[idx] - self.alphas_cumprod[idx]) * output / self.model_output_denom_coeff[idx]
        )
        self.counter += 1

        return prev_sample

    def add_noise(self, init_latents, noise, idx, latent_timestep):
        sqrt_alpha_prod = self.alphas_cumprod[idx] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[idx]) ** 0.5
        noisy_latents = sqrt_alpha_prod * init_latents + sqrt_one_minus_alpha_prod * noise

        return noisy_latents
    
class UniPCMultistepScheduler():
    def __init__(
        self,
        device = 'cuda',
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
    ):  
        self.device = device
        self.betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.last_sample = None
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final

    def set_timesteps(self, num_inference_steps: int):
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
            .round()[::-1][:-1]
            .copy()
            .astype(np.int64)
        )

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(self.device)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    def convert_model_output(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor) -> torch.FloatTensor:
        if self.predict_x0:
            if self.prediction_type == "epsilon":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.FloatTensor,
        prev_timestep: int,
        sample: torch.FloatTensor,
        order: int,
    ) -> torch.FloatTensor:
        
        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = self.timestep_list[-1], prev_timestep
        m0 = model_output_list[-1]
        x = sample


        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=self.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=self.device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=self.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.FloatTensor,
        this_timestep: int,
        last_sample: torch.FloatTensor,
        this_sample: torch.FloatTensor,
        order: int,
    ) -> torch.FloatTensor:
        
        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = timestep_list[-1], this_timestep
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=self.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=self.device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=self.device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()

        use_corrector = (
            step_index > 0 and step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, timestep, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                this_timestep=timestep,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # now prepare to run the predictor
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            prev_timestep=prev_timestep,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=self.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(self.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def configure(self):
        pass

    def __len__(self):
        return self.num_train_timesteps

def save_image(images, image_path_dir, image_name_prefix):
    """
    Save the generated images to png files.
    """
    images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    for i in range(images.shape[0]):
        image_path  = os.path.join(image_path_dir, image_name_prefix+str(i+1)+'-'+str(random.randint(1000,9999))+'.png')
        print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
        Image.fromarray(images[i]).save(image_path)

def preprocess_image(image):
    """
    image: torch.Tensor
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous()
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    """
    image: PIL.Image.Image
    mask: PIL.Image.Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def add_arguments(parser):
    # Stable Diffusion configuration
    parser.add_argument('--version', type=str, default="1.5", choices=["1.4", "1.5", "2.0-base", "2.0", "2.1-base", "2.1", "xl-1.0"], help="Version of Stable Diffusion")
    parser.add_argument('prompt', nargs = '*', help="Text prompt(s) to guide image generation")
    parser.add_argument('--negative-prompt', nargs = '*', default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--repeat-prompt', type=int, default=1, choices=[1, 2, 4, 8, 16], help="Number of times to repeat the prompt (batch size multiplier)")
    parser.add_argument('--height', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--denoising-steps', type=int, default=50, help="Number of denoising steps")

    # ONNX export
    parser.add_argument('--onnx-opset', type=int, default=17, choices=range(7,18), help="Select ONNX opset version to target for exported models")
    parser.add_argument('--onnx-dir', default='onnx', help="Output directory for ONNX export")
    parser.add_argument('--onnx-refit-dir', help="ONNX models to load the weights from")
    parser.add_argument('--force-onnx-export', action='store_true', help="Force ONNX export of CLIP, UNET, and VAE models")
    parser.add_argument('--force-onnx-optimize', action='store_true', help="Force ONNX optimizations for CLIP, UNET, and VAE models")

    # Framework model ckpt
    parser.add_argument('--framework-model-dir', default='pytorch_model', help="Directory for HF saved models")

    # TensorRT engine build
    parser.add_argument('--engine-dir', default='engine', help="Output directory for TensorRT engines")
    parser.add_argument('--force-engine-build', action='store_true', help="Force rebuilding the TensorRT engine")
    parser.add_argument('--build-static-batch', action='store_true', help="Build TensorRT engines with fixed batch size.")
    parser.add_argument('--build-dynamic-shape', action='store_true', help="Build TensorRT engines with dynamic image shapes.")
    parser.add_argument('--build-enable-refit', action='store_true', help="Enable Refit option in TensorRT engines during build.")
    parser.add_argument('--build-preview-features', action='store_true', help="Build TensorRT engines with preview features.")
    parser.add_argument('--build-all-tactics', action='store_true', help="Build TensorRT engines using all tactic sources.")
    parser.add_argument('--timing-cache', default=None, type=str, help="Path to the precached timing measurements to accelerate build.")

    # TensorRT inference
    parser.add_argument('--num-warmup-runs', type=int, default=5, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--nvtx-profile', action='store_true', help="Enable NVTX markers for performance profiling")
    parser.add_argument('--seed', type=int, default=None, help="Seed for random generator to get consistent results")
    parser.add_argument('--use-cuda-graph', action='store_true', help="Enable cuda graph")

    parser.add_argument('--output-dir', default='output', help="Output directory for logs and image artifacts")
    parser.add_argument('--hf-token', type=str, help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    return parser


