#
# Copyright 2022 The HuggingFace Inc. team.
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

from collections import OrderedDict
from copy import copy
import numpy as np
import os
import math
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
import random
from scipy import integrate
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Engine():
    def __init__(
        self,
        model_name,
        engine_dir,
    ):
        self.engine_path = os.path.join(engine_dir, model_name+'.plan')
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(self, onnx_path, fp16, input_profile=None, enable_preview=False):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        preview_features = []
        if enable_preview:
            trt_version = [int(i) for i in trt.__version__.split(".")]
            # FASTER_DYNAMIC_SHAPES_0805 should only be used for TRT 8.5.1 or above.
            if trt_version[0] > 8 or \
                (trt_version[0] == 8 and (trt_version[1] > 5 or (trt_version[1] == 5 and trt_version[2] >= 1))):
                preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]

        engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=fp16, profiles=[p],
            preview_features=preview_features))
        save_engine(engine, path=self.engine_path)

    def activate(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
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
        pred_original_sample = latents - sigma * output
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

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.algorithm_type = algorithm_type
        self.predict_epsilon = predict_epsilon
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.lower_order_final = lower_order_final

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
            if self.predict_epsilon:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            else:
                x0_pred = model_output
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
            if self.predict_epsilon:
                return model_output
            else:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon

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

def save_image(images, image_path_dir, image_name_prefix):
    """
    Save the generated images to png files.
    """
    images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    for i in range(images.shape[0]):
        image_path  = os.path.join(image_path_dir, image_name_prefix+str(i+1)+'-'+str(random.randint(1000,9999))+'.png')
        print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
        Image.fromarray(images[i]).save(image_path)
