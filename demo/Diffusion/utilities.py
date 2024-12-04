#
# Copyright (c) Alibaba, Inc. and its affiliates.
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warnings
from importlib import import_module
from collections import OrderedDict
from cuda import cudart
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import load_image
from enum import Enum, auto
import gc
from io import BytesIO
import numpy as np
import onnx
from onnx import numpy_helper
import os
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine
)
from polygraphy.logger import G_LOGGER
import random
import requests
import tensorrt as trt
import torch
import types
import gc

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def GiB(val):
    return val * 1 << 30

# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL     : torch.bool,
    trt.DataType.UINT8    : torch.uint8,
    trt.DataType.INT8     : torch.int8,
    trt.DataType.INT32    : torch.int32,
    trt.DataType.INT64    : torch.int64,
    trt.DataType.HALF     : torch.float16,
    trt.DataType.FLOAT    : torch.float32,
    trt.DataType.BF16     : torch.bfloat16
}

# Define valid optimization levels for TensorRT engine build
VALID_OPTIMIZATION_LEVELS = list(range(6))

def import_from_diffusers(model_name, module_name):
    try:
        module = import_module(module_name)
        return getattr(module, model_name)
    except ImportError:
        warnings.warn(f"Failed to import {module_name}. The {model_name} model will not be available.", ImportWarning)
    except AttributeError:
        warnings.warn(f"The {model_name} model is not available in the installed version of diffusers.", ImportWarning)
    return None

def unload_model(model):
    if model:
        del model
        torch.cuda.empty_cache()
        gc.collect()

def replace_lora_layers(model):
    def lora_forward(self, x, scale=None):
        return self._torch_forward(x)

    for name, module in model.named_modules():
        if isinstance(module, LoRACompatibleConv):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias

            new_conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias is not None,
            )

            new_conv.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_conv.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleConv layer with the Conv2d layer
            path = name.split(".")
            sub_module = model
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_conv)
            new_conv._torch_forward = new_conv.forward
            new_conv.forward = types.MethodType(lora_forward, new_conv)

        elif isinstance(module, LoRACompatibleLinear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias

            new_linear = torch.nn.Linear(in_features, out_features, bias=bias is not None)

            new_linear.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_linear.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleLinear layer with the Linear layer
            path = name.split(".")
            sub_module = model
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_linear)
            new_linear._torch_forward = new_linear.forward
            new_linear.forward = types.MethodType(lora_forward, new_linear)

def merge_loras(model, lora_loader):
    paths, weights, scale = lora_loader.paths, lora_loader.weights, lora_loader.scale
    for i, path in enumerate(paths):
        print(f"[I] Loading LoRA: {path}, weight {weights[i]}")
        state_dict, network_alphas = lora_loader.lora_state_dict(path, unet_config=model.config)
        lora_loader.load_lora_into_unet(state_dict, network_alphas=network_alphas,
        unet=model, adapter_name=path)

    model.set_adapters(paths, weights=weights)
    # NOTE: fuse_lora an experimental API in Diffusers
    model.fuse_lora(adapter_names=paths, lora_scale=scale)
    model.unload_lora()
    return model

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
    IMG2VID = auto()
    INPAINT = auto()
    CONTROLNET = auto()
    XL_BASE = auto()
    XL_REFINER = auto()
    CASCADE_PRIOR = auto()
    CASCADE_DECODER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_img2vid(self):
        return self == self.IMG2VID

    def is_inpaint(self):
        return self == self.INPAINT

    def is_controlnet(self):
        return self == self.CONTROLNET

    def is_sd_xl_base(self):
        return self == self.XL_BASE

    def is_sd_xl_refiner(self):
        return self == self.XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()

    def is_cascade_prior(self):
        return self == self.CASCADE_PRIOR

    def is_cascade_decoder(self):
        return self == self.CASCADE_DECODER

    def is_cascade(self):
        return self.is_cascade_prior() or self.is_cascade_decoder()

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

    def refit(self, refit_weights, updated_weight_names):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        refitted_weights = set()

        def refit_single_weight(trt_weight_name):
            # get weight from state dict
            trt_datatype = refitter.get_weights_prototype(trt_weight_name).dtype
            refit_weights[trt_weight_name] = refit_weights[trt_weight_name].to(trt_to_torch_dtype_dict[trt_datatype])

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name]))
            trt_wt_location = trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in updated_weight_names:
                continue

            refit_single_weight(trt_weight_name)

        # iterate through missing weights required by tensorrt - addresses the case where lora_scale=0
        for trt_weight_name in refitter.get_missing_weights():
            refit_single_weight(trt_weight_name)

        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(self,
        onnx_path,
        strongly_typed=False,
        fp16=True,
        bf16=False,
        tf32=False,
        int8=False,
        fp8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        native_instancenorm=True,
        verbose=False,
        weight_streaming=False,
        **extra_build_args
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        if not enable_all_tactics:
            extra_build_args['tactic_sources'] = []

        flags = []
        if native_instancenorm:
            flags.append(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

        # Weight streaming requires the engine to have strong typing, therefore builder flags specifying precision, such as int8 and fp16, should not be enabled.
        # Please find more details in our developer guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#streaming-weights.
        if weight_streaming:
            strongly_typed = True
            fp16 = False
            bf16 = False
            int8 = False
            fp8 = False

        network = network_from_onnx_path(
            onnx_path,
            flags=flags,
            strongly_typed=strongly_typed
        )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        with G_LOGGER.verbosity(G_LOGGER.EXTRA_VERBOSE if verbose else G_LOGGER.ERROR):
            engine = engine_from_network(
                network,
                config=CreateConfig(fp16=fp16,
                    bf16=bf16,
                    tf32=tf32,
                    int8=int8,
                    fp8=fp8,
                    refittable=enable_refit,
                    profiles=[p],
                    load_timing_cache=timing_cache,
                    weight_streaming=weight_streaming,
                    **extra_build_args
                ),
                save_timing_cache=timing_cache
            )
            save_engine(engine, path=self.engine_path)

    def load(self, weight_streaming=False, weight_streaming_budget_percentage=None):
        if self.engine is not None:
            print(f"[W]: Engine {self.engine_path} already loaded, skip reloading")
            return
        if not hasattr(self,'engine_bytes_cpu') or self.engine_bytes_cpu is None:
            # keep a cpu copy of the engine to reduce reloading time.
            print(f"Loading TensorRT engine to cpu bytes: {self.engine_path}")
            self.engine_bytes_cpu = bytes_from_path(self.engine_path)
        print(f"Loading TensorRT engine from bytes: {self.engine_path}")
        self.engine = engine_from_bytes(self.engine_bytes_cpu)
        if weight_streaming:
            if weight_streaming_budget_percentage is None:
                warnings.warn(f"Weight streaming budget is not set for {self.engine_path}. Weights will not be streamed.")
            else:
                self.engine.weight_streaming_budget_v2 = int(weight_streaming_budget_percentage / 100 * self.engine.streamable_weights_size)
    
    def unload(self):
        if self.engine is not None:
            print(f"Unloading TensorRT engine: {self.engine_path}")
            del self.engine
            self.engine = None
            gc.collect()
        else:
            print(f"[W]: Unload an unloaded engine {self.engine_path}, skip unloading")

    def activate(self, device_memory=None):
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()

    def reactivate(self, device_memory):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype=trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor


    def deallocate_buffers(self):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            del self.tensors[binding]

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

def save_image(images, image_path_dir, image_name_prefix, image_name_suffix):
    """
    Save the generated images to png files.
    """
    for i in range(images.shape[0]):
        image_path  = os.path.join(image_path_dir, image_name_prefix+str(i+1)+'-'+str(random.randint(1000,9999))+'-'+image_name_suffix+'.png')
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

# Taken from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L620
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling tensor2vid or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the terms and conditions at the top of the file
def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

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

def get_refit_weights(state_dict, onnx_opt_path, weight_name_mapping, weight_shape_mapping):
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)
    # Create initializer data hashes
    initializer_hash_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash

    refit_weights = OrderedDict()
    updated_weight_names = set() # save names of updated weights to refit only the required weights
    for wt_name, wt in state_dict.items():
        # query initializer to compare
        initializer_name = weight_name_mapping[wt_name]
        initializer_hash = initializer_hash_mapping[initializer_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[wt_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
        if initializer_hash != wt_hash:
            updated_weight_names.add(initializer_name)
        # Store all weights as the refitter may require unchanged weights too
        # docs: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c
        refit_weights[initializer_name] = wt.contiguous()
    return refit_weights, updated_weight_names

def load_calib_prompts(batch_size, calib_data_path):
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

def load_calibration_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = load_image(img_path)
            if image is not None:
                images.append(image)
    return images


class PercentileAmaxes:
    def __init__(self, total_step, percentile) -> None:
        self.data = {}
        self.total_step = total_step
        self.percentile = percentile
        self.i = 0

    def append(self, item):
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data.keys():
            self.data[_cur_step] = item
        else:
            self.data[_cur_step] = np.maximum(self.data[_cur_step], item)
        self.i += 1

def add_arguments(parser):
    # Stable Diffusion configuration
    parser.add_argument('--version', type=str, default="1.5", choices=("1.4", "1.5", "dreamshaper-7", "2.0-base", "2.0", "2.1-base", "2.1", "xl-1.0", "xl-turbo", "svd-xt-1.1", "sd3", "cascade", "flux.1-dev", "flux.1-schnell"), help="Version of Stable Diffusion")
    parser.add_argument('prompt', nargs = '*', help="Text prompt(s) to guide image generation")
    parser.add_argument('--negative-prompt', nargs = '*', default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--batch-size', type=int, default=1, choices=[1, 2, 4], help="Batch size (repeat prompt)")
    parser.add_argument('--batch-count', type=int, default=1, help="Number of images to generate in sequence, one at a time.")
    parser.add_argument('--height', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--denoising-steps', type=int, default=30, help="Number of denoising steps")
    parser.add_argument('--scheduler', type=str, default=None, choices=("DDIM", "DDPM", "EulerA", "Euler", "LCM", "LMSD", "PNDM", "UniPC", "DDPMWuerstchen", "FlowMatchEuler"), help="Scheduler for diffusion process")
    parser.add_argument('--guidance-scale', type=float, default=7.5, help="Value of classifier-free guidance scale (must be greater than 1)")
    parser.add_argument('--lora-scale', type=float, default=1.0, help="Controls how much to influence the outputs with the LoRA parameters. (must between 0 and 1)")
    parser.add_argument('--lora-weight', type=float, nargs='+', default=None, help="The LoRA adapter(s) weights to use with the UNet. (must between 0 and 1)")
    parser.add_argument('--lora-path', type=str, nargs='+', default=None, help="Path to LoRA adaptor. Ex: 'latent-consistency/lcm-lora-sdv1-5'")

    # ONNX export
    parser.add_argument('--onnx-opset', type=int, default=19, choices=range(7,20), help="Select ONNX opset version to target for exported models")
    parser.add_argument('--onnx-dir', default='onnx', help="Output directory for ONNX export")

    # Framework model ckpt
    parser.add_argument('--framework-model-dir', default='pytorch_model', help="Directory for HF saved models")

    # TensorRT engine build
    parser.add_argument('--engine-dir', default='engine', help="Output directory for TensorRT engines")
    parser.add_argument('--int8', action='store_true', help="Apply int8 quantization.")
    parser.add_argument('--fp8', action='store_true', help="Apply fp8 quantization.")
    parser.add_argument('--quantization-level', type=float, default=0.0, choices=[0.0, 1.0, 2.0, 2.5, 3.0, 4.0], help="int8/fp8 quantization level, 1: CNN, 2: CNN + FFN, 2.5: CNN + FFN + QKV, 3: CNN + Almost all Linear (Including FFN, QKV, Proj and others), 4: CNN + Almost all Linear + fMHA, 0: Default to 2.5 for int8 and 4.0 for fp8.")
    parser.add_argument('--optimization-level', type=int, default=None, help=f"Set the builder optimization level to build the engine with. A higher level allows TensorRT to spend more building time for more optimization options. Must be one of {VALID_OPTIMIZATION_LEVELS}.")
    parser.add_argument('--build-static-batch', action='store_true', help="Build TensorRT engines with fixed batch size.")
    parser.add_argument('--build-dynamic-shape', action='store_true', help="Build TensorRT engines with dynamic image shapes.")
    parser.add_argument('--build-enable-refit', action='store_true', help="Enable Refit option in TensorRT engines during build.")
    parser.add_argument('--build-all-tactics', action='store_true', help="Build TensorRT engines using all tactic sources.")
    parser.add_argument('--timing-cache', default=None, type=str, help="Path to the precached timing measurements to accelerate build.")

    # TensorRT inference
    parser.add_argument('--num-warmup-runs', type=int, default=5, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--use-cuda-graph', action='store_true', help="Enable cuda graph")
    parser.add_argument('--nvtx-profile', action='store_true', help="Enable NVTX markers for performance profiling")
    parser.add_argument('--torch-inference', default='', help="Run inference with PyTorch (using specified compilation mode) instead of TensorRT.")

    parser.add_argument('--seed', type=int, default=None, help="Seed for random generator to get consistent results")
    parser.add_argument('--output-dir', default='output', help="Output directory for logs and image artifacts")
    parser.add_argument('--hf-token', type=str, help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    return parser

def process_pipeline_args(args):
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {args.image_height} and {args.width}.")

    max_batch_size = 4
    if args.batch_size > max_batch_size:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed {max_batch_size}.")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

    if args.optimization_level is None:
        if args.int8 or args.fp8:
            args.optimization_level = 4
        else:
            args.optimization_level = 3

    if args.optimization_level not in VALID_OPTIMIZATION_LEVELS:
        raise ValueError(f"Optimization level {args.optimization_level} not valid.  Valid values are: {VALID_OPTIMIZATION_LEVELS}")

    if args.int8 and not any(args.version.startswith(prefix) for prefix in ['xl', '1.4', '1.5', '2.1']):
        raise ValueError(f"int8 quantization is only supported for SDXL, SD1.4, SD1.5 and SD2.1 pipelines.")

    if args.fp8 and not any(args.version.startswith(prefix) for prefix in ('xl', '1.4', '1.5', '2.1', 'flux.1-dev', 'flux.1-schnell')):
        raise ValueError(f"fp8 quantization is only supported for SDXL, SD1.4, SD1.5, SD2.1 and FLUX pipelines.")

    if args.fp8 and args.int8:
        raise ValueError(f"Cannot apply both int8 and fp8 quantization, please choose only one.")

    if args.fp8:
        device_info = torch.cuda.get_device_properties(0)
        version = device_info.major * 10 + device_info.minor
        if version < 89:
            raise ValueError(f"Cannot apply FP8 quantization for GPU with compute capability {version / 10.0}.  Only Ada and Hopper are supported.")

    if args.quantization_level == 0.0:
        def override_quant_level(level : float, dtype_str : str):
            args.quantization_level = level
            print(f"The default quantization level has been set to {level} for {dtype_str}.")

        if args.fp8:
            override_quant_level(3.0 if args.version in ("1.4", "1.5", "flux.1-dev", "flux.1-schnell") else 4.0, "FP8")
        elif args.int8:
            override_quant_level(3.0, "INT8")

    if args.lora_path and not any(args.version.startswith(prefix) for prefix in ('1.5', '2.1', 'xl')):
        raise ValueError(f"LoRA adapter support is only supported for SD1.5, SD2.1 and SDXL pipelines")

    if args.lora_weight:
        for weight in (weight for weight in args.lora_weight if not 0 <= weight <= 1):
            raise ValueError(f"LoRA adapter weights must be between 0 and 1, provided {weight}")

    if not 0 <= args.lora_scale <= 1:
        raise ValueError(f"LoRA scale value must be between 0 and 1, provided {args.lora_scale}")

    kwargs_init_pipeline = {
        'version': args.version,
        'max_batch_size': max_batch_size,
        'denoising_steps': args.denoising_steps,
        'scheduler': args.scheduler,
        'guidance_scale': args.guidance_scale,
        'output_dir': args.output_dir,
        'hf_token': args.hf_token,
        'verbose': args.verbose,
        'nvtx_profile': args.nvtx_profile,
        'use_cuda_graph': args.use_cuda_graph,
        'lora_scale': args.lora_scale,
        'lora_weight': args.lora_weight,
        'lora_path': args.lora_path,
        'framework_model_dir': args.framework_model_dir,
        'torch_inference': args.torch_inference,
    }

    kwargs_load_engine = {
        'onnx_opset': args.onnx_opset,
        'opt_batch_size': args.batch_size,
        'opt_image_height': args.height,
        'opt_image_width': args.width,
        'optimization_level': args.optimization_level,
        'static_batch': args.build_static_batch,
        'static_shape': not args.build_dynamic_shape,
        'enable_all_tactics': args.build_all_tactics,
        'enable_refit': args.build_enable_refit,
        'timing_cache': args.timing_cache,
        'int8': args.int8,
        'fp8': args.fp8,
        'quantization_level': args.quantization_level,
    }

    args_run_demo = (args.prompt, args.negative_prompt, args.height, args.width, args.batch_size, args.batch_count, args.num_warmup_runs, args.use_cuda_graph)

    return kwargs_init_pipeline, kwargs_load_engine, args_run_demo
