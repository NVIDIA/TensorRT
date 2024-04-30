#
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

from collections import OrderedDict
from cuda import cudart
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils.torch_utils import randn_tensor
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

def merge_loras(model, lora_dict, lora_alphas, lora_scales):
    assert len(lora_scales) == len(lora_dict)
    for path, lora in lora_dict.items():
        print(f"[I] Fusing LoRA: {path}, scale {lora_scales[path]}")
        model.load_attn_procs(lora, network_alphas=lora_alphas[path])
        model.fuse_lora(lora_scale=lora_scales[path])
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
    INPAINT = auto()
    CONTROLNET = auto()
    XL_BASE = auto()
    XL_REFINER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

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

    def refit(self, refit_weights, is_fp16):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in refit_weights:
                continue

            # get weight from state dict
            trt_datatype = trt.DataType.FLOAT
            if is_fp16:
                refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
                trt_datatype = trt.DataType.HALF

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name]))
            trt_wt_location = trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(self,
        onnx_path,
        fp16=True,
        tf32=False,
        int8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        verbose=False,
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

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        with G_LOGGER.verbosity(G_LOGGER.EXTRA_VERBOSE if verbose else G_LOGGER.ERROR):
            engine = engine_from_network(
                network,
                config=CreateConfig(fp16=fp16,
                    tf32=tf32,
                    int8=int8,
                    refittable=enable_refit,
                    profiles=[p],
                    load_timing_cache=timing_cache,
                    **extra_build_args
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
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor


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
            refit_weights[initializer_name] = wt.contiguous()
    return refit_weights

def load_calib_prompts(batch_size, calib_data_path):
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

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
    parser.add_argument('--version', type=str, default="1.5", choices=["1.4", "1.5", "dreamshaper-7", "2.0-base", "2.0", "2.1-base", "2.1", "xl-1.0", "xl-turbo"], help="Version of Stable Diffusion")
    parser.add_argument('prompt', nargs = '*', help="Text prompt(s) to guide image generation")
    parser.add_argument('--negative-prompt', nargs = '*', default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--batch-size', type=int, default=1, choices=[1, 2, 4], help="Batch size (repeat prompt)")
    parser.add_argument('--batch-count', type=int, default=1, help="Number of images to generate in sequence, one at a time.")
    parser.add_argument('--height', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--denoising-steps', type=int, default=30, help="Number of denoising steps")
    parser.add_argument('--scheduler', type=str, default=None, choices=["DDIM", "DDPM", "EulerA", "Euler", "LCM", "LMSD", "PNDM", "UniPC"], help="Scheduler for diffusion process")
    parser.add_argument('--guidance-scale', type=float, default=7.5, help="Value of classifier-free guidance scale (must be greater than 1)")
    parser.add_argument('--lora-scale', type=float, nargs='+', default=None, help="Scale of LoRA weights, default 1 (must between 0 and 1)")
    parser.add_argument('--lora-path', type=str, nargs='+', default=None, help="Path to LoRA adaptor. Ex: 'latent-consistency/lcm-lora-sdv1-5'")

    # ONNX export
    parser.add_argument('--onnx-opset', type=int, default=19, choices=range(7,20), help="Select ONNX opset version to target for exported models")
    parser.add_argument('--onnx-dir', default='onnx', help="Output directory for ONNX export")

    # Framework model ckpt
    parser.add_argument('--framework-model-dir', default='pytorch_model', help="Directory for HF saved models")

    # TensorRT engine build
    parser.add_argument('--engine-dir', default='engine', help="Output directory for TensorRT engines")
    parser.add_argument('--int8', action='store_true', help="Apply int8 quantization.")
    parser.add_argument('--quantization-level', type=float, default=2.5, choices=[1.0, 2.0, 2.5, 3.0], help="int8/fp8 quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC")
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

    if args.int8 and not args.version.startswith('xl'):
        raise ValueError(f"int8 quantization only supported for SDXL pipeline.")

    if args.lora_scale:
        for lora_scale in (lora_scale for lora_scale in args.lora_scale if not 0 <= lora_scale <= 1):
            raise ValueError(f"Scale of LoRA weights must be between 0 and 1, provided {lora_scale}")

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
        'lora_path': args.lora_path,
        'framework_model_dir': args.framework_model_dir,
        'torch_inference': args.torch_inference,
    }

    kwargs_load_engine = {
        'onnx_opset': args.onnx_opset,
        'opt_batch_size': args.batch_size,
        'opt_image_height': args.height,
        'opt_image_width': args.width,
        'static_batch': args.build_static_batch,
        'static_shape': not args.build_dynamic_shape,
        'enable_all_tactics': args.build_all_tactics,
        'enable_refit': args.build_enable_refit,
        'timing_cache': args.timing_cache,
        'int8': args.int8,
        'quantization_level': args.quantization_level,
        'denoising_steps': args.denoising_steps,
    }

    args_run_demo = (args.prompt, args.negative_prompt, args.height, args.width, args.batch_size, args.batch_count, args.num_warmup_runs, args.use_cuda_graph)

    return kwargs_init_pipeline, kwargs_load_engine, args_run_demo
