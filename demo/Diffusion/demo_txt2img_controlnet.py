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

import argparse
from cuda import cudart
import tensorrt as trt
from utilities import TRT_LOGGER, add_arguments
from txt2img_pipeline import Txt2ImgPipeline
from diffusers.utils import load_image
import cv2, controlnet_aux
from PIL import Image
import numpy as np
import os

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion [Controlnet] Txt2Img Demo")
    parser = add_arguments(parser)
    parser.add_argument('--scheduler', type=str, default="DDIM", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")
    # Controlnet arguments configuration
    parser.add_argument('--controlnet-type', nargs='*', type=str, default=None, help="Controlnet type, can be `None`, `str` or `str` list from ['canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble', 'seg']")
    # parser.add_argument('--controlnet-type', nargs='*', type=str, default=["canny", "normal"], help="Controlnet type, can be `None`, `str` or `str` list from ['canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble', 'seg']")
    parser.add_argument('--controlnet-scale', nargs='*', type=float, default=[1.0], help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet, can be `None`, `float` or `float` list")
    
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion [Controlnet] txt2img demo using TensorRT")
    args = parseArgs()

    # Process prompt
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}")
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt
    
    # Controlnet configuration
    if not isinstance(args.controlnet_type, list):
        if args.controlnet_type is None:
            print("[I] StableDiffusion without controlnet")
        else:
            raise ValueError(f"`--controlnet-type` must be of type `str` or `str` list, but is {type(args.controlnet_type)}")
    else:
        print(f"[I] StableDiffusion with controlnet: {args.controlnet_type}")
    controlnet_imgs = None if args.controlnet_type is None else []
    if args.controlnet_type is not None:
        if len(args.controlnet_scale) != len(args.controlnet_type):
            args.controlnet_scale = args.controlnet_scale * (len(args.controlnet_type) // len(args.controlnet_scale))
        
        def resize(x, resolution):
            x = np.array(x, dtype=np.uint8)
            Ht, Wt = resolution
            if x.ndim == 2:
                Ho, Wo = x.shape
                Co = 1
            else:
                Ho, Wo, Co = x.shape
            if Co == 3 or Co == 4 or Co == 1:
                k = float(Ht + Wt) / float(Ho + Wo)
                x = cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
                return Image.fromarray(x)
        for controlnet in args.controlnet_type:
            if controlnet == "canny":
                prompt = ["Stormtrooper's lecture"] * args.repeat_prompt
                negative_prompt = [""] * args.repeat_prompt
                control_image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
                control_image = controlnet_aux.CannyDetector()(control_image)
                controlnet_imgs.append(resize(control_image, [args.height, args.width]))
            elif controlnet == "normal":
                prompt = ["Stormtrooper's lecture"] * args.repeat_prompt
                negative_prompt = [""] * args.repeat_prompt
                control_image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
                control_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(control_image)
                controlnet_imgs.append(resize(control_image, [args.height, args.width]))
            else:
                raise ValueError(f"You should implement the conditonal image of this controlnet: {controlnet}") 

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    max_batch_size = 16
    # FIXME VAE build fails due to element limit. Limitting batch size is WAR
    if args.build_dynamic_shape or image_height > 512 or image_width > 512:
        max_batch_size = 4

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

    # Initialize demo
    demo = Txt2ImgPipeline(
        scheduler=args.scheduler,
        denoising_steps=args.denoising_steps,
        output_dir=args.output_dir,
        version=args.version,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=args.use_cuda_graph,
        controlnets=args.controlnet_type)

    # Load TensorRT engines and pytorch modules
    os.makedirs(args.engine_dir, exist_ok=True)
    os.makedirs(args.onnx_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    demo.loadEngines(args.engine_dir, args.onnx_dir, args.onnx_opset,
        opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, enable_all_tactics=args.build_all_tactics, \
        timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)
    demo.loadResources(image_height, image_width, batch_size, args.seed)

    if args.use_cuda_graph:
        # inference once to get cuda graph
        images = demo.infer(prompt, negative_prompt, image_height, image_width, controlnet_imgs=controlnet_imgs, controlnet_scales=args.controlnet_scale, warmup=True, verbose=False)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images = demo.infer(prompt, negative_prompt, image_height, image_width, controlnet_imgs=controlnet_imgs, controlnet_scales=args.controlnet_scale, warmup=True, verbose=False)

    print("[I] Running StableDiffusion [Controlnet] pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images = demo.infer(prompt, negative_prompt, image_height, image_width, controlnet_imgs=controlnet_imgs, controlnet_scales=args.controlnet_scale, seed=args.seed, verbose=args.verbose)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    demo.teardown()
