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
import os
from cuda import cudart
import tensorrt as trt
import torch
from utilities import TRT_LOGGER, add_arguments
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0"], help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")

    parser.add_argument('--scheduler', type=str, default="DDIM", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")

    parser.add_argument('--onnx-base-dir', default='onnx_xl_base', help="Directory for SDXL-Base ONNX models")
    parser.add_argument('--onnx-refiner-dir', default='onnx_xl_refiner', help="Directory for SDXL-Refiner ONNX models")
    parser.add_argument('--engine-base-dir', default='engine_xl_base', help="Directory for SDXL-Base TensorRT engines")
    parser.add_argument('--engine-refiner-dir', default='engine_xl_refiner', help="Directory for SDXL-Refiner TensorRT engines")

    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing TensorRT accelerated StableDiffusionXL txt2img pipeline")
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

    def init_pipeline(pipeline_class, refinfer, onnx_dir, engine_dir):
        # Initialize demo
        demo = pipeline_class(
            scheduler=args.scheduler,
            denoising_steps=args.denoising_steps,
            output_dir=args.output_dir,
            version=args.version,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            max_batch_size=max_batch_size,
            use_cuda_graph=args.use_cuda_graph,
            refiner=refinfer,
            framework_model_dir=args.framework_model_dir)

        # Load TensorRT engines and pytorch modules
        demo.loadEngines(engine_dir, args.framework_model_dir, onnx_dir, args.onnx_opset,
            opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
            force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
            force_build=args.force_engine_build,
            static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
            enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, \
            enable_all_tactics=args.build_all_tactics, \
            timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)
        return demo

    demo_base = init_pipeline(Txt2ImgXLPipeline, False, args.onnx_base_dir, args.engine_base_dir)
    demo_refiner = init_pipeline(Img2ImgXLPipeline, True, args.onnx_refiner_dir, args.engine_refiner_dir)
    max_device_memory = max(demo_base.calculateMaxDeviceMemory(), demo_refiner.calculateMaxDeviceMemory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    demo_base.activateEngines(shared_device_memory)
    demo_refiner.activateEngines(shared_device_memory)
    demo_base.loadResources(image_height, image_width, batch_size, args.seed)
    demo_refiner.loadResources(image_height, image_width, batch_size, args.seed)

    def run_sd_xl_inference(warmup=False, verbose=False):
        images, time_base = demo_base.infer(prompt, negative_prompt, image_height, image_width, warmup=warmup, verbose=verbose, seed=args.seed, return_type="latents")
        images, time_refiner = demo_refiner.infer(prompt, negative_prompt, images, image_height, image_width, warmup=warmup, verbose=verbose, seed=args.seed)
        return images, time_base + time_refiner

    if args.use_cuda_graph:
        # inference once to get cuda graph
        images, _ = run_sd_xl_inference(warmup=True, verbose=False)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images, _ = run_sd_xl_inference(warmup=True, verbose=False)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, pipeline_time = run_sd_xl_inference(warmup=False, verbose=args.verbose)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    print('|------------|--------------|')
    print('| {:^10} | {:>9.2f} ms |'.format('e2e', pipeline_time))
    print('|------------|--------------|')

    demo_base.teardown()
    demo_refiner.teardown()
