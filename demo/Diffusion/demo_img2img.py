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

from img2img_pipeline import Img2ImgPipeline
from utilities import preprocess_image, TRT_LOGGER, add_arguments, download_image

import PIL
from PIL import Image

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Img2Img Demo")
    parser = add_arguments(parser)
    parser.add_argument('--scheduler', type=str, default="DDIM", choices=["DDIM", "EulerA", "LMSD", "DPM", "PNDM"], help="Scheduler for diffusion process")
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion img2img demo using TensorRT")
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

    if args.input_image:
        input_image = Image.open(args.input_image)
    else:
        url = "https://pajoca.com/wp-content/uploads/2022/09/tekito-yamakawa-1.png"
        input_image = download_image(url)

    image_width, image_height = input_image.size

    # Validate image dimensions
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

    if isinstance(input_image, PIL.Image.Image):
        input_image = preprocess_image(input_image)

    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    max_batch_size = 16
    if args.build_dynamic_shape:
        max_batch_size = 4

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

    # Initialize demo
    demo = Img2ImgPipeline(
        scheduler=args.scheduler,
        denoising_steps=args.denoising_steps,
        output_dir=args.output_dir,
        version=args.version,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size)

    # Load TensorRT engines and pytorch modules
    demo.loadEngines(args.engine_dir, args.framework_model_dir, args.onnx_dir, args.onnx_opset,
        opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, enable_all_tactics=args.build_all_tactics, \
        timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)

    max_device_memory = max(demo.calculateMaxDeviceMemory(), demo.calculateMaxDeviceMemory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    demo.activateEngines(shared_device_memory)

    demo.loadResources(image_height, image_width, batch_size, args.seed)

    if args.use_cuda_graph:
        # inference once to get cuda graph
        images = demo.infer(prompt, negative_prompt, input_image, image_height, image_width, strength=0.75, warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images = demo.infer(prompt, negative_prompt, input_image, image_height, image_width, strength=0.75, warmup=True)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()

    images = demo.infer(prompt, negative_prompt, input_image, image_height, image_width, seed=args.seed, strength=0.75)

    if args.nvtx_profile:
        cudart.cudaProfilerStop()
