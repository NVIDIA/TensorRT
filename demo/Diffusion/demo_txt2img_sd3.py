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

import argparse
from PIL import Image
from cuda import cudart

from stable_diffusion_3_pipeline import StableDiffusion3Pipeline
from utilities import PIPELINE_TYPE, add_arguments
from utils_sd3.other_impls import preprocess_image_sd3

def parseArgs():
    # Stable Diffusion 3 configuration
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion 3 Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="sd3", choices=["sd3"], help="Version of Stable Diffusion")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--shift', type=int, default=1.0, help="Shift parameter for SD3")
    parser.add_argument('--cfg-scale', type=int, default=5, help="CFG Scale for SD3")
    parser.add_argument('--denoising-steps', type=int, default=50, help="Number of denoising steps")
    parser.add_argument('--denoising-percentage', type=float, default=0.6, help="Percentage of denoising steps to run. This parameter is only used if input-image is provided")
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")

    return parser.parse_args()

def process_pipeline_args(args):
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {args.image_height} and {args.width}.")

    max_batch_size = 4
    if args.batch_size > max_batch_size:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed {max_batch_size}.")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

    input_image = None
    if args.input_image:
        input_image = Image.open(args.input_image)

        image_width, image_height = input_image.size
        if image_height != args.height or image_width != args.width:
            print(f"[I] Resizing input_image to {args.height}x{args.width}")
            input_image = input_image.resize((args.height, args.width), Image.LANCZOS)
            image_height, image_width = args.height, args.width

        input_image = preprocess_image_sd3(input_image)

    kwargs_init_pipeline = {
        'version': args.version,
        'max_batch_size': max_batch_size,
        'output_dir': args.output_dir,
        'hf_token': args.hf_token,
        'verbose': args.verbose,
        'nvtx_profile': args.nvtx_profile,
        'use_cuda_graph': args.use_cuda_graph,
        'framework_model_dir': args.framework_model_dir,
        'torch_inference': args.torch_inference,
        'shift': args.shift,
        'cfg_scale': args.cfg_scale,
        'denoising_steps': args.denoising_steps,
        'denoising_percentage': args.denoising_percentage,
        'input_image': input_image
    }

    kwargs_load_engine = {
        'onnx_opset': args.onnx_opset,
        'opt_batch_size': args.batch_size,
        'opt_image_height': args.height,
        'opt_image_width': args.width,
        'static_batch': args.build_static_batch,
        'static_shape': not args.build_dynamic_shape,
        'enable_all_tactics': args.build_all_tactics,
        'timing_cache': args.timing_cache,
    }

    args_run_demo = (args.prompt, args.negative_prompt, args.height, args.width, args.batch_size, args.batch_count, args.num_warmup_runs, args.use_cuda_graph)

    return kwargs_init_pipeline, kwargs_load_engine, args_run_demo

if __name__ == "__main__":
    print("[I] Initializing Stable Diffusion 3 demo using TensorRT")
    args = parseArgs()

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    # Initialize demo
    demo = StableDiffusion3Pipeline(
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        **kwargs_init_pipeline)

    # Load TensorRT engines and pytorch modules
    demo.loadEngines(
        args.engine_dir,
        args.framework_model_dir,
        args.onnx_dir,
        **kwargs_load_engine)

    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(demo.calculateMaxDeviceMemory())
    demo.activateEngines(shared_device_memory)
    demo.loadResources(args.height, args.width, args.batch_size, args.seed)

    # Run inference
    demo.run(*args_run_demo)

    demo.teardown()
