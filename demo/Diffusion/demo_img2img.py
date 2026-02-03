#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configure dependencies before any external imports
from demo_diffusion import deps
deps.configure("sd")

import argparse

import PIL
from cuda.bindings import runtime as cudart
from PIL import Image

from demo_diffusion import dd_argparse
from demo_diffusion import image as image_module
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Img2Img Demo")
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion img2img demo using TensorRT")
    args = parseArgs()

    if args.input_image:
        input_image = Image.open(args.input_image)
    else:
        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        input_image = image_module.download_image(url)

    image_width, image_height = input_image.size
    if image_height != args.height or image_width != args.width:
        print(f"[I] Resizing input_image to {args.height}x{args.width}")
        input_image = input_image.resize((args.width, args.height))
        image_height, image_width = args.height, args.width

    if isinstance(input_image, PIL.Image.Image):
        input_image = image_module.preprocess_image(input_image)

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = dd_argparse.process_pipeline_args(args)

    # Initialize demo
    demo = pipeline_module.StableDiffusionPipeline(
        pipeline_type=pipeline_module.PIPELINE_TYPE.IMG2IMG, **kwargs_init_pipeline
    )

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
    demo_kwargs = {'input_image': input_image, 'image_strength': 0.75}
    demo.run(*args_run_demo, **demo_kwargs)

    demo.teardown()
