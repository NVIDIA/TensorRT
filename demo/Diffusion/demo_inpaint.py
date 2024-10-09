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

from cuda import cudart
from PIL import Image

from stable_diffusion_pipeline import StableDiffusionPipeline
from utilities import PIPELINE_TYPE, TRT_LOGGER, add_arguments, download_image, process_pipeline_args

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Inpaint Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="1.5", choices=["1.5", "2.0"], help="Stable Diffusion version. Only 1.5 and 2.0 supported for inpainting.")
    parser.add_argument('--scheduler', type=str, default="PNDM", choices=["PNDM"], help="Scheduler for diffusion process")
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")
    parser.add_argument('--mask-image', type=str, default="", help="Path to the mask image")
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion inpainting demo using TensorRT")
    args = parseArgs()

    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB")
    else:
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        input_image = download_image(img_url)

    if args.mask_image:
        mask_image = Image.open(args.mask_image).convert("RGB")
    else:
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        mask_image = download_image(mask_url)

    image_width, image_height = input_image.size
    if image_height != args.height or image_width != args.width:
        print(f"[I] Resizing input_image to {args.height}x{args.width}")
        input_image = input_image.resize((args.height, args.width))
        image_height, image_width = args.height, args.width

    mask_width, mask_height = mask_image.size
    if mask_height != args.height or mask_width != args.width:
        print(f"[I] Resizing mask_image to {args.height}x{args.width}")
        mask_image = mask_image.resize((args.height, args.width))
        mask_height, mask_width = args.height, args.width

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    # Initialize demo
    demo = StableDiffusionPipeline(
        pipeline_type=PIPELINE_TYPE.INPAINT,
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
    demo_kwargs = {'input_image': input_image, 'image_strength': 0.75, 'mask_image': mask_image}
    demo.run(*args_run_demo, **demo_kwargs)

    demo.teardown()
