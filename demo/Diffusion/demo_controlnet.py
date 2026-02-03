#
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

# Configure dependencies before any external imports
from demo_diffusion import deps
deps.configure("sd")

import argparse

import controlnet_aux
import torch
from cuda.bindings import runtime as cudart
from PIL import Image

from demo_diffusion import dd_argparse
from demo_diffusion import image as image_module
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion ControlNet Demo", conflict_handler='resolve')
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument('--scheduler', type=str, default="UniPC", choices=["DDIM", "DPM", "EulerA", "LMSD", "PNDM", "UniPC"], help="Scheduler for diffusion process")
    parser.add_argument('--input-image', nargs = '+', type=str, default=[], help="Path to the input image/images already prepared for ControlNet modality. For example: canny edged image for canny ControlNet, not just regular rgb image")
    parser.add_argument('--controlnet-type', nargs='+', type=str, default=["canny"], help="Controlnet type, can be `None`, `str` or `str` list from ['canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble', 'seg']")
    parser.add_argument('--controlnet-scale', nargs='+', type=float, default=[1.0], help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet, can be `None`, `float` or `float` list")
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion controlnet demo using TensorRT")
    args = parseArgs()

    # Controlnet configuration
    if not isinstance(args.controlnet_type, list):
        raise ValueError(f"`--controlnet-type` must be of type `str` or `str` list, but is {type(args.controlnet_type)}")

    # Controlnet configuration
    if not isinstance(args.controlnet_scale, list):
        raise ValueError(f"`--controlnet-scale`` must be of type `float` or `float` list, but is {type(args.controlnet_scale)}")

    # Check number of ControlNets to ControlNet scales
    if len(args.controlnet_type) != len(args.controlnet_scale):
        raise ValueError(f"Numbers of ControlNets {len(args.controlnet_type)} should be equal to number of ControlNet scales {len(args.controlnet_scale)}.")

    # Convert controlnet scales to tensor
    controlnet_scale = torch.FloatTensor(args.controlnet_scale)

    # Check images
    input_images = []
    if len(args.input_image) > 0:
        for image in args.input_image:
            input_images.append(Image.open(image))
    else:
        for controlnet in args.controlnet_type:
            if controlnet == "canny":
                if args.version == "xl-1.0":
                    canny_image = image_module.download_image(
                        "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/out_bird.png"
                    )
                    # "out_bird.png" has 5 images combined in a row. We pick the first image which is the input image.
                    canny_image = canny_image.crop((0, 0, canny_image.width / 5, canny_image.height))
                elif args.version == "1.5":
                    canny_image = image_module.download_image(
                        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
                    )
                    canny_image = controlnet_aux.CannyDetector()(canny_image)
                else:
                    raise ValueError(
                        f"This demo supports ControlNets for v1.4 and SDXL base pipelines only. Version provided: {args.version}"
                    )
                input_images.append(canny_image.resize((args.width, args.height)))
            elif controlnet == "normal":
                normal_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png"
                )
                normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(normal_image)
                input_images.append(normal_image.resize((args.width, args.height)))
            elif controlnet == "depth":
                depth_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
                )
                depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
                input_images.append(depth_image.resize((args.width, args.height)))
            elif controlnet == "hed":
                hed_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png"
                )
                hed_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(hed_image)
                input_images.append(hed_image.resize((args.width, args.height)))
            elif controlnet == "mlsd":
                mlsd_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png"
                )
                mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(mlsd_image)
                input_images.append(mlsd_image.resize((args.width, args.height)))
            elif controlnet == "openpose":
                openpose_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"
                )
                openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(openpose_image)
                input_images.append(openpose_image.resize((args.width, args.height)))
            elif controlnet == "scribble":
                scribble_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png"
                )
                scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(scribble_image, scribble=True)
                input_images.append(scribble_image.resize((args.width, args.height)))
            elif controlnet == "seg":
                seg_image = image_module.download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png"
                )
                seg_image = controlnet_aux.SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")(seg_image)
                input_images.append(seg_image.resize((args.width, args.height)))
            else:
                raise ValueError(f"You should implement the conditonal image of this controlnet: {controlnet}")
    assert len(input_images) > 0

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = dd_argparse.process_pipeline_args(args)

    # Initialize demo
    demo = pipeline_module.StableDiffusionPipeline(
        pipeline_type=(
            pipeline_module.PIPELINE_TYPE.CONTROLNET
            if args.version != "xl-1.0"
            else pipeline_module.PIPELINE_TYPE.XL_CONTROLNET
        ),
        controlnets=args.controlnet_type,
        **kwargs_init_pipeline,
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
    demo_kwargs = {'input_image': input_images, 'controlnet_scales': controlnet_scale}
    demo.run(*args_run_demo, **demo_kwargs)

    demo.teardown()
