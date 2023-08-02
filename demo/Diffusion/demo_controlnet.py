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

import argparse
from cuda import cudart
import tensorrt as trt
import nvtx
import torch
from controlnet_pipeline import ControlNetPipeline
from utilities import TRT_LOGGER, add_arguments, download_image
import controlnet_aux
from PIL import Image

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Img2Img Demo")
    parser = add_arguments(parser)
    parser.add_argument('--scheduler', type=str, default="UniPCMultistepScheduler", choices=["DDIM", "EulerA", "LMSD", "DPM", "PNDM", "UniPCMultistepScheduler"], help="Scheduler for diffusion process")
    parser.add_argument('--input-image', nargs = '+', type=str, default=[], help="Path to the input image/images already prepared for ControlNet modality. For example: canny edged image for canny ControlNet, not just regular rgb image")
    parser.add_argument('--controlnet-type', nargs='+', type=str, default=["canny"], help="Controlnet type, can be `None`, `str` or `str` list from ['canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble', 'seg']")
    parser.add_argument('--controlnet-scale', nargs='+', type=float, default=[1.0], help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet, can be `None`, `float` or `float` list")
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion controlnet demo using TensorRT")
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
                canny_image = download_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
                canny_image = controlnet_aux.CannyDetector()(canny_image)
                input_images.append(canny_image.resize((args.height, args.width)))
            elif controlnet == "normal":
                normal_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png")
                normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(normal_image)
                input_images.append(normal_image.resize((args.height, args.width)))
            elif controlnet == "depth":
                depth_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
                depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
                input_images.append(depth_image.resize((args.height, args.width)))
            elif controlnet == "hed":
                hed_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png")
                hed_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(hed_image)
                input_images.append(hed_image.resize((args.height, args.width)))
            elif controlnet == "mlsd":
                mlsd_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png")
                mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(mlsd_image)
                input_images.append(mlsd_image.resize((args.height, args.width)))
            elif controlnet == "openpose":
                openpose_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png")
                openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(openpose_image)
                input_images.append(openpose_image.resize((args.height, args.width)))
            elif controlnet == "scribble":
                scribble_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png")
                scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(scribble_image, scribble=True)
                input_images.append(scribble_image.resize((args.height, args.width)))
            elif controlnet == "seg":
                seg_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png")
                seg_image = controlnet_aux.SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")(seg_image)
                input_images.append(seg_image.resize((args.height, args.width)))
            else:
                raise ValueError(f"You should implement the conditonal image of this controlnet: {controlnet}")
    
    assert len(input_images) > 0

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")
    
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
    demo = ControlNetPipeline(
        scheduler=args.scheduler,
        denoising_steps=args.denoising_steps,
        output_dir=args.output_dir,
        version=args.version,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        use_cuda_graph=args.use_cuda_graph,
        controlnet=args.controlnet_type,
        max_batch_size=max_batch_size)

    # Load TensorRT engines and pytorch modules
    demo.loadEngines(args.engine_dir, args.framework_model_dir, args.onnx_dir, args.onnx_opset,
        opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, \
        enable_all_tactics=args.build_all_tactics, \
        timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)

    max_device_memory = max(demo.calculateMaxDeviceMemory(), demo.calculateMaxDeviceMemory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    demo.activateEngines(shared_device_memory)

    demo.loadResources(image_height, image_width, batch_size, args.seed)

    if args.use_cuda_graph:
        # inference once to get cuda graph
        images = demo.infer(prompt, negative_prompt, input_images, image_height, image_width, controlnet_scale, warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images = demo.infer(prompt, negative_prompt, input_images, image_height, image_width, controlnet_scale, warmup=True)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()

    images = demo.infer(prompt, negative_prompt, input_images, image_height, image_width, controlnet_scale, seed=args.seed)

    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    demo.teardown()
