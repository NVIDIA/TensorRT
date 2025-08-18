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

import argparse
import os

import controlnet_aux
from cuda.bindings import runtime as cudart
from PIL import Image

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parse_args():
    parser = argparse.ArgumentParser(description="Options for Flux Img2Img Demo", conflict_handler="resolve")
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument(
        "--version",
        type=str,
        default="flux.1-dev",
        choices=("flux.1-dev", "flux.1-schnell", "flux.1-dev-canny", "flux.1-dev-depth"),
        help="Version of Flux",
    )
    parser.add_argument(
        "--prompt2",
        default=None,
        nargs="*",
        help="Text prompt(s) to be sent to the T5 tokenizer and text encoder. If not defined, prompt will be used instead",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of image to generate (must be multiple of 8)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of image to generate (must be multiple of 8)",
    )
    parser.add_argument("--denoising-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Value of classifier-free guidance scale (must be greater than 1)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        help="Maximum sequence length to use with the prompt. Can be up to 512 for the dev and 256 for the schnell variant.",
    )
    parser.add_argument(
        "--t5-ws-percentage",
        type=int,
        default=None,
        help="Set runtime weight streaming budget as the percentage of the size of streamable weights for the T5 model. This argument only takes effect when --ws is set. 0 streams the most weights and 100 or None streams no weights. ",
    )

    parser.add_argument(
        "--transformer-ws-percentage",
        type=int,
        default=None,
        help="Set runtime weight streaming budget as the percentage of the size of streamable weights for the transformer model. This argument only takes effect when --ws is set. 0 streams the most weights and 100 or None streams no weights.",
    )
    parser.add_argument("--control-image", type=str, default=None, help="Path to the control image")
    parser.add_argument("--input-image", type=str, default=None, help="Path to the input conditioning image")
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Indicates extent to transform the reference `image`. Must be between 0 and 1. A value of 1 essentially ignores the input image.",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Path to the calibration dataset for quantization (only enabled for controlnet)",
    )

    return parser.parse_args()


def process_demo_args(args):
    batch_size = args.batch_size
    prompt = args.prompt
    # If prompt2 is not defined, use prompt instead
    prompt2 = args.prompt2 or prompt

    # Process input args
    if not isinstance(prompt, list):
        raise ValueError(f"`prompt` must be of type `list[str]`, but is {type(prompt)}")
    prompt = prompt * batch_size

    if not isinstance(prompt2, list):
        raise ValueError(f"`prompt2` must be of type `str` list, but is {type(prompt2)}")
    if len(prompt2) == 1:
        prompt2 = prompt2 * batch_size

    max_seq_supported_by_model = {
        "flux.1-schnell": 256,
        "flux.1-dev": 512,
        "flux.1-dev-canny": 512,
        "flux.1-dev-depth": 512,
    }[args.version]
    if args.max_sequence_length is not None:
        if args.max_sequence_length > max_seq_supported_by_model:
            raise ValueError(
                f"For {args.version}, `max_sequence_length` cannot be greater than {max_seq_supported_by_model} but is {args.max_sequence_length}"
            )
    else:
        args.max_sequence_length = max_seq_supported_by_model

    controlnet_type = "depth" if "depth" in args.version else "canny" if "canny" in args.version else ""
    if controlnet_type:
        if args.input_image:
            raise ValueError(
                f"--input-image is a valid input for versions [flux.1-dev, flux.1-schnell]. Provided {args.version}"
            )
        if not args.control_image:
            raise ValueError(
                "--control-image input is required for versions [flux.1-dev-canny, flux.1-dev-depth]. Please provide it using --control-image flag."
            )
        args.control_image = Image.open(args.control_image).convert("RGB")

        if controlnet_type == "canny":
            processor = controlnet_aux.CannyDetector()
            args.control_image = processor(
                args.control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
            )
        elif controlnet_type == "depth":
            args.control_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(
                args.control_image
            )
        else:
            raise ValueError("Invalid controlnet type")
    else:
        if args.control_image:
            raise ValueError(
                f"--control-image is a valid input for versions [flux.1-dev-canny, flux.1-dev-depth]. Provided {args.version}"
            )
        if not args.input_image:
            raise ValueError(
                "--input-image is required for the img2img pipeline. Please provide it using the --input-image flag."
            )
        args.input_image = Image.open(args.input_image).convert("RGB").resize((args.height, args.width))

    if args.fp8:
        if not controlnet_type:
            raise ValueError("--fp8 is currently not supported for Flux img2img pipelines.")

        if not args.calibration_dataset:
            args.calibration_dataset = os.path.join(f"{controlnet_type}-eval", "benchmark")
            print(f"[W] Calibration dataset path not provided, setting default path to {args.calibration_dataset}.")

        if not os.path.exists(args.calibration_dataset):
            print(
                f"[W] Could not find the calibration dataset at {args.calibration_dataset}, and will fallback to using pre-exported ONNX models. Please follow the instructions in README to download calibration dataset and provide the path if pre-exported ONNX models are not provided either."
            )

    if args.fp4 and not controlnet_type:
        raise ValueError("--fp4 is currently not supported for Flux img2img pipelines.")

    kwargs_run_demo = {
        "prompt": prompt,
        "prompt2": prompt2,
        "height": args.height,
        "width": args.width,
        "batch_count": args.batch_count,
        "num_warmup_runs": args.num_warmup_runs,
        "use_cuda_graph": args.use_cuda_graph,
        "control_image": args.control_image,
        "input_image": args.input_image,
        "image_strength": args.image_strength,
    }

    return kwargs_run_demo


if __name__ == "__main__":
    print("[I] Initializing Flux img2img demo using TensorRT")
    args = parse_args()

    _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(args)
    kwargs_run_demo = process_demo_args(args)

    # Initialize demo
    demo = pipeline_module.FluxPipeline.FromArgs(args, pipeline_type=pipeline_module.PIPELINE_TYPE.IMG2IMG)

    # Load TensorRT engines and pytorch modules
    demo.load_engines(
        framework_model_dir=args.framework_model_dir,
        **kwargs_load_engine,
    )

    if args.onnx_export_only:
        print("[I] ONNX export completed. Exiting...")
        demo.teardown()
        exit(0)

    # Since VAE and VAE_encoder require by far the largest device memories, in low-vram mode
    # we allocate the required device memory individually before each model is run.
    if demo.low_vram:
        demo.device_memory_sizes = demo.get_device_memory_sizes()
    else:
        _, shared_device_memory = cudart.cudaMalloc(demo.calculate_max_device_memory())
        demo.activate_engines(shared_device_memory)

    demo.load_resources(args.height, args.width, args.batch_size, args.seed)

    # Run inference
    demo.run(**kwargs_run_demo)

    demo.teardown()
