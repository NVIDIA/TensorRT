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

from cuda import cudart

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    # Stable Diffusion 3.5 configuration
    parser = argparse.ArgumentParser(
        description="Options for Stable Diffusion 3.5 Txt2Img Demo", conflict_handler="resolve"
    )
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument(
        "--version",
        type=str,
        default="3.5-medium",
        choices={"3.5-medium", "3.5-large"},
        help="Version of Stable Diffusion 3.5",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Value of classifier-free guidance scale (must be greater than 1)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Maximum sequence length to use with the prompt.",
    )
    parser.add_argument("--denoising-steps", type=int, default=50, help="Number of denoising steps")

    return parser.parse_args()

def process_demo_args(args):
    batch_size = args.batch_size
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    # Process prompt
    if not isinstance(prompt, list):
        raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
    prompt = prompt * batch_size

    if not isinstance(negative_prompt, list):
        raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
    if len(negative_prompt) == 1:
        negative_prompt = negative_prompt * batch_size

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 8 but specified as: {args.image_height} and {args.width}."
        )

    max_batch_size = 4
    if args.batch_size > max_batch_size:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed {max_batch_size}.")

    if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
        raise ValueError(
            "Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`"
        )

    kwargs_run_demo = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "batch_count": args.batch_count,
        "num_warmup_runs": args.num_warmup_runs,
        "use_cuda_graph": args.use_cuda_graph,
    }

    return kwargs_run_demo


if __name__ == "__main__":
    print("[I] Initializing Stable Diffusion 3.5 demo using TensorRT")
    args = parseArgs()

    _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(args)
    kwargs_run_demo = process_demo_args(args)

    # Initialize demo
    demo = pipeline_module.StableDiffusion35Pipeline.FromArgs(args, pipeline_type=pipeline_module.PIPELINE_TYPE.TXT2IMG)

    # Load TensorRT engines and pytorch modules
    demo.load_engines(
        framework_model_dir=args.framework_model_dir,
        **kwargs_load_engine,
    )

    if demo.low_vram:
        demo.device_memory_sizes = demo.get_device_memory_sizes()
    else:
        _, shared_device_memory = cudart.cudaMalloc(demo.calculate_max_device_memory())
        demo.activate_engines(shared_device_memory)

    # Load resources
    demo.load_resources(
        image_height=args.height,
        image_width=args.width,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Run inference
    demo.run(**kwargs_run_demo)

    demo.teardown()
