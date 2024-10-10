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

from flux_pipeline import FluxPipeline
from utilities import PIPELINE_TYPE, add_arguments, process_pipeline_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Options for Flux Txt2Img Demo", conflict_handler="resolve"
    )
    parser = add_arguments(parser)
    parser.add_argument(
        "--version",
        type=str,
        default="flux.1-dev",
        choices=["flux.1-dev"],
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
    parser.add_argument(
        "--denoising-steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Value of classifier-free guidance scale (must be greater than 1)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the prompt",
    )
    return parser.parse_args()


def process_demo_args(args):
    batch_size = args.batch_size
    prompt = args.prompt
    # If prompt2 is not defined, use prompt instead
    prompt2 = args.prompt2 or prompt

    # Process input args
    if not isinstance(prompt, list):
        raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
    prompt = prompt * batch_size

    if not isinstance(prompt2, list):
        raise ValueError(
            f"`prompt2` must be of type `str` list, but is {type(prompt2)}"
        )
    if len(prompt2) == 1:
        prompt2 = prompt2 * batch_size

    if args.max_sequence_length is not None and args.max_sequence_length > 512:
        raise ValueError(
            f"`max_sequence_length` cannot be greater than 512 but is {args.max_sequence_length}"
        )

    args_run_demo = (
        prompt,
        prompt2,
        args.height,
        args.width,
        args.batch_count,
        args.num_warmup_runs,
        args.use_cuda_graph,
    )

    return args_run_demo


if __name__ == "__main__":
    print("[I] Initializing Flux txt2img demo using TensorRT")
    args = parse_args()

    kwargs_init_pipeline, kwargs_load_engine, _ = process_pipeline_args(args)
    args_run_demo = process_demo_args(args)

    # Initialize demo
    demo = FluxPipeline(
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        max_sequence_length=args.max_sequence_length,
        **kwargs_init_pipeline,
    )

    # Load TensorRT engines and pytorch modules
    demo.load_engines(
        args.engine_dir, args.framework_model_dir, args.onnx_dir, **kwargs_load_engine
    )

    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(demo.calculate_max_device_memory())
    demo.activate_engines(shared_device_memory)
    demo.load_resources(args.height, args.width, args.batch_size, args.seed)

    # Run inference
    demo.run(*args_run_demo)

    demo.teardown()
