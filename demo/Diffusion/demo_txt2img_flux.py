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
deps.configure("flux")

import argparse

from cuda.bindings import runtime as cudart

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Options for Flux Txt2Img Demo", conflict_handler="resolve"
    )
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument(
        "--version",
        type=str,
        default="flux.1-dev",
        choices=("flux.1-dev", "flux.1-schnell"),
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
        help="Set runtime weight streaming budget as the percentage of the size of streamable weights for the transformer model. This argument only takes effect when --ws is set. 0 streams the most weights and 100 or None streams no weights."
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

    max_seq_supported_by_model = {
        "flux.1-schnell": 256,
        "flux.1-dev": 512,
    }[args.version]
    if args.max_sequence_length is not None:
        if args.max_sequence_length > max_seq_supported_by_model:
            raise ValueError(
                f"For {args.version}, `max_sequence_length` cannot be greater than {max_seq_supported_by_model} but is {args.max_sequence_length}"
            )
    else:
        args.max_sequence_length = max_seq_supported_by_model

    kwargs_run_demo = {
        "prompt": prompt,
        "prompt2": prompt2,
        "height": args.height,
        "width": args.width,
        "batch_count": args.batch_count,
        "num_warmup_runs": args.num_warmup_runs,
        "use_cuda_graph": args.use_cuda_graph,
    }

    return kwargs_run_demo


if __name__ == "__main__":
    print("[I] Initializing Flux txt2img demo using TensorRT")
    args = parse_args()

    _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(args)
    kwargs_run_demo = process_demo_args(args)

    # Initialize demo
    pipeline_type = pipeline_module.PIPELINE_TYPE.TXT2IMG
    demo = pipeline_module.FluxPipeline.FromArgs(args, pipeline_type=pipeline_type)

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
    images = demo.run(**kwargs_run_demo)

    demo.teardown()

    # save images
    demo.save_images(kwargs_run_demo["prompt"], images)
