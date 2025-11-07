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
from diffusers.utils import load_image, load_video

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parse_args():
    parser = argparse.ArgumentParser(description="Options for Cosmos video2world Demo", conflict_handler="resolve")
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument(
        "--version",
        type=str,
        default="cosmos-predict2-2b-video2world",
        choices=("cosmos-predict2-2b-video2world", "cosmos-predict2-14b-video2world"),
        help="Version of Cosmos",
    )
    parser.add_argument('--input-image', type=str, default=None, help="Path to the input image")
    parser.add_argument('--input-video', type=str, default=None, help="Path to the input video")
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Height of image to generate (must be multiple of 8)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of image to generate (must be multiple of 8)",
    )
    parser.add_argument("--denoising-steps", type=int, default=35, help="Number of denoising steps")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Value of classifier-free guidance scale (must be greater than 1)",
    )
    parser.add_argument("--num-frames", type=int, default=93, help="The number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second of the generated video.")
    parser.add_argument("--num-videos-per-prompt", type=int, default=1, help="The number of videos to generate per prompt.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the prompt.",
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
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision by default.",
    )
    return parser.parse_args()


def process_demo_args(args):
    batch_size = args.batch_size
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    # Process input args
    if not isinstance(prompt, list):
        raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
    prompt = prompt * batch_size
    if not isinstance(negative_prompt, list):
        raise ValueError(f"`negative_prompt` must be of type `str` list, but is {type(negative_prompt)}")
    negative_prompt = negative_prompt * batch_size

    # process input image and input video
    if args.input_image and args.input_video:
        raise ValueError("Only one of --input-image or --input-video can be provided")
    if args.input_image:
        args.input_image = load_image(args.input_image)
    elif args.input_video:
        args.input_video = load_video(args.input_video)
    else:
        # load default image
        args.input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png")

    kwargs_run_demo = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "batch_count": args.batch_count,
        "num_warmup_runs": args.num_warmup_runs,
        "use_cuda_graph": args.use_cuda_graph,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "input_image": args.input_image,
        "input_video": args.input_video,
        "num_videos_per_prompt": args.num_videos_per_prompt,
    }

    return kwargs_run_demo


if __name__ == "__main__":
    print("[I] Initializing Cosmos video2world demo using TensorRT")
    args = parse_args()

    # Enforce torch-inference is enabled
    if not args.torch_inference:
        print("[W] The video2world demo only supports the PyTorch backend. Enabling torch-inference with 'eager' mode.")
        args.torch_inference = "eager"

    _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(args)
    kwargs_run_demo = process_demo_args(args)

    # Initialize demo
    demo = pipeline_module.CosmosPipeline.FromArgs(args, pipeline_type=pipeline_module.PIPELINE_TYPE.VIDEO2WORLD)

    # Load TensorRT engines and pytorch modules
    demo.load_engines(
        framework_model_dir=args.framework_model_dir,
        **kwargs_load_engine,
    )

    if args.onnx_export_only:
        print("[I] ONNX export completed. Exiting...")
        demo.teardown()
        exit(0)

    # In low-vram mode we allocate the required device memory individually before each model is run.
    if demo.low_vram:
        demo.device_memory_sizes = demo.get_device_memory_sizes()
    else:
        _, shared_device_memory = cudart.cudaMalloc(demo.calculate_max_device_memory())
        demo.activate_engines(shared_device_memory)

    demo.load_resources(args.height, args.width, args.batch_size, args.seed)

    # Run inference
    videos = demo.run(**kwargs_run_demo)

    demo.teardown()

    # save video
    demo.save_video(kwargs_run_demo["prompt"], videos, check_integrity=True)
