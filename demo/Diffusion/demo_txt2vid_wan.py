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
deps.configure("cosmos")

import argparse

from cuda.bindings import runtime as cudart

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Wan 2.2 Txt2Vid Demo", conflict_handler='resolve')
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument('--version', type=str, default="wan2.2-t2v-a14b", help="Version of Wan")
    parser.add_argument('--guidance-scale', type=float, default=4.0, help="Guidance scale for high-noise stage (Wan default: 4.0)")
    parser.add_argument('--guidance-scale-2', type=float, default=3.0, help="Guidance scale for low-noise stage (Wan default: 3.0)")
    parser.add_argument('--boundary-ratio', type=float, default=0.875, help="Boundary ratio for two-stage denoising (default: 0.875)")
    parser.add_argument('--denoising-steps', type=int, default=40, help="Number of denoising steps (Wan default: 40)")
    parser.add_argument('--num-warmup-runs', type=int, default=1, help="Number of warmup runs before benchmarking")
    parser.add_argument(
        '--negative-prompt',
        nargs='*',
        default= (
            "vivid colors, overexposed, static, blurry details, subtitles, style, "
            "work of art, painting, picture, still, overall grayish, worst quality, "
            "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
            "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
            "static image, cluttered background, three legs, many people in the background, "
            "walking backwards"
        ),
        help="Negative prompt (Wan team default, English translation)"
    )
    return parser.parse_args()


def process_demo_args(args):
    args.height = 720
    args.width = 1280
    args.num_frames = 81
    args.max_sequence_length = 512
    
    # require static batch = 1
    if args.batch_size > 1:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed (max=1 for Wan).")
    args.batch_size = 1
    args.build_static_batch = True
    args.build_dynamic_shape = False
    
    print(f"[I] Building Wan 2.2 T2V with fixed resolution: {args.height}×{args.width}, {args.num_frames} frames")
    
    negative_prompt = args.negative_prompt
    if isinstance(negative_prompt, list):
        negative_prompt = ' '.join(negative_prompt) if negative_prompt else ""
    
    kwargs_run_demo = {
        'prompt': args.prompt,
        'height': args.height,
        'width': args.width,
        'num_frames': args.num_frames,
        'batch_size': args.batch_size,
        'batch_count': args.batch_count,
        'num_warmup_runs': args.num_warmup_runs,
        'use_cuda_graph': args.use_cuda_graph,
        'negative_prompt': negative_prompt,
        'num_inference_steps': args.denoising_steps,
    }
    
    return kwargs_run_demo


if __name__ == "__main__":
    print("[I] Initializing Wan 2.2 txt2vid demo using TensorRT")
    args = parseArgs()
    
    kwargs_run_demo = process_demo_args(args)
    
    _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(args)
    
    # Initialize demo
    demo = pipeline_module.WanPipeline.FromArgs(args, pipeline_type=pipeline_module.PIPELINE_TYPE.TXT2VID)
    
    # Load TensorRT engines and pytorch modules
    demo.load_engines(
        framework_model_dir=args.framework_model_dir,
        **kwargs_load_engine,
    )
    
    # In low-vram mode we allocate the required device memory individually before each model is run
    if args.low_vram:
        demo.device_memory_sizes = demo.get_device_memory_sizes()
    else:
        _, shared_device_memory = cudart.cudaMalloc(demo.calculate_max_device_memory())
        demo.activate_engines(shared_device_memory)
    
    # Load resources
    demo.load_resources(args.height, args.width, args.batch_size, args.seed)
    
    # Run inference
    demo.run(**kwargs_run_demo)
    
    demo.teardown()

