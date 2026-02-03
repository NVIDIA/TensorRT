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
import os

import torch
from cuda.bindings import runtime as cudart

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Cascade Txt2Img Demo", conflict_handler='resolve')
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument('--version', type=str, default="cascade", choices=["cascade"], help="Version of Stable Cascade")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Width of image to generate (must be multiple of 8)")
    parser.add_argument('--lite', action='store_true', help="Use the Lite Version of the Stage B and Stage C models")
    parser.add_argument('--prior-guidance-scale', type=float, default=4.0, help="Value of classifier-free guidance scale for the prior")
    parser.add_argument('--decoder-guidance-scale', type=float, default=0.0, help="Value of classifier-free guidance scale for the decoder")
    parser.add_argument('--prior-denoising-steps', type=int, default=20, help="Number of denoising steps for the prior")
    parser.add_argument('--decoder-denoising-steps', type=int, default=10, help="Number of denoising steps for the decoder")
    return parser.parse_args()


class StableCascadeDemoPipeline(pipeline_module.StableCascadePipeline):
    def __init__(self, prior_denoising_steps, decoder_denoising_steps, prior_guidance_scale, decoder_guidance_scale, lite, **kwargs):
        self.nvtx_profile = kwargs['nvtx_profile']
        self.prior = pipeline_module.StableCascadePipeline(
            pipeline_type=pipeline_module.PIPELINE_TYPE.CASCADE_PRIOR,
            denoising_steps=prior_denoising_steps,
            guidance_scale=prior_guidance_scale,
            return_latents=True,
            lite=lite,
            **kwargs,
        )
        self.decoder = pipeline_module.StableCascadePipeline(
            pipeline_type=pipeline_module.PIPELINE_TYPE.CASCADE_DECODER,
            denoising_steps=decoder_denoising_steps,
            guidance_scale=decoder_guidance_scale,
            lite=lite,
            **kwargs,
        )

    def loadEngines(self, framework_model_dir, onnx_dir, engine_dir, **kwargs):
        prior_suffix = "prior_lite" if self.prior.lite else "prior"
        decoder_suffix = "decoder_lite" if self.decoder.lite else "decoder"
        self.prior.loadEngines(
            os.path.join(engine_dir, prior_suffix),
            framework_model_dir,
            os.path.join(onnx_dir, prior_suffix),
            **kwargs)
        self.decoder.loadEngines(
            os.path.join(engine_dir, decoder_suffix),
            framework_model_dir,
            os.path.join(onnx_dir, decoder_suffix),
            **kwargs)

    def activateEngines(self, shared_device_memory=None):
        self.prior.activateEngines(shared_device_memory)
        self.decoder.activateEngines(shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, seed):
        self.prior.loadResources(image_height, image_width, batch_size, seed)
        # Use a different seed for decoder
        self.decoder.loadResources(image_height, image_width, batch_size, ((seed+1) if seed is not None else None))

    def get_max_device_memory(self):
        max_device_memory = self.prior.calculateMaxDeviceMemory()
        max_device_memory = max(max_device_memory, self.decoder.calculateMaxDeviceMemory())
        return max_device_memory

    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph):
        # Process prompt
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                latents, _ = self.prior.infer(prompt, negative_prompt, height, width, warmup=True)
                latents = latents.to(torch.float16) if self.decoder.fp16 else latents
                images, _ = self.decoder.infer(prompt, negative_prompt, height, width, image_embeddings=latents, warmup=True)

        for _ in range(batch_count):
            print("[I] Running Stable Cascade pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            latents, time_prior = self.prior.infer(prompt, negative_prompt, height, width, warmup=False)
            latents = latents.to(torch.float16) if self.decoder.fp16 else latents
            images, time_decoder = self.decoder.infer(prompt, negative_prompt, height, width, image_embeddings=latents, warmup=False)

            if self.nvtx_profile:
                cudart.cudaProfilerStop()
            print('|-----------------|--------------|')
            print('| {:^15} | {:>9.2f} ms |'.format('e2e', time_prior + time_decoder))
            print('|-----------------|--------------|')

    def teardown(self):
        self.prior.teardown()
        self.decoder.teardown()


if __name__ == "__main__":
    print("[I] Initializing StableCascade txt2img demo using TensorRT")
    args = parseArgs()

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = dd_argparse.process_pipeline_args(args)

    # Initialize demo
    _ = kwargs_init_pipeline.pop('guidance_scale')
    _ = kwargs_init_pipeline.pop('denoising_steps')
    demo = StableCascadeDemoPipeline(
        args.prior_denoising_steps,
        args.decoder_denoising_steps,
        args.prior_guidance_scale,
        args.decoder_guidance_scale,
        args.lite,
        **kwargs_init_pipeline
    )

    # Load TensorRT engines and pytorch modules
    demo.loadEngines(
        args.framework_model_dir,
        args.onnx_dir,
        args.engine_dir,
        **kwargs_load_engine,
    )

    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(demo.get_max_device_memory())
    demo.activateEngines(shared_device_memory)
    demo.loadResources(args.height, args.width, args.batch_size, args.seed)

    # Run inference
    demo.run(*args_run_demo)

    demo.teardown()
