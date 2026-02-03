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

from cuda.bindings import runtime as cudart

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Txt2Img Demo")
    parser = dd_argparse.add_arguments(parser)
    return parser.parse_args()

if __name__ == "__main__":
    print("[I] Initializing StableDiffusion txt2img demo using TensorRT")
    args = parseArgs()

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = dd_argparse.process_pipeline_args(args)

    # Initialize demo
    demo = pipeline_module.StableDiffusionPipeline(
        pipeline_type=pipeline_module.PIPELINE_TYPE.TXT2IMG, **kwargs_init_pipeline
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
    demo.run(*args_run_demo)

    demo.teardown()
