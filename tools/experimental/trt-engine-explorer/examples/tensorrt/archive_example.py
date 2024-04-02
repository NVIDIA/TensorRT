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

"""
This file contains an example of creating a TensorRT Engine Archive programmatically.
"""

import logging
import tensorrt as trt
from trex.archiving import EngineArchive


model_name = "./tests/resources/single_relu.onnx"
tea_name = "/tmp/test.tea"

input_shapes = [[1, 3, 224,224], [1, 3, 224,224], [1, 3, 224,224]]


def example_build_engine(tea: EngineArchive, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with tea.Builder(TRT_LOGGER) as builder, builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_name, 'rb') as model:
            if not parser.parse(model.read()):
                logging.error(f"Parsing {model_name} failed")
                for error in range(parser.num_errors):
                    logging.error(parser.get_error(error))
                return

        logging.info('Building the engine...')
        builder.max_workspace_size = 1 << 30
        config = builder.create_builder_config()
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, False)
        optimization_profiles = [builder.create_optimization_profile()]
        for profile in optimization_profiles:
            profile.set_shape("input.1", *input_shapes)
            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        del engine


def example_run_engine(tea: EngineArchive, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    plan = tea.readf("engine.trt")
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)
        with engine.create_execution_context() as context:
            pass
        del engine


def main():
    tea = EngineArchive(tea_name)
    logging.info(f"Building an engine archive from {model_name}")
    example_build_engine(tea)
    example_run_engine(tea)
    logging.info(f"Created an engine archive at {tea_name}")


if __name__ == "__main__":
    main()

