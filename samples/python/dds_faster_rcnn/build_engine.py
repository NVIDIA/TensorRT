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

import os
import sys
import logging
import argparse

import tensorrt as trt

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        one_GiB = 2**30
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * one_GiB)
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """

        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                raise RuntimeError(
                    f"Failed to load ONNX file: {onnx_path}. Check the logs for more details or run with --verbose."
                )

        log.info("Network Description")

        profile = self.builder.create_optimization_profile()
        profile.set_shape("image", min=(3, 1, 1), opt=(3, 800, 800), max=(3, 800, 1312))
        self.config.add_optimization_profile(profile)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        for input in inputs:
            log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        for output in outputs:
            log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    def create_engine(
        self,
        engine_path,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info(f"Building Engine in {engine_path}")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        log.info(f"Reading timing cache from file: {args.timing_cache}")
        common.setup_timing_cache(self.config, args.timing_cache)

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create engine. Check the logs for more details or run with --verbose.")

        log.info(f"Serializing timing cache to file: {args.timing_cache}")
        common.save_timing_cache(self.config, args.timing_cache)

        with open(engine_path, "wb") as f:
            log.info(f"Serializing engine to file: {engine_path}")
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", required=True, help="The output path for the TRT engine")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument(
        "-w",
        "--workspace",
        default=8,
        type=int,
        help="The max memory workspace size to allow in Gb, default: 8",
    )
    parser.add_argument(
        "--timing_cache",
        default="./timing.cache",
        help="The file path for timing cache, default: ./timing.cache",
    )
    args = parser.parse_args()
    main(args)
