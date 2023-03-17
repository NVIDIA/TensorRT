#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

from image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


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
        self.config.max_workspace_size = workspace * (2 ** 30)

        self.network = None
        self.parser = None

    def create_network(self, onnx_path, batch_size, dynamic_batch_size=None):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        :param batch_size: Static batch size to build the engine with.
        :param dynamic_batch_size: Dynamic batch size to build the engine with, if given,
        batch_size is ignored, pass as a comma-separated string or int list as MIN,OPT,MAX
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        log.info("Network Description")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        profile = self.builder.create_optimization_profile()
        dynamic_inputs = False
        for input in inputs:
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
            if input.shape[0] == -1:
                dynamic_inputs = True
                if dynamic_batch_size:
                    if type(dynamic_batch_size) is str:
                        dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                    assert len(dynamic_batch_size) == 3
                    min_shape = [dynamic_batch_size[0]] + list(input.shape[1:])
                    opt_shape = [dynamic_batch_size[1]] + list(input.shape[1:])
                    max_shape = [dynamic_batch_size[2]] + list(input.shape[1:])
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape)
                    log.info("Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(
                        input.name, min_shape, opt_shape, max_shape))
                else:
                    shape = [batch_size] + list(input.shape[1:])
                    profile.set_shape(input.name, shape, shape, shape)
                    log.info("Input '{}' Optimization Profile with shape {}".format(input.name, shape))
        if dynamic_inputs:
            self.config.add_optimization_profile(profile)

        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    def set_mixed_precision(self):
        """
        Experimental precision mode.
        Enable mixed-precision mode. When set, the layers defined here will be forced to FP16 to maximize
        INT8 inference accuracy, while having minimal impact on latency.
        """
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # All convolution operations in the first four blocks of the graph are pinned to FP16.
        # These layers have been manually chosen as they give a good middle-point between int8 and fp16
        # accuracy in COCO, while maintining almost the same latency as a normal int8 engine.
        # To experiment with other datasets, or a different balance between accuracy/latency, you may
        # add or remove blocks.
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if layer.type == trt.LayerType.CONVOLUTION and any([
                    # AutoML Layer Names:
                    "/stem/" in layer.name,
                    "/blocks_0/" in layer.name,
                    "/blocks_1/" in layer.name,
                    "/blocks_2/" in layer.name,
                    # TFOD Layer Names:
                    "/stem_conv2d/" in layer.name,
                    "/stack_0/block_0/" in layer.name,
                    "/stack_1/block_0/" in layer.name,
                    "/stack_1/block_1/" in layer.name,
                ]):
                self.network.get_layer(i).precision = trt.DataType.HALF
                log.info("Mixed-Precision Layer {} set to HALF STRICT data type".format(layer.name))

    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16', 'int8', or 'mixed'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision in ["fp16", "int8", "mixed"]:
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)
        if precision in ["int8", "mixed"]:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = EngineCalibrator(calib_cache)
            if calib_cache is None or not os.path.exists(calib_cache):
                calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                calib_dtype = trt.nptype(inputs[0].dtype)
                self.config.int8_calibrator.set_image_batcher(
                    ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                 exact_batches=True, shuffle_files=True))

        engine_bytes = None
        try:
            engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        except AttributeError:
            engine = self.builder.build_engine(self.network, self.config)
            engine_bytes = engine.serialize()
            del engine
        assert engine_bytes
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.batch_size, args.dynamic_batch_size)
    if args.precision == "mixed":
        builder.set_mixed_precision()
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True,
                        help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", required=True,
                        help="The output path for the TRT engine")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="The static batch size to build the engine with, default: 1")
    parser.add_argument("-d", "--dynamic_batch_size", default=None,
                        help="Enable dynamic batch size by providing a comma-separated MIN,OPT,MAX batch size, "
                             "if this option is set, --batch_size is ignored, example: -d 1,16,32, "
                             "default: None, build static engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8", "mixed"],
                        help="The precision mode to build in, either fp32/fp16/int8/mixed, default: fp16")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=8, type=int,
                        help="The max memory workspace size to allow in Gb, default: 8")
    parser.add_argument("--calib_input",
                        help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default=None,
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    args = parser.parse_args()
    if args.precision in ["int8", "mixed"] and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 or mixed precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    main(args)
