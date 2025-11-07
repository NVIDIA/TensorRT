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

import os
import sys
import argparse
import math
import time
import datetime

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def main(args):

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(0) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(args.original_onnx, 'rb') as onnx_model:
            parser.parse(onnx_model.read())

        with builder.create_builder_config() as config:

            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)

            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch = False)

            profile = builder.create_optimization_profile()
            profile.set_shape("gpu_0/data_0", min=[1, 3, 224, 224], opt=[1, 3, 224, 224], max=[1, 3, 224, 224])
            config.add_optimization_profile(profile)

            def build_and_save_engine(builder, network, config, output):
                start_time = time.time()
                engine_bytes = builder.build_serialized_network(network, config)
                assert engine_bytes is not None
                with open(output, 'wb') as f:
                    f.write(engine_bytes)
                total_time = time.time() - start_time
                print("built and saved {} in time {}".format(output, str(datetime.timedelta(seconds=int(total_time)))))

            # build weight-stripped engine and generate timing cache.
            build_and_save_engine(builder, network, config, args.output_stripped_engine)

            # build normal engine with the same timing cache.
            config.flags &= ~(1 << int(trt.BuilderFlag.STRIP_PLAN))
            build_and_save_engine(builder, network, config, args.output_normal_engine)

def get_default_model_file():
    # Set the data path to the directory that contains the ONNX model.
    _, data_files = common.find_sample_data(
        description="Runs a ResNet50 network with a TensorRT inference engine.",
        subfolder="resnet50",
        find_files=["ResNet50.onnx"],
    )
    onnx_model_file = data_files[0]
    return onnx_model_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--stripped_onnx", default=None, type=str,
                        help="The ONNX model file to load for building stripped engine.")
    parser.add_argument("--original_onnx", default=None, type=str,
                        help="The ONNX model file to load for building normal engine.")
    parser.add_argument("--output_stripped_engine", default='stripped_engine.trt', type=str,
                        help="The output path for the weight-stripped TRT engine.")
    parser.add_argument("--output_normal_engine", default='normal_engine.trt', type=str,
                        help="The output path for the full TRT engine.")
    args, _ = parser.parse_known_args()

    onnx_model_file = get_default_model_file()
    if args.stripped_onnx is None:
        args.stripped_onnx = onnx_model_file
    if args.original_onnx is None:
        args.original_onnx = onnx_model_file

    if not os.path.exists(args.stripped_onnx):
        parser.print_help()
        print(f"--stripped_onnx {args.stripped_onnx} does not exist.")
        sys.exit(1)
    if not os.path.exists(args.original_onnx):
        parser.print_help()
        print(f"--original_onnx {args.original_onnx} does not exist.")
        sys.exit(1)

    main(args)
