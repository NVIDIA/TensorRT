# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Copies a template file from a source directory (e.g. packaging/) to a destination (usually a build dir),
replacing variables (e.g. `##TENSORRT_VERSION##`) with concrete values.
Sub-directory structure is preserved.

This file is used by the CMake build when building the python wheels.
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(
        description="Copy files from a source to a destination, replacing variables with concrete values"
    )
    parser.add_argument("--src-dir", help="The absolute path to the source directory.", required=True)
    parser.add_argument("--dst-dir", help="The absolute path to the destination directory.", required=True)
    parser.add_argument("--filepath", help="The template file to copy, relative to the source directory.", required=True)
    parser.add_argument("--trt-module", help="The TensorRT module name. One of 'tensorrt', 'tensorrt_lean', or 'tensorrt_dispatch' for non-RTX, and 'tensorrt_rtx' for RTX builds.", required=True)
    parser.add_argument("--trt-py-version", help="The version string for the python bindings being built. Usually `major.minor.patch.build`.", required=True)
    parser.add_argument("--cuda-version", help="The Cuda version (major.minor).", required=True)
    parser.add_argument("--trt-version", help="The TensorRT version (major.minor.patch).", required=True)
    parser.add_argument("--plugin-disabled", help="Whether the plugin is disabled.",  type=int, choices=[0,1], default=0, required=False)
    parser.add_argument("--trt-nvinfer-name", help="The name of the nvinfer library.", required=True)
    parser.add_argument("--trt-onnxparser-name", help="The name of the onnxparser library.", required=True)
    args, _ = parser.parse_known_args()

    if not os.path.isdir(args.src_dir):
        raise ValueError(f"Provided src-dir {args.src_dir} is not a directory.")

    if not os.path.isdir(args.dst_dir):
        raise ValueError(f"Provided dst-dir {args.dst_dir} is not a directory.")

    target_path = os.path.join(args.src_dir, args.filepath)

    if not os.path.exists(target_path):
        raise ValueError(f"Target file {target_path} does not exist.")

    with open(target_path, 'r', encoding="utf-8") as file:
        contents = file.read()
        contents = contents.replace("##TENSORRT_MODULE##", args.trt_module)
        contents = contents.replace("##TENSORRT_PYTHON_VERSION##", args.trt_py_version)
        contents = contents.replace("##CUDA_MAJOR##", args.cuda_version.split(".")[0])
        contents = contents.replace("##TENSORRT_MAJOR##", args.trt_version.split(".")[0])
        contents = contents.replace("##TENSORRT_MINOR##", args.trt_version.split(".")[1])
        contents = contents.replace("##TENSORRT_PLUGIN_DISABLED##", "True" if args.plugin_disabled == 1 else "False")
        contents = contents.replace("##TENSORRT_NVINFER_NAME##", args.trt_nvinfer_name)
        contents = contents.replace("##TENSORRT_ONNXPARSER_NAME##", args.trt_onnxparser_name)

        dest_path = os.path.join(args.dst_dir, args.filepath)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'w', encoding="utf-8") as of:
            of.write(contents)


if __name__ == "__main__":
    main()
