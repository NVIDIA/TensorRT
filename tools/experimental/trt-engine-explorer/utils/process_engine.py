#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import cmd
import os
import argparse
import subprocess
from typing import List
from device_info import device_info


def run_trtexec(trt_cmdline: List[str], build_log_file: str):
    '''Execute trtexec'''
    success = False
    with open(build_log_file, 'w') as logf:
        log_str = None
        try:
            log = subprocess.run(
                trt_cmdline, check=True,
                # Redirection
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            success = True
            log_str = log.stdout
        except subprocess.CalledProcessError as err:
            log_str = err.output
        except FileNotFoundError as err:
            log_str = f"\nError: {err.strerror}: {err.filename}"
            print(log_str)
        logf.write(log_str)
    return success


def build_engine_cmd(args, onnx_path: str, engine_path: str, timing_cache_path: str):
    cmd_line = ["trtexec",
        "--verbose",
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        "--nvtxMode=verbose",
        "--buildOnly",
        "--workspace=1024",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    add_trtexec_args(args.trtexec, cmd_line)
    build_log_file = os.path.join(args.outdir, "log.build.txt")
    return cmd_line, build_log_file


def build_engine(args, timing_cache_path: str) -> bool:
    onnx_path = args.input
    onnx_fname = os.path.basename(onnx_path)
    outdir = args.outdir
    engine_path = os.path.join(outdir, onnx_fname) + ".engine"

    print("Building the engine:")
    cmd_line, build_log_file = build_engine_cmd(
        args, onnx_path, engine_path, timing_cache_path)
    print(" ".join(cmd_line))
    if args.print_only:
        return True
    success = run_trtexec(cmd_line, build_log_file)
    if success:
        print("\nSuccessfully built the engine.\n")
    else:
        print("\nFailed to build the engine.")
        print(f"See logfile in: {build_log_file}\n")
        print("Troubleshooting:")
        print("1. Make sure that you are running this script in an environment "
              "which has trtexec built and accessible from $PATH.")
        print("2. If this is a Jupyter notebook, make sure the "
              " trtexec is in the $PATH of the Jupyter server.")
    return success


def profile_engine_cmd(args, engine_path:str, timing_cache_path: str):
    profiling_json_fname = f"{engine_path}.profile.json"
    graph_json_fname = f"{engine_path}.graph.json"
    timing_json_fname = f"{engine_path}.timing.json"
    cmd_line = ["trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        "--separateProfileRun",
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        "--nvtxMode=verbose",
        f"--loadEngine={engine_path}",
        f"--exportTimes={timing_json_fname}",
        f"--exportProfile={profiling_json_fname}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]

    add_trtexec_args(args.trtexec, cmd_line)

    profile_log_file = os.path.join(args.outdir, "log.profile.txt")
    return cmd_line, profile_log_file


def get_engine_path(args, from_onnx: bool):
    if from_onnx:
        onnx_path = args.input
        onnx_fname = os.path.basename(onnx_path)
        outdir = args.outdir
        engine_path = os.path.join(outdir, onnx_fname) + ".engine"
    else:
        engine_path = args.input
    return engine_path


def profile_engine(args, timing_cache_path:str, from_onnx: bool) -> bool:
    engine_path = get_engine_path(args, from_onnx)

    print("Profiling the engine:")
    cmd_line, profile_log_file = profile_engine_cmd(
        args, engine_path, timing_cache_path)
    print(" ".join(cmd_line))
    if args.print_only:
        return True
    success = run_trtexec(cmd_line, profile_log_file)
    if success:
        print("\nSuccessfully profiled the engine.\n")
        metadata_json_fname = f"{engine_path}.metadata.json"
        device_info(metadata_json_fname)
    else:
        print("\nFailed to profile the engine.")
        print(f"See logfile in: {profile_log_file}\n")
    return success


def generate_engine_svg(args, from_onnx: bool) -> bool:
    engine_path = get_engine_path(args, from_onnx)
    if from_onnx:
        graph_json_fname = f"{engine_path}.graph.json"
    else:
        graph_json_fname = engine_path

    try:
        from draw_engine import draw_engine
        draw_engine(graph_json_fname)
    except ModuleNotFoundError:
        print("Can't generate plan SVG graph because some package is not installed")


def create_artifacts_directory(path: str):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def process_engine(args, build: bool, profile: bool, draw: bool) -> bool:
    timing_cache_path = "./timing.cache"
    success = True
    if build:
        success = build_engine(args, timing_cache_path)
    if profile and success:
        success = profile_engine(args, timing_cache_path, from_onnx=build)
    if draw and success:
        success = generate_engine_svg(args, from_onnx=build)
    print(f"Artifcats directory: {args.outdir}")
    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description='Utility to build and profile TensorRT engines.')

    # Positional arguments.
    parser.add_argument('input', help="input file (ONNX model or TensorRT engine file)")
    parser.add_argument('outdir', help="directory to store output artifacts")
    parser.add_argument('trtexec', nargs='*',
        help="trtexec agruments (without a preceding --). "
             "For example: int8 shapes=input_ids:32x512,attention_mask:32x512",
        default=None)

    # Optional arguments.
    parser.add_argument('--print_only', action='store_true',
        help='print the command-line and exit')
    parser.add_argument('--build_engine', '-b', action='store_true', default=None,
        help='build the engine')
    parser.add_argument('--profile_engine', '-p', action='store_true', default=None,
        help='profile the engine')
    parser.add_argument('--draw_engine', '-d', action='store_true', default=None,
        help='draw the engine')
    args = parser.parse_args()
    return args


def add_trtexec_args(trt_args, cmd_line):
    for arg in trt_args:
        cmd_line.append(f"--{arg}")


def get_subcmds(args):
    all = (not args.build_engine and
           not args.profile_engine and
           not args.draw_engine)
    build, profile, draw = [True]*3 if all else [
        args.build_engine, args.profile_engine, args.draw_engine]
    return build, profile, draw


if __name__ == "__main__":
    args = parse_args()
    create_artifacts_directory(args.outdir)
    build, profile, draw = get_subcmds(args)
    process_engine(args, build, profile, draw)
