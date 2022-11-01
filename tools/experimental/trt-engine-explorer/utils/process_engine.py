#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Use this script to:
1. Build a TensorRT engine from an ONNX file.
2. Profile an engine plan file.
3. Generate JSON files for exploration with trex.
4. Draw an SVG graph from an engine.

Note: this script requires Graphviz for function `draw_engine`.
Graphviz can be installed manually:
    $ sudo apt-get --yes install graphviz
"""


import os
import json
import argparse
import subprocess
from typing import List, Dict, Tuple
from parse_trtexec_log import parse_build_log, parse_profiling_log
from config_gpu import GPUMonitor, GPUConfigurator, get_max_clocks


def run_trtexec(trt_cmdline: List[str], build_log_file: str):
    '''Execute trtexec'''
    success = False
    with open(build_log_file, 'w') as logf:
        log_str = None
        try:
            log = subprocess.run(
                trt_cmdline,
                check=True,
                # Redirection
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True)
            success = True
            log_str = log.stdout
        except subprocess.CalledProcessError as err:
            log_str = err.output
        except FileNotFoundError as err:
            log_str = f"\nError: {err.strerror}: {err.filename}"
            print(log_str)
        logf.write(log_str)
    return success


def build_engine_cmd(
    args: Dict,
    onnx_path: str,
    engine_path: str,
    timing_cache_path: str
) -> Tuple[List[str], str]:
    cmd_line = ["trtexec",
        "--verbose",
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        "--nvtxMode=verbose",
        "--buildOnly",
        "--workspace=8192",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    append_trtexec_args(args.trtexec, cmd_line)

    build_log_fname = f"{engine_path}.build.log"
    return cmd_line, build_log_fname


def build_engine(
    args: Dict,
    timing_cache_path: str
) -> bool:
    def generate_build_metadata(log_file: str, output_json: str):
        """Parse trtexec engine build log file and write to a JSON file"""
        build_metadata = parse_build_log(log_file)
        with open(output_json, 'w') as fout:
            json.dump(build_metadata , fout)
            print(f"Engine building metadata: generated output file {output_json}")

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
        build_md_json_fname = f"{engine_path}.build.metadata.json"
        generate_build_metadata(build_log_file, build_md_json_fname)
    else:
        print("\nFailed to build the engine.")
        print(f"See logfile in: {build_log_file}\n")
        print("Troubleshooting:")
        print("1. Make sure that you are running this script in an environment "
              "which has trtexec built and accessible from $PATH.")
        print("2. If this is a Jupyter notebook, make sure the "
              " trtexec is in the $PATH of the Jupyter server.")
    return success


def profile_engine_cmd(
    args: Dict,
    engine_path:str,
    timing_cache_path: str
):
    profiling_json_fname = f"{engine_path}.profile.json"
    graph_json_fname = f"{engine_path}.graph.json"
    timing_json_fname = f"{engine_path}.timing.json"
    cmd_line = ["trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        # Profiling affects the performance of your kernel!
        # Always run and time without profiling.
        "--separateProfileRun",
        "--useSpinWait",
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        "--nvtxMode=verbose",
        f"--loadEngine={engine_path}",
        f"--exportTimes={timing_json_fname}",
        f"--exportProfile={profiling_json_fname}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]

    append_trtexec_args(args.trtexec, cmd_line)

    profile_log_fname = f"{engine_path}.profile.log"
    return cmd_line, profile_log_fname


def get_engine_path(args: Dict, add_suffix: bool):
    if add_suffix:
        onnx_path = args.input
        onnx_fname = os.path.basename(onnx_path)
        outdir = args.outdir
        engine_path = os.path.join(outdir, onnx_fname) + ".engine"
    else:
        engine_path = args.input
    return engine_path


def get_gpu_config_args(args):
    def freq_to_int(clk_freq: str, max_clk_freq: int):
        use_max_freq = clk_freq == 'max'
        clk_freq = max_clk_freq if use_max_freq else int(clk_freq)
        clk_freq = min(max_clk_freq, clk_freq)
        return clk_freq

    dev = args.dev
    power_limit = args.power_limit
    dont_lock_clocks = args.dont_lock_clocks
    # Parse frequencies arguments.
    max_mem_clk_freq, max_compute_clk_freq = get_max_clocks(dev)
    compute_clk_freq = freq_to_int(args.compute_clk_freq, max_compute_clk_freq)
    mem_clk_freq = freq_to_int(args.memory_clk_freq, max_mem_clk_freq)
    return power_limit, compute_clk_freq, mem_clk_freq, dev, dont_lock_clocks


def profile_engine(
    args: Dict,
    timing_cache_path:str,
    add_suffix: bool
) -> bool:
    def generate_profiling_metadata(log_file: str, output_json: str):
        """Parse trtexec profiling session log file and write to a JSON file"""
        profiling_metadata = parse_profiling_log(log_file)
        with open(output_json, 'w') as fout:
            json.dump(profiling_metadata , fout)
            print(f"Profiling metadata: generated output file {output_json}")

    engine_path = get_engine_path(args, add_suffix)

    print("Profiling the engine:")
    cmd_line, profile_log_file = profile_engine_cmd(
        args, engine_path, timing_cache_path)
    print(" ".join(cmd_line))
    if args.print_only:
        return True

    with GPUMonitor(args.monitor), GPUConfigurator(*get_gpu_config_args(args)):
        success = run_trtexec(cmd_line, profile_log_file)

    if success:
        print("\nSuccessfully profiled the engine.\n")
        profiling_md_json_fname = f"{engine_path}.profile.metadata.json"
        generate_profiling_metadata(profile_log_file, profiling_md_json_fname)
    else:
        print("\nFailed to profile the engine.")
        print(f"See logfile in: {profile_log_file}\n")
    return success


def generate_engine_svg(args: Dict, add_suffix: bool) -> bool:
    if args.print_only:
        return
    engine_path = get_engine_path(args, add_suffix)

    if add_suffix:
        graph_json_fname = f"{engine_path}.graph.json"
    else:
        graph_json_fname = engine_path

    try:
        from draw_engine import draw_engine
        print(f"Generating graph diagram: {graph_json_fname}")
        draw_engine(graph_json_fname)
    except ModuleNotFoundError:
        print("Can't generate plan SVG graph because some package is not installed")


def create_artifacts_directory(path: str):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def process_engine(
    args: Dict,
    build: bool,
    profile: bool,
    draw: bool
) -> bool:
    timing_cache_path = "./timing.cache"
    success = True
    if build:
        success = build_engine(args, timing_cache_path)
    if profile and success:
        success = profile_engine(args, timing_cache_path, add_suffix=build)
    if draw and success:
        success = generate_engine_svg(args, add_suffix=build)
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
    parser.add_argument('--memory-clk-freq',
        default='max',
        help="Set memory clock frequency (MHz)")
    parser.add_argument('--compute-clk-freq',
        default='max',
        help="Set compute clock frequency (MHz)")
    parser.add_argument('--power-limit', default=None, type=int, help="Set power limit")
    parser.add_argument('--dev', default=0, help="GPU device ID")
    parser.add_argument('--dont-lock-clocks',
        action='store_true',
        help="Do not lock the clocks. "
             "If set, overrides --compute-clk-freq and --memory-clk-freq")
    parser.add_argument('--monitor',
        action='store_true',
        help="Monitor GPU temperature, power, clocks and utilization while profiling.")

    parser.add_argument('--print-only', action='store_true',
        help='print the command-line and exit')
    parser.add_argument('--build-engine', '-b', action='store_true', default=None,
        help='build the engine')
    parser.add_argument('--profile-engine', '-p', action='store_true', default=None,
        help='profile the engine')
    parser.add_argument('--draw-engine', '-d', action='store_true', default=None,
        help='draw the engine')
    args = parser.parse_args()
    return args


def append_trtexec_args(trt_args: Dict, cmd_line: List[str]):
    for arg in trt_args:
        cmd_line.append(f"--{arg}")


def get_subcmds(args: Dict):
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
