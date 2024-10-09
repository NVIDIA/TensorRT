#!/usr/bin/env python3
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
trtexec log file parsing
"""


import re
from typing import Tuple, List, Dict, Any
import argparse
import trex.archiving as archiving


def __to_float(line: str) -> float:
    """Scan the input string and extract the first float instance."""
    # https://docs.python.org/3/library/re.html#simulating-scanf
    float_match = re.search(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
    if float_match is None:
        raise ValueError
    start, end = float_match.span()
    return float(line[start : end])


def __get_stats(line: str) -> List[float]:
    """Parse a string containing pairs of "key = value" and return the list of values.

    Here's a sample input line: "min = 0.87854 ms, max = 0.894043 ms, mean = 0.881251 ms"
    The values are expected to be floats.
    Split the kv list to "k = v" substrings, then split each substring to
    k, v and return float(v)
    """
    return [__to_float(substr.split("=")[1]) for substr in line.split(",")]


class FileSection:
    def __init__(self, section_header: str):
        self.section_header = section_header
        self.dict = {}

    def entered_section(self, line: str):
        s = re.search(self.section_header, line)
        return s is not None

    def parse_line(self, line: str) -> bool:
        def parse_kv_line(line: str) -> Tuple[Any, Any]:
            """Parse a log line that reports a key-value pair.

            The log line has this format: [mm/dd/yyyy-hh:mm:ss] [I] key_name: key_value
            """
            match = re.search(r'(\[\d+/\d+/\d+-\d+:\d+:\d+\] \[I\] )', line)
            if match is not None:
                match_end = match.span()[1]
                kv_line = line[match_end:].strip()
                if not kv_line.count(":"):
                    return None, None
                kv = kv_line.split(":")
                if len(kv) > 1:
                    return kv[0], kv[1][1:]
                if len(kv) == 1:
                    return kv[0], None
            return None, None

        k, v = parse_kv_line(line)
        if k is not None and v is not None:
            self.dict[k] = v
            return True
        if k is not None:
            return True
        return False


def __parse_log_file(file_name: str, sections: List, tea: archiving.EngineArchive) -> List[Dict]:
    def entered_section(sections, line) -> bool:
        for section in sections:
            if section.entered_section(line):
                return section
        return None

    current_section = None
    with archiving.get_reader(tea, file_name) as reader:
        for line in reader.readlines():
            if current_section is None:
                current_section = entered_section(sections, line)
            else:
                if not current_section.parse_line(line):
                    sections.remove(current_section)
                    current_section = entered_section(sections, line)
    dicts = [section.dict for section in sections]
    return dicts


def parse_build_log(file_name: str, tea: archiving.EngineArchive) -> List[Dict]:
    """Parse the TensorRT engine build log and extract the builder configuration.

    Returns the model and engine build configurations as dictionaries.
    """
    model_options = FileSection("=== Model Options ===")
    build_options = FileSection("=== Build Options ===")
    device_information = FileSection("=== Device Information ===")

    sections = [model_options, build_options, device_information]
    __parse_log_file(file_name, sections, tea)
    return {
        "model_options": model_options.dict,
        "build_options": build_options.dict,
        "device_information": device_information.dict
    }


def parse_profiling_log(file_name: str, tea: archiving.EngineArchive):
    performance_summary = FileSection("=== Performance summary ===")
    inference_options = FileSection("=== Inference Options ===")
    device_information = FileSection("=== Device Information ===")
    sections = [
        performance_summary,
        inference_options,
        device_information]
    __parse_log_file(file_name, sections, tea)

    def post_process_perf(perf_summary: dict):
        """Normalize the log results to a standard format"""
        for k, v in perf_summary.items():
            if k in ["Throughput", "Total Host Walltime", "Total GPU Compute Time"]:
                perf_summary[k] = __to_float(v)
            if k in ["Latency", "Enqueue Time", "H2D Latency", "GPU Compute Time", "D2H Latency"]:
                perf_summary[k] = __get_stats(v)
        return perf_summary

    def post_process_device_info(device_info: dict):
        """Convert some value fields to float"""
        for k, v in device_info.items():
            if k in ["Compute Clock Rate", "Memory Bus Width", "Memory Clock Rate",
                "Compute Capability", "SMs"]:
                device_info[k] = __to_float(v)
        return device_info

    return {
        "performance_summary": post_process_perf(performance_summary.dict),
        "inference_options": inference_options.dict,
        "device_information": post_process_device_info(device_information.dict)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="name of engine build log file to parse.")
    args = parser.parse_args()
    parse_build_log(args.input)