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
trtexec log file parsing
"""


import json
from pickle import BUILD
import re
from typing import Tuple, List, Dict, Any
from enum import Enum


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

    def parse_line(self, line: str):
        def parse_kv_line(line: str) -> Tuple[Any, Any]:
            """Parse a log line that reports a key-value pair.

            The log line has this format: [mm/dd/yyyy-hh:mm:ss] [I] key_name: key_value
            """
            match = re.search(r'(\[\d+/\d+/\d+-\d+:\d+:\d+\] \[I\] )', line)
            if match is not None:
                match_end = match.span()[1]
                kv_line = line[match_end:].strip()
                kv = kv_line.split(": ")
                if len(kv) > 1:
                    return kv[0], kv[1]
            return None, None

        k,v = parse_kv_line(line)
        if k is not None and v is not None:
            self.dict[k] = v
            return True
        if k is not None:
            return True
        return False


def __parse_log_file(file_name: str, sections: List) -> List[Dict]:
    current_section = None
    with open(file_name, "r") as file:
        for line in file.readlines():
            if current_section is None:
                for section in sections:
                    if section.entered_section(line):
                        current_section = section
                        break
            else:
                if not current_section.parse_line(line):
                    current_section = None
    dicts = [section.dict for section in sections]
    return dicts


def parse_build_log(file_name: str) -> List[Dict]:
    """Parse the TensorRT engine build log and extract the builder configuration.

    Returns the model and engine build configurations as dictionaries.
    """
    model_options = FileSection("=== Model Options ===")
    build_options = FileSection("=== Build Options ===")
    sections = [model_options, build_options]
    __parse_log_file(file_name, sections)
    return {
        "model_options": model_options.dict,
        "build_options": build_options.dict,
    }


def parse_profiling_log(file_name: str):
    performance_summary = FileSection("=== Performance summary ===")
    inference_options = FileSection("=== Inference Options ===")
    device_information = FileSection("=== Device Information ===")
    sections = [
        performance_summary,
        inference_options,
        device_information]
    __parse_log_file(file_name, sections)

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
