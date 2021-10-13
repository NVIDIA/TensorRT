#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

"""
Utils specific to GPT2 network.
"""

# torch
import torch

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda


@use_cuda
def gpt2_inference(gpt2, input_ids, timing_profile, use_cuda=True):
    gpt2_stmt = lambda: gpt2(input_ids=input_ids)
    gpt2_e2e_median_time = measure_python_inference_code(
        gpt2_stmt, number=timing_profile.number, iterations=timing_profile.iterations
    )
    return (gpt2_stmt(), gpt2_e2e_median_time)


# Code specifically for Pythonic inference measurement used across all GPT2 related scripts
@use_cuda
def full_inference_greedy(gpt2, input_ids, timing_profile, max_length, use_cuda=True):
    def _e2e():
        return gpt2.generate(input_ids, max_length=max_length)  # greedy search

    full_e2e_median_time = measure_python_inference_code(
        _e2e,
        number=timing_profile.number,
        iterations=timing_profile.iterations,
    )
    return (_e2e(), full_e2e_median_time)
