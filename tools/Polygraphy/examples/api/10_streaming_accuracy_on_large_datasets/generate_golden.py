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
Generates the "golden" reference outputs that example.py compares against, saving them to the
`golden/` directory.

This script exists only to make the example self-contained. In a real workflow you would *already*
have golden data -- from a trusted backend, a previous release, or a reference implementation -- so
you would not normally generate it like this.
"""

from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.comparator import Comparator

# `load_data` yields one feed_dict at a time, so the whole dataset is never materialized in memory.
from data_loader import load_data


def main():
    # Stream one run and save each iteration's outputs as it is produced. An extensionless
    #   `save_outputs_path` writes one JSON file per iteration into a *directory* (which must be
    #   empty or not yet exist), keeping memory roughly constant.
    #
    # `Comparator.run(streaming=True)` is a lazy generator -- no inference runs until it is
    #   consumed -- so we drain it with a `for` loop purely for the side effect of writing `golden/`.
    for _ in Comparator.run(
        [OnnxrtRunner(SessionFromOnnx("identity.onnx"), name="golden")],
        data_loader=load_data(),
        save_outputs_path="golden",
        streaming=True,
    ):
        pass
    print("Saved golden outputs to 'golden/'")


if __name__ == "__main__":
    main()
