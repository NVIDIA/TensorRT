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
This script compares ONNX-Runtime runs of an identity model over a large dataset using streaming
accuracy comparison, so that memory usage stays roughly constant no matter how large the dataset is.

It also demonstrates the "golden run" pattern: stream a fresh run and compare it against reference
outputs saved on disk (in `golden/`, as written by `generate_golden.py`), again at constant memory.
"""

from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.comparator import Comparator, RunResults

# `load_data` is a generator that yields one feed_dict at a time, so the whole dataset is never
#   materialized in memory at once.
from data_loader import load_data


def main():
    # `Comparator.run(..., streaming=True)` is a generator yielding one single-iteration `RunResults`
    #   per input. All runners are activated at once and the outputs are discarded after each
    #   iteration, so memory usage stays roughly constant. `Comparator.compare_accuracy()` accepts a
    #   list of such streams, comparing each iteration as it is produced and accumulating only the
    #   (tiny) accuracy metrics; for a single live run, pass a one-element list.
    #
    # Runners are single-use, so we build fresh ones for each run below; naming them makes the
    #   comparison output (and the saved golden run) easier to read.
    runners = [
        OnnxrtRunner(SessionFromOnnx("identity.onnx"), name="onnxrt-run-0"),
        OnnxrtRunner(SessionFromOnnx("identity.onnx"), name="onnxrt-run-1"),
    ]
    results = Comparator.compare_accuracy(
        [Comparator.run(runners, data_loader=load_data(), streaming=True)]
    )

    # `compare_accuracy` returns an `AccuracyResults` whose `bool(...)` is True only if every result
    #   passed, so `bool(results)` is a correct overall pass/fail check.
    #
    # TIP: The `compare_func` parameter controls how outputs are compared. We use the default
    #   `SimpleCompareFunc` here; see API example 01 for metric-based comparisons (L2, PSNR, ...).
    assert bool(results)
    print("Streaming comparison of two live runs matched")

    # Next, we demonstrate the "golden run" pattern: comparing a live run against reference outputs
    #   saved on disk. The `golden/` directory here was produced by `generate_golden.py`, but in a
    #   real workflow this reference data would already exist, so you would not generate it yourself.
    #
    # `RunResults.load_streaming()` lazily yields the saved golden run one iteration at a time, so
    #   comparing against it also stays at constant memory. `compare_accuracy` merges the two streams
    #   per iteration (each stream's runners appended in order).
    results = Comparator.compare_accuracy(
        [
            Comparator.run(
                [OnnxrtRunner(SessionFromOnnx("identity.onnx"), name="onnxrt")],
                data_loader=load_data(),
                streaming=True,
            ),
            RunResults.load_streaming("golden"),
        ]
    )
    assert bool(results)
    print("Streaming comparison against the saved golden run matched")


if __name__ == "__main__":
    main()
