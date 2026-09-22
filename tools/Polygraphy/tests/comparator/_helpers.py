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
# Shared helpers for building synthetic RunResults in the comparator tests.
import numpy as np

from polygraphy.comparator import IterationResult, RunResults


def _single_iteration(values_by_runner):
    # A single-iteration RunResults mapping each runner name to one IterationResult.
    run_results = RunResults()
    for name, values in values_by_runner.items():
        run_results.append(
            (
                name,
                [
                    IterationResult(
                        outputs={"out": np.array(values, dtype=np.float32)},
                        runner_name=name,
                    )
                ],
            )
        )
    return run_results


def _compare_accuracy(outputs_by_runner, compare_func, check_average=False):
    # Build a single-iteration RunResults from {runner_name: {output_name: values}} and run
    # Comparator.compare_accuracy on it, returning the AccuracyResults.
    from polygraphy.comparator import Comparator

    run_results = RunResults()
    for name, outputs in outputs_by_runner.items():
        run_results[name] = [
            IterationResult(
                outputs={
                    out: np.array(values, dtype=np.float32)
                    for out, values in outputs.items()
                },
                runner_name=name,
            )
        ]
    return Comparator.compare_accuracy(
        run_results, compare_func=compare_func, check_average=check_average
    )


def _multi_iteration_results(arrays_by_runner):
    run_results = RunResults()
    for name, arrays in arrays_by_runner.items():
        run_results[name] = [
            IterationResult(
                outputs={"out": np.array(a, dtype=np.float32)}, runner_name=name
            )
            for a in arrays
        ]
    return run_results
