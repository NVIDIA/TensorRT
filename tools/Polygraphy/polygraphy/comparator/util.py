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
import functools

from polygraphy import mod, util, config
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")


def cast_up(buffer):
    dtype = np.dtype(buffer.dtype)

    if dtype == np.dtype(np.float16):
        buffer = buffer.astype(np.float32)
    elif dtype in list(map(np.dtype, [np.int8, np.uint8, np.int16, np.uint16])):
        buffer = buffer.astype(np.int32)
    elif dtype == np.dtype(np.uint32):
        buffer = buffer.astype(np.int64)
    return buffer


def use_higher_precision(func):
    """
    Decorator that will cast the input numpy buffer(s) to a higher precision before computation.
    """

    @functools.wraps(func)
    def wrapped(*buffers):
        if any(util.is_empty_shape(buffer.shape) for buffer in buffers):
            return 0

        new_buffers = [cast_up(buffer) for buffer in buffers]
        return func(*new_buffers)

    return wrapped


@use_higher_precision
def compute_max(buffer):
    return np.amax(buffer)


# Returns index of max value
@use_higher_precision
def compute_argmax(buffer):
    return np.unravel_index(np.argmax(buffer), buffer.shape)


@use_higher_precision
def compute_min(buffer):
    return np.amin(buffer)


# Returns index of min value
@use_higher_precision
def compute_argmin(buffer):
    return np.unravel_index(np.argmin(buffer), buffer.shape)


@use_higher_precision
def compute_mean(buffer):
    return np.mean(buffer)


@use_higher_precision
def compute_stddev(buffer):
    return np.std(buffer)


@use_higher_precision
def compute_variance(buffer):
    return np.var(buffer)


@use_higher_precision
def compute_median(buffer):
    return np.median(buffer)


@use_higher_precision
def compute_average_magnitude(buffer):
    return np.mean(np.abs(buffer))


def str_histogram(output, hist_range=None):
    if np.issubdtype(output.dtype, np.bool_):
        return ""

    try:
        try:
            hist, bin_edges = np.histogram(output, range=hist_range)
        except ValueError as err:
            G_LOGGER.verbose(f"Could not generate histogram. Note: Error was: {err}")
            return ""

        max_num_elems = compute_max(hist)
        if not max_num_elems:  # Empty tensor
            return

        bin_edges = [f"{bin:.3g}" for bin in bin_edges]
        max_start_bin_width = max(len(bin) for bin in bin_edges)
        max_end_bin_width = max(len(bin) for bin in bin_edges[1:])

        MAX_WIDTH = 40
        ret = "---- Histogram ----\n"
        ret += f"{'Bin Range':{max_start_bin_width + max_end_bin_width + 5}}|  Num Elems | Visualization\n"
        for num, bin_start, bin_end in zip(hist, bin_edges, bin_edges[1:]):
            bar = "#" * int(MAX_WIDTH * float(num) / float(max_num_elems))
            ret += f"({bin_start:<{max_start_bin_width}}, {bin_end:<{max_end_bin_width}}) | {num:10} | {bar}\n"
        return ret
    except Exception as err:
        G_LOGGER.verbose(f"Could not generate histogram.\nNote: Error was: {err}")
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
        return ""


def str_output_stats(output, runner_name=None):
    ret = ""
    if runner_name:
        ret += f"{runner_name} | Stats: "

    try:
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            ret += f"mean={compute_mean(output):.5g}, std-dev={compute_stddev(output):.5g}, var={compute_variance(output):.5g}, median={compute_median(output):.5g}, min={compute_min(output):.5g} at {compute_argmin(output)}, max={compute_max(output):.5g} at {compute_argmax(output)}, avg-magnitude={compute_average_magnitude(output):.5g}\n"
    except Exception as err:
        G_LOGGER.verbose(f"Could not generate statistics.\nNote: Error was: {err}")
        ret += "<Error while computing statistics>"
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
    return ret


def log_output_stats(output, info_hist=False, runner_name=None, hist_range=None):
    ret = str_output_stats(output, runner_name)
    G_LOGGER.info(ret)
    with G_LOGGER.indent():
        # Show histogram on failures.
        G_LOGGER.log(
            lambda: str_histogram(output, hist_range), severity=G_LOGGER.INFO if info_hist else G_LOGGER.VERBOSE
        )
