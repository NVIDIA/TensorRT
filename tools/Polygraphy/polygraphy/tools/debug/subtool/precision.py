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

import math

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import ModelArgs, TrtConfigArgs
from polygraphy.tools.debug.subtool.base import BaseCheckerSubtool

trt = mod.lazy_import("tensorrt")
trt_util = mod.lazy_import("polygraphy.backend.trt.util")


class BaseMarker:
    def __init__(self, max_layers, direction, num_layers_to_mark):
        self.max_layers = max_layers
        self.direction = direction
        self.num_layers_to_mark = num_layers_to_mark
        self.good = max_layers + 1  # Pretend marking all the layers gives us good accuracy.

    def select_layers(self):
        if self.direction == "forward":
            G_LOGGER.info(f"Selecting first {self.num_layers_to_mark} layer(s) to run in higher precision")
            return range(0, self.num_layers_to_mark)
        else:
            G_LOGGER.info(f"Selecting last {self.num_layers_to_mark} layer(s) to run in higher precision")
            return range(self.max_layers - self.num_layers_to_mark, self.max_layers)

    def success_message(self):
        which_layers = "first" if self.direction == "forward" else "last"
        G_LOGGER.finish(
            f"To achieve acceptable accuracy, try running the {which_layers} {self.good} layer(s) in higher precision"
        )


class BisectMarker(BaseMarker):
    def __init__(self, max_layers, direction) -> None:
        super().__init__(max_layers, direction, max_layers)
        self.bad = 0

    def step(self, success):
        if success:
            self.good = self.num_layers_to_mark
            # On successes, we want num_layers_to_mark to go closer to self.bad
            round_func = math.floor
        else:
            self.bad = self.num_layers_to_mark
            round_func = math.ceil

        old_num_layers_to_mark = self.num_layers_to_mark
        self.num_layers_to_mark = int(round_func((self.good + self.bad) / 2.0))

        # Prevent infinite looping:
        if old_num_layers_to_mark == self.num_layers_to_mark:
            return True

        # If good and bad are within 1 layer of each other,
        # then we already have the information we need.
        if abs(self.good - self.bad) <= 1:
            if self.good >= self.max_layers:
                G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            else:
                self.success_message()
            return True

        if self.num_layers_to_mark > self.max_layers:
            G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            return True

        return False

    def remaining(self):
        return int(math.ceil(math.log2(self.good - self.bad)))


class LinearMarker(BaseMarker):
    def __init__(self, max_layers, direction) -> None:
        super().__init__(max_layers, direction, 0)

    def step(self, success):
        if success:
            self.good = self.num_layers_to_mark
        self.num_layers_to_mark += 1

        if success:
            self.success_message()
            return True

        if self.num_layers_to_mark > self.max_layers:
            G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            return True

        return False

    def remaining(self):
        return self.max_layers - self.num_layers_to_mark


class Precision(BaseCheckerSubtool):
    """
    [EXPERIMENTAL] Iteratively mark layers to run in a higher precision to find a compromise between performance and quality.

    `debug precision` follows the same general process as other `debug` subtools (refer to the help output
    of the `debug` tool for more background information and details).

    Each iteration will generate an engine called 'polygraphy_debug.engine' in the current directory.
    """

    def __init__(self):
        super().__init__("precision", precision_constraints_default="obey", allow_no_artifacts_warning=False)

    def add_parser_args(self, parser):
        parser.add_argument(
            "--mode",
            help="How layers are selected to run in higher precision. "
            "'bisect' will use binary search, and 'linear' will iteratively mark one extra layer at a time",
            choices=["bisect", "linear"],
            default="bisect",
        )
        parser.add_argument(
            "--dir",
            "--direction",
            help="Order in which layers are marked to run in higher precision. "
            "'forward' will start marking layers from network inputs, and 'reverse' will start "
            "from the network outputs",
            choices=["forward", "reverse"],
            default="forward",
            dest="direction",
        )
        parser.add_argument(
            "-p",
            "--precision",
            help="Precision to use when marking layers to run in higher precision",
            choices=["float32", "float16"],
            default="float32",
        )

    def setup(self, args, network):
        self.precision = {"float32": trt.float32, "float16": trt.float16}[args.precision]

        if self.precision == trt.float16 and not self.arg_groups[TrtConfigArgs].fp16:
            G_LOGGER.critical(
                "Cannot mark layers to run in float16 if it is not enabled in the builder configuration.\n"
                "Please also specify `--fp16` as a command-line option"
            )

        if self.precision == trt.float16 and not self.arg_groups[TrtConfigArgs].int8:
            G_LOGGER.warning(
                "Using float16 as the higher precision, but float16 is also the lowest precision available. "
                "Did you mean to set --int8 as well?"
            )

        if not any(
            [
                self.arg_groups[TrtConfigArgs].tf32,
                self.arg_groups[TrtConfigArgs].fp16,
                self.arg_groups[TrtConfigArgs].int8,
            ]
        ):
            G_LOGGER.critical("Please enable at least one precision besides float32 (e.g. --int8, --fp16, --tf32)")

        if self.arg_groups[ModelArgs].model_type == "engine":
            G_LOGGER.critical(
                "The precision tool cannot work with engines, as they cannot be modified. "
                "Please provide a different format, such as an ONNX model or TensorRT network script."
            )

        G_LOGGER.start(f"Using {self.precision} as higher precision")

        if args.mode == "linear":
            self.layer_marker = LinearMarker(len(network), args.direction)
        elif args.mode == "bisect":
            self.layer_marker = BisectMarker(len(network), args.direction)

        self.original_precisions = {}
        for index, layer in enumerate(network):
            if layer.precision_is_set:
                self.original_precisions[index] = layer.precision

    def mark_layers(self, network, indices):
        EXCLUDE_LAYER_NAMES = ["CONSTANT"]
        EXCLUDE_LAYERS = [getattr(trt.LayerType, attr) for attr in EXCLUDE_LAYER_NAMES if hasattr(trt.LayerType, attr)]

        # First, reset, since changes from the previous call will persist.
        for index, layer in enumerate(network):
            if index in self.original_precisions:
                layer.precision = self.original_precisions[index]
            else:
                layer.reset_precision()

        marked_indices = set()
        for index in indices:
            layer = network.get_layer(index)

            def should_exclude():
                has_non_execution_output = any(
                    not layer.get_output(i).is_execution_tensor for i in range(layer.num_outputs)
                )
                return layer.type in EXCLUDE_LAYERS or has_non_execution_output

            if not should_exclude():
                G_LOGGER.extra_verbose(f"Running layer in higher precision: {trt_util.str_from_layer(layer, index)}")
                layer.precision = self.precision
                marked_indices.add(index)

        G_LOGGER.verbose(f"Marking layer(s): {marked_indices} to run in {self.precision} precision")

    def process_network(self, network):
        indices = list(self.layer_marker.select_layers())
        self.mark_layers(network, indices)

    def step(self, success):
        return self.layer_marker.step(success)

    def remaining(self):
        return self.layer_marker.remaining()
