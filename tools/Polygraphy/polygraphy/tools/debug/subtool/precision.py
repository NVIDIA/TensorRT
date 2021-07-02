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

import math

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import ModelArgs, TrtConfigArgs
from polygraphy.tools.debug.subtool.base import BaseCheckerSubtool

trt = mod.lazy_import("tensorrt")
trt_util = mod.lazy_import("polygraphy.backend.trt.util")


class BaseMarker(object):
    def __init__(self, max_layers, direction, num_layers):
        self.max_layers = max_layers
        self.direction = direction
        self.num_layers = num_layers
        self.good = max_layers + 1  # Pretend marking all the layers gives us good accuracy.
        self.iteration = 0

    def select_layers(self):
        self.iteration += 1
        if self.direction == "forward":
            G_LOGGER.info("Selecting first {:} layer(s) to run in higher precision".format(self.num_layers))
            return range(0, self.num_layers)
        else:
            G_LOGGER.info("Selecting last {:} layer(s) to run in higher precision".format(self.num_layers))
            return range(self.max_layers - self.num_layers, self.max_layers)

    def success_message(self):
        which_layers = "first" if self.direction == "forward" else "last"
        G_LOGGER.finish(
            "To achieve acceptable accuracy, try running the {:} {:} "
            "layer(s) in higher precision".format(which_layers, self.good)
        )


class BisectMarker(BaseMarker):
    def __init__(self, max_layers, direction) -> None:
        super().__init__(max_layers, direction, max_layers)
        self.bad = 0

    def select_layers(self, prev_success):
        if prev_success:
            self.good = self.num_layers
            # On successes, we want num_layers to go closer to self.bad
            round_func = math.floor
        else:
            self.bad = self.num_layers
            round_func = math.ceil

        self.num_layers = round_func((self.good + self.bad) / 2.0)
        return super().select_layers()

    def stop(self, index, success):
        # If good and bad are within 1 layer of each other,
        # then we already have the information we need.
        if abs(self.good - self.bad) <= 1:
            if self.good >= self.max_layers:
                G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            else:
                self.success_message()
            return True

        if index >= (self.max_layers - 1):
            G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            return True
        return False

    def remaining(self):
        return int(math.log2(self.max_layers) - self.iteration)


class LinearMarker(BaseMarker):
    def __init__(self, max_layers, direction) -> None:
        super().__init__(max_layers, direction, 0)

    def select_layers(self, prev_success):
        if prev_success:
            self.good = self.num_layers
        self.num_layers += 1
        return super().select_layers()

    def stop(self, index, success):
        if success:
            self.success_message()
            return True

        if index >= (self.max_layers - 1):
            G_LOGGER.error("Could not find a configuration that satisfied accuracy requirements.")
            return True
        return False

    def remaining(self):
        return self.max_layers - self.iteration


class Precision(BaseCheckerSubtool):
    """
    Iteratively mark layers to run in a higher precision to find a
    compromise between performance and quality.
    Each iteration will generate an engine called 'polygraphy_debug.engine' in the current directory.
    """

    def __init__(self):
        super().__init__("precision", strict_types_default=True, prefer_artifacts=False)

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
            choices=["fp32", "fp16"],
            default="fp32",
        )

    def setup(self, args, network):
        self.precision = {"fp32": trt.float32, "fp16": trt.float16}[args.precision]

        if self.precision == trt.float16 and not self.arg_groups[TrtConfigArgs].fp16:
            G_LOGGER.critical(
                "Cannot mark layers to run in fp16 if it is not enabled in the builder configuration.\n"
                "Please also specify `--fp16` as a command-line option"
            )

        if self.precision == trt.float16 and not self.arg_groups[TrtConfigArgs].int8:
            G_LOGGER.warning(
                "Using fp16 as the higher precision, but fp16 is also the lowest precision available. "
                "Did you mean to set --int8 as well?"
            )

        if not any(
            [
                self.arg_groups[TrtConfigArgs].tf32,
                self.arg_groups[TrtConfigArgs].fp16,
                self.arg_groups[TrtConfigArgs].int8,
            ]
        ):
            G_LOGGER.critical("Please enable at least one precision besides fp32 (e.g. --int8, --fp16, --tf32)")

        if self.arg_groups[ModelArgs].model_type == "engine":
            G_LOGGER.critical(
                "The precision tool cannot work with engines, as they cannot be modified. "
                "Please provide a different format, such as an ONNX or TensorFlow model."
            )

        G_LOGGER.start("Using {:} as higher precision".format(self.precision))

        if args.mode == "linear":
            self.layer_marker = LinearMarker(len(network), args.direction)
        elif args.mode == "bisect":
            self.layer_marker = BisectMarker(len(network), args.direction)

    def mark_layers(self, network, indices):
        EXCLUDE_LAYER_NAMES = ["CONSTANT"]
        EXCLUDE_LAYERS = [getattr(trt.LayerType, attr) for attr in EXCLUDE_LAYER_NAMES if hasattr(trt.LayerType, attr)]

        # First, reset, since changes from the previous call will persist.
        for layer in network:
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
                G_LOGGER.extra_verbose(
                    "Running layer in higher precision: {:}".format(trt_util.str_from_layer(layer, index))
                )
                layer.precision = self.precision
                marked_indices.add(index)

        G_LOGGER.verbose("Marking layer(s): {:} to run in {:} precision".format(marked_indices, self.precision))

    def process_network(self, network, prev_success):
        indices = list(self.layer_marker.select_layers(prev_success))
        self.mark_layers(network, indices)

    def stop(self, index, success):
        return self.layer_marker.stop(index, success)

    def remaining(self):
        return self.layer_marker.remaining()
