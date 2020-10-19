#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.tools.util import misc as tool_util, args as args_util
from polygraphy.tools.base import Tool
from polygraphy.util import misc
from polygraphy.logger import G_LOGGER

from collections import OrderedDict
import math


################################# SUBTOOLS #################################


# Subtool base class for accuracy checkers
class STCheckerBase(Tool):
    def add_parser_args(self, parser, mode=True):
        parser.add_argument("--golden", help="Golden outputs for accuracy comparison", required=True)
        parser.add_argument("-s", "--show-output", help="Show logging output when checking network accuracy", action="store_true")
        if mode:
            parser.add_argument("--mode", help="How layers are marked to run in higher precision. "
                                        "'forward' will start marking layers from network inputs, and 'reverse' will start "
                                        "from the network outputs", choices=["forward", "reverse"], default="reverse")
        parser.add_argument("-p", "--precision", help="Precision to use when marking layers to run in higher precision",
                            choices=["float32", "float16"], default="float32")
        args_util.add_model_args(parser, model_required=True)
        args_util.add_comparator_args(parser, iters=False, read=False, write=False, subprocess=False, fail_fast=False)
        args_util.add_dataloader_args(parser)
        args_util.add_trt_args(parser)
        args_util.add_tf_args(parser, tftrt=False, artifacts=False, runtime=False)
        args_util.add_onnx_args(parser, write=False)
        args_util.add_tf_onnx_args(parser)


    def __call__(self, args):
        import tensorrt as trt

        if not args.calibration_cache:
            G_LOGGER.warning("Not using a calibration cache. Using a calibration cache may significantly speed up the search process")

        self.precision = {"float32": trt.float32, "float16": trt.float16}[args.precision]
        if self.precision == trt.float16 and not args.fp16:
            args.fp16 = True
        if self.precision == trt.float16 and not args.int8:
            G_LOGGER.warning("Using float16 as the higher precision, but float16 is also the lowest precision available. Did you mean to set --int8 as well?")

        if not any([args.tf32, args.fp16, args.int8]):
            G_LOGGER.critical("Please enable at least one precision besides float32 (e.g. --int8, --fp16)")

        if args.model_type == "engine":
            G_LOGGER.critical("The precision tool cannot work with engines, as they cannot be modified. "
                              "Please provide a different format, such as an ONNX or TensorFlow model.")

        self.args = args

        self.golden = OrderedDict()
        self.golden.update(misc.pickle_load(args.golden))

        self.builder, self.network, self.parser = tool_util.get_trt_network_loader(args)()
        with self.builder, self.network, self.parser:
            indices = self.find()

        if indices is not None:
            G_LOGGER.info("To achieve acceptable accuracy, try running layers: {:} in {:} precision".format(
                                indices, self.precision))
        else:
            G_LOGGER.critical("Could not find a configuration that resulted in acceptable accuracy")


    # Determines layer indices based on direction and number of layers to mark
    def layer_indices(self, num_layers):
        if self.args.mode == "forward":
            start = 0
            end = num_layers
        elif self.args.mode == "reverse":
            start = self.network.num_layers - num_layers
            end = self.network.num_layers
        return range(start, end)


    def mark_layers(self, indices):
        def layer_to_str(layer):
            outputs = [layer.get_output(i).name for i in range(layer.num_outputs)]
            return "{:}: {:}".format(layer.name, outputs)

        # First, reset, since changes from the previous call will persist.
        for layer in self.network:
            layer.reset_precision()

        for index in indices:
            layer = self.network.get_layer(index)
            G_LOGGER.verbose("Running layer in higher precision: {:}".format(layer_to_str(layer)))
            layer.precision = self.precision
        G_LOGGER.info("Will run layer(s): {:} in {:} precision".format(indices, self.precision))


    def check_network(self, suffix):
        """
        Checks whether the provided network is accurate compared to golden values.

        Returns:
            OrderedDict[str, OutputCompareResult]:
                    A mapping of output names to an object describing whether they matched, and what the
                    required tolerances were.
        """
        from polygraphy.comparator import Comparator, CompareFunc, DataLoader
        from polygraphy.backend.trt import EngineFromNetwork, TrtRunner, ModifyNetwork, SaveEngine

        with G_LOGGER.verbosity(severity=G_LOGGER.severity if self.args.show_output else G_LOGGER.CRITICAL):
            data_loader = tool_util.get_data_loader(self.args)

            self.args.strict_types = True # HACK: Override strict types so things actually run in the right precision.
            config = tool_util.get_trt_config_loader(self.args, data_loader)(self.builder, self.network)

            suffix = "-{:}-{:}".format(suffix, self.precision)
            engine_path = misc.insert_suffix(self.args.save_engine, suffix)

            self.builder, self.network, self.parser = ModifyNetwork((self.builder, self.network, self.parser),
                                                                    outputs=self.args.trt_outputs)()

            engine_loader = SaveEngine(EngineFromNetwork((self.builder, self.network, self.parser), config),
                                       path=engine_path)

            runners = [TrtRunner(engine_loader)]

            results = Comparator.run(runners, data_loader=data_loader)
            if self.args.validate:
                Comparator.validate(results)
            results.update(self.golden)


            compare_func = CompareFunc.basic_compare_func(atol=self.args.atol, rtol=self.args.rtol, check_shapes=not self.args.no_shape_check)
            accuracy_result = Comparator.compare_accuracy(results, compare_func=compare_func)

        tolerances = list(accuracy_result.values())[0][0] # First iteration of first runner pair
        for name, req_tol in tolerances.items():
            if bool(req_tol):
                G_LOGGER.success("PASSED | Output: {:} | Required Tolerances: {:}".format(name, req_tol))
            else:
                G_LOGGER.error("FAILED | Output: {:} | Required Tolerances: {:}".format(name, req_tol))
        return accuracy_result


class STWorstFirst(STCheckerBase):
    """
    [EXPERIMENTAL] Progressively mark the worst layers (those that introduce the largest error)
    to run in a higher precision. Assumes that the layerwise outputs
    are topologically sorted.
    """
    def __init__(self):
        self.name = "worst-first"

    def add_parser_args(self, parser):
        super().add_parser_args(parser, mode=False)
        parser.add_argument("--top", help="Number of additional layers to mark each successive iteration", type=int, default=5)


    def find(self):
        def run(indices):
            self.mark_layers(indices)
            return self.check_network("-".join(map(str, indices)))


        # Finds num worst indices in acc_results
        def find_worst(num, acc_results):
            acc_mapping = list(acc_results.values())[0][0] # First iteration of first runner-pair.

            # Compute for each layer: atol / prev_atol, to determine which layers contribute the greatest error.
            # It is not enough to simply find the max(atol), because that doesn't account for error introduced
            # by previous layers.
            items = list(acc_mapping.items())
            ratios = []
            for (_, prev_tols), (outname, cur_tols) in zip(items[:-1], items[1:]):
                ratio = cur_tols.required_atol / prev_tols.required_atol
                ratios.append((ratio, outname))

            # Mark more layers on each iteration
            ratios = sorted(ratios, reverse=True)[:num]
            G_LOGGER.verbose("Found worst {:} layers (Format: (error ratio, tensor name)): {:}".format(num, ratios))
            return [output_mapping[outname] for (ratio, outname) in ratios]


        if not args_util.get(self.args, "trt_outputs"):
            G_LOGGER.critical("worst-first requires all outputs to be marked as network outputs mode to determine where errors are being introduced. "
                              "Please enable --trt-outputs mark all, and ensure that your golden outputs also include layer-wise results")

        output_mapping = {} # Maps output tensor names to producer layer indices
        for layer_index, layer in enumerate(self.network):
            for out_index in range(layer.num_outputs):
                output_mapping[layer.get_output(out_index).name] = layer_index

        indices = []
        acc_results = run(indices)
        max_outputs = len(list(acc_results.values())[0][0]) - 1

        iter_num = 0
        # indices will be at most one less than the number of layers, since we're comparing layers against subsequent ones.
        while not bool(acc_results) and len(indices) < max_outputs:
            iter_num += 1
            indices = find_worst(self.args.top * iter_num, acc_results)
            acc_results = run(indices)

        if bool(acc_results):
            return indices


class STLinear(STCheckerBase):
    """
    Use a linear search to determine how many layers need to be run in
    higher precision to achieve the desired accuracy.
    """
    def __init__(self):
        self.name = "linear"


    def find(self):
        which_layers = {"forward": "first", "reverse": "last"}[self.args.mode]

        for index in range(self.network.num_layers + 1):
            indices = self.layer_indices(index)
            self.mark_layers(indices)
            success = self.check_network("{:}-{:}".format(which_layers, index))
            if success:
                return indices


class STBisect(STCheckerBase):
    """
    Use a binary search to determine how many layers need to be run in
    higher precision to achieve the desired accuracy.
    """
    def __init__(self):
        self.name = "bisect"


    def find(self):
        which_layers = {"forward": "first", "reverse": "last"}[self.args.mode]

        num_layers = 0
        # Keep track of what works and what doesn't
        known_good = self.network.num_layers + 1
        known_bad = 0

        while known_good != known_bad and num_layers != known_good:
            with G_LOGGER.indent():
                G_LOGGER.info("Last known good: {which_layers} {known_good} layer(s) in {precision} precision.\n"
                              "Last known bad: {which_layers} {known_bad} layer(s) in {precision} precision".format(
                                    which_layers=which_layers, known_good=min(known_good, self.network.num_layers),
                                    precision=self.precision, known_bad=known_bad))

            indices = self.layer_indices(num_layers)
            self.mark_layers(indices)
            success = self.check_network("{:}-{:}".format(which_layers, num_layers))
            if success:
                # Try something between
                known_good = num_layers
            else:
                known_bad = num_layers
            # Try something in between the known good value, and the known bad value.
            num_layers = math.ceil((known_bad + known_good) / 2.0)

        if known_good <= self.network.num_layers:
            return indices


################################# MAIN TOOL #################################

class Precision(Tool):
    """
    [EXPERIMENTAL] Debug low precision models by suggesting which layers should be run in
    higher precisions to preserve accuracy.
    """
    def __init__(self):
        self.name = "precision"


    def add_parser_args(self, parser):
        subparsers = parser.add_subparsers(title="Precision Modes", dest="mode")
        subparsers.required = True

        PRECISION_SUBTOOLS = [
            STBisect(),
            STLinear(),
            STWorstFirst(),
        ]

        for subtool in PRECISION_SUBTOOLS:
            subtool.setup_parser(subparsers)


    def __call__(self, args):
        pass
