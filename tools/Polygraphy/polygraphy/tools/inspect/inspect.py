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
from polygraphy.tools.util import args as args_util, misc as tool_util
from polygraphy.common import constants, TensorMetadata
from polygraphy.tools.base import Tool
from polygraphy.util import misc
from polygraphy.logger import G_LOGGER


def inspect_trt(args):
    from polygraphy.backend.trt import util as trt_util

    if args.model_type == "engine":
        if args.mode != "none":
            G_LOGGER.warning("Displaying layer information for TensorRT engines is not currently supported")

        with tool_util.get_trt_serialized_engine_loader(args)() as engine:
            engine_str = trt_util.str_from_engine(engine)
            G_LOGGER.info("==== TensorRT Engine ====\n{:}".format(engine_str))
    else:
        builder, network, parser = tool_util.get_trt_network_loader(args)()
        with builder, network, parser:
            network_str = trt_util.str_from_network(network, mode=args.mode)
            G_LOGGER.info("==== TensorRT Network ====\n{:}".format(network_str))


def inspect_onnx(args):
    from polygraphy.backend.onnx import util as onnx_util

    onnx_model = tool_util.get_onnx_model_loader(args)()
    model_str = onnx_util.str_from_onnx(onnx_model, mode=args.mode)
    G_LOGGER.info("==== ONNX Model ====\n{:}".format(model_str))


def inspect_tf(args):
    from polygraphy.backend.tf import util as tf_util

    tf_graph, _ = tool_util.get_tf_model_loader(args)()
    graph_str = tf_util.str_from_graph(tf_graph, mode=args.mode)
    G_LOGGER.info("==== TensorFlow Graph ====\n{:}".format(graph_str))


################################# SUBTOOLS #################################

class STModel(Tool):
    """
    Display information about a model, including inputs and outputs, as well as layers and their attributes.
    """
    def __init__(self):
        self.name = "model"


    def add_parser_args(self, parser):
        parser.add_argument("--convert-to", "--display-as", help="Convert the model to the specified format before displaying",
                            choices=["onnx", "trt"], dest="display_as")
        parser.add_argument("--mode", "--layer-info", help="Display layers: {{"
                            "'none': Display no layer information, "
                            "'basic': Display layer inputs and outputs, "
                            "'attrs': Display layer inputs, outputs and attributes, "
                            "'full': Display layer inputs, outputs, attributes, and weights"
                            "}}",
                            choices=["none", "basic", "attrs", "full"], dest="mode", default="none")
        args_util.add_model_args(parser, model_required=True, inputs=False)
        args_util.add_trt_args(parser, write=False, config=False, outputs=False)
        args_util.add_tf_args(parser, tftrt=False, artifacts=False, runtime=False, outputs=False)
        args_util.add_onnx_args(parser, outputs=False)
        args_util.add_tf_onnx_args(parser)


    def __call__(self, args):
        func = None

        if args.model_type in ["frozen", "keras", "ckpt"]:
            func = inspect_tf

        if args.model_type == "onnx" or args.display_as == "onnx":
            func = inspect_onnx

        if args.model_type == "engine" or args.display_as == "trt":
            func = inspect_trt

        if func is None:
            G_LOGGER.critical("Could not determine how to display this model. Maybe you need to specify --display-as?")
        func(args)


class STResults(Tool):
    """
    Display information about results saved from Polygraphy's Comparator.run()
    (for example, outputs saved by `--save-results` from `polygraphy run`).
    """
    def __init__(self):
        self.name = "results"


    def add_parser_args(self, parser):
        parser.add_argument("results", help="Path to a file containing Comparator.run() results from Polygraphy")
        parser.add_argument("--all", help="Show information on all iterations present in the results instead of just the first",
                            action="store_true")
        parser.add_argument("-s", "--show-values", help="Show values of output tensors instead of just metadata", action="store_true")


    def __call__(self, args):
        run_results = misc.pickle_load(args.results)

        def meta_from_iter_result(iter_result):
            meta = TensorMetadata()
            for name, arr in iter_result.items():
                meta.add(name, dtype=arr.dtype, shape=arr.shape)
            return meta


        results_str = ""
        results_str += "==== Run Results ({:} runners) ====\n\n".format(len(run_results))

        for runner_name, iters in run_results.items():
            results_str += "---- Runner: {:} ({:} iterations) ----\n".format(runner_name, len(iters))

            for index, iter_result in enumerate(iters):
                if args.show_values:
                    for name, arr in iter_result.items():
                        results_str += "{:} [dtype={:}, shape={:}]\n{:}\n\n".format(name, arr.dtype, arr.shape, misc.indent_block(str(arr)))
                else:
                    iter_meta = meta_from_iter_result(iter_result)
                    if len(iters) > 1 and args.all:
                        results_str += misc.indent_block("Iteration: {:} | ".format(index))
                    results_str += "{:}\n".format(iter_meta)

                if not args.all:
                    break
            results_str += "\n"
        results_str = misc.indent_block(results_str, level=0)
        G_LOGGER.info(results_str)


################################# MAIN TOOL #################################

class Inspect(Tool):
    """
    Display information about supported files.
    """
    def __init__(self):
        self.name = "inspect"


    def add_parser_args(self, parser):
        subparsers = parser.add_subparsers(title="Inspection Subtools", dest="subtool")
        subparsers.required = True

        SUBTOOLS = [
            STModel(),
            STResults(),
        ]

        for subtool in SUBTOOLS:
            subtool.setup_parser(subparsers)


    def __call__(self, args):
        pass
