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
import argparse
import copy
import os

from polygraphy.common import TensorMetadata, constants
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc

# The functions in this file include flags to control the set of options that are generated.

def add_model_args(parser, model_required=False, inputs="--inputs"):
    model_args = parser.add_argument_group("Model", "Model Options")
    model_args.add_argument("model_file", help="Path to the model", nargs=None if model_required else '?')
    model_args.add_argument("--model-type", help="The type of the input model: {{'frozen': TensorFlow frozen graph, 'keras': Keras model, "
                            "'ckpt': TensorFlow checkpoint directory, 'onnx': ONNX model, 'engine': TensorRT engine, 'uff': UFF file [deprecated], "
                            "'caffe': Caffe prototxt [deprecated]}}", choices=["frozen", "keras", "ckpt", "onnx", "uff", "caffe", "engine"],
                            default=None)
    if inputs:
        model_args.add_argument(inputs, inputs.replace("inputs", "input") + "-shapes", help="Model input(s) and their shape(s). Format: {arg_name} <name>,<shape>. "
                                "For example: {arg_name} image:1,1x3x224x224 other_input,10".format(arg_name=inputs), nargs="+", default=None, dest="inputs")


def add_dataloader_args(parser):
    data_loader_args = parser.add_argument_group("Data Loader", "Options for modifying data used for inference")
    data_loader_args.add_argument("--seed", metavar="SEED", help="Seed to use for random inputs",
                                    type=int, default=None)
    data_loader_args.add_argument("--int-min", help="Minimum integer value for random integer inputs", type=int, default=None)
    data_loader_args.add_argument("--int-max", help="Maximum integer value for random integer inputs", type=int, default=None)
    data_loader_args.add_argument("--float-min", help="Minimum float value for random float inputs", type=float, default=None)
    data_loader_args.add_argument("--float-max", help="Maximum float value for random float inputs", type=float, default=None)


def add_comparator_args(parser, iters=True, accuracy=True, validate=True, read=True, write=True, fail_fast=True, subprocess=True, top_k=False):
    comparator_args = parser.add_argument_group("Comparator", "Options for changing result comparison behavior")
    if iters:
        comparator_args.add_argument("--warm-up", metavar="NUM", help="Number of warm-up runs before timing inference", type=int, default=None)
        comparator_args.add_argument("--iterations", metavar="NUM", help="Number of inference iterations", type=int, default=None)
    if accuracy:
        comparator_args.add_argument("--no-shape-check", help="Disable checking that output shapes match exactly", action="store_true", default=None)
        comparator_args.add_argument("--rtol", metavar="RTOL", help="Relative tolerance for output comparison. See "
                                     "https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html for details", type=float, default=None)
        comparator_args.add_argument("--atol", metavar="ATOL", help="Absolute tolerance for output comparison. See "
                                     "https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html for details", type=float, default=None)
    if validate:
        comparator_args.add_argument("--validate", help="Check outputs for NaNs", action="store_true", default=None)
    if read:
        comparator_args.add_argument("--load-results", help="Path(s) to load results from runners.", nargs="+", default=[])
    if write:
        comparator_args.add_argument("--save-results", help="Path to save results from runners.", default=None)
    if fail_fast:
        comparator_args.add_argument("--fail-fast", help="Fail fast (stop comparing after the first failure)", action="store_true", default=None)
    if subprocess:
        comparator_args.add_argument("--use-subprocess", help="Run runners in isolated subprocesses. Cannot be used with a debugger",
                                     action="store_true", default=None)
    if top_k:
        comparator_args.add_argument("--top-k", help="[EXPERIMENTAL] Apply Top-K (i.e. find indices of K largest values) to the outputs before comparing them.", type=int, default=None)
    return comparator_args


def add_runner_args(parser):
       # Appends to args.runners
    class StoreRunnerOrdered(argparse.Action):
         def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, "runners"):
                namespace.runners = []
            namespace.runners.append(option_string.lstrip("-").replace("-", "_"))

    runner_args = parser.add_argument_group("Runners", "Options for selecting runners. Zero or more runners may be specified")
    runner_args.add_argument("--trt", help="Run inference using TensorRT", action=StoreRunnerOrdered, nargs=0)
    runner_args.add_argument("--trt-legacy", help="Run inference using Legacy TensorRT Runner. Only supports networks using implicit batch mode",
                             action=StoreRunnerOrdered, nargs=0)
    runner_args.add_argument("--tf", help="Run inference using TensorFlow", action=StoreRunnerOrdered, nargs=0)
    runner_args.add_argument("--onnxrt", help="Run inference using ONNX Runtime", action=StoreRunnerOrdered, nargs=0)
    runner_args.add_argument("--onnxtf", help="Run inference using the ONNX-TensorFlow Backend", action=StoreRunnerOrdered, nargs=0)
    runner_args.add_argument("--cntk", help="[EXPERIMENTAL] Run inference on a CNTK model using CNTK", action=StoreRunnerOrdered, nargs=0)


def add_trt_args(parser, write=True, config=True, outputs=True, network_api=False):
    trt_args = parser.add_argument_group("TensorRT", "Options for TensorRT")
    if write:
        trt_args.add_argument("--save-engine", help="Path to save a TensorRT engine file", default=None)
    if config:
        trt_args.add_argument("--trt-min-shapes", action='append', help="The minimum shapes the optimization profile(s) will support. "
                              "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                              "Format: --trt-min-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])
        trt_args.add_argument("--trt-opt-shapes", action='append', help="The shapes for which the optimization profile(s) will be most performant. "
                              "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                              "Format: --trt-opt-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])
        trt_args.add_argument("--trt-max-shapes", action='append', help="The maximum shapes the optimization profile(s) will support. "
                              "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                              "Format: --trt-max-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])

        trt_args.add_argument("--tf32", help="Enable tf32 precision in TensorRT", action="store_true", default=None)
        trt_args.add_argument("--fp16", help="Enable fp16 precision in TensorRT", action="store_true", default=None)
        trt_args.add_argument("--int8", help="Enable int8 precision in TensorRT", action="store_true", default=None)
        trt_args.add_argument("--strict-types", help="Enable strict types in TensorRT, forcing it to choose tactics based on the "
                                                     "layer precision set, even if another precision is faster.", action="store_true", default=None)
        # Workspace uses float to enable scientific notation (e.g. 1e9)
        trt_args.add_argument("--workspace", metavar="BYTES", help="Memory in bytes to allocate for the TensorRT builder's workspace", type=float, default=None)
        trt_args.add_argument("--calibration-cache", help="Path to the calibration cache", default=None)
        trt_args.add_argument("--plugins", help="Path(s) of additional plugin libraries to load", nargs="+", default=None)
    trt_args.add_argument("--explicit-precision", help="Enable explicit precision mode", action="store_true", default=None)
    trt_args.add_argument("--ext", help="Enable parsing ONNX models with externally stored weights", action="store_true", default=None)
    if outputs:
        trt_args.add_argument("--trt-outputs", help="Name(s) of TensorRT output(s). "
                              "Using '--trt-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
        trt_args.add_argument("--trt-exclude-outputs", help="[EXPERIMENTAL] Name(s) of TensorRT output(s) to unmark as outputs. ",
                              nargs="+", default=None)
    if network_api:
        trt_args.add_argument("--network-api", help="[EXPERIMENTAL] Generated script will include placeholder code for defining a TensorRT Network using "
                              "the network API. Only valid if --gen/--gen-script is also enabled.", action="store_true", default=None)


def add_trt_legacy_args(parser):
    trt_legacy_args = parser.add_argument_group("TensorRT Legacy", "[DEPRECATED] Options for TensorRT Legacy. Reuses TensorRT options, but does not support int8 mode, or dynamic shapes")
    trt_legacy_args.add_argument("-p", "--preprocessor", help="The preprocessor to use for the UFF converter", default=None)
    trt_legacy_args.add_argument("--uff-order", help="The order of the input", default=None)
    trt_legacy_args.add_argument("--batch-size", metavar="SIZE", help="The batch size to use in TensorRT when it cannot be automatically determined", type=int, default=None)
    trt_legacy_args.add_argument("--model", help="Model file for Caffe models. The deploy file should be provided as the model_file positional argument", dest="caffe_model")
    trt_legacy_args.add_argument("--save-uff", help="Save intermediate UFF files", action="store_true", default=None)


def add_tf_args(parser, tftrt=True, artifacts=True, runtime=True, outputs=True):
    tf_args = parser.add_argument_group("TensorFlow", "Options for TensorFlow")
    tf_args.add_argument("--ckpt", help="[EXPERIMENTAL] Name of the checkpoint to load. Required if the `checkpoint` file is missing. Should not include file extension "
                         "(e.g. to load `model.meta` use `--ckpt=model`)", default=None)
    if outputs:
        tf_args.add_argument("--tf-outputs", help="Name(s) of TensorFlow output(s). "
                             "Using '--tf-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
    if artifacts:
        tf_args.add_argument("--save-pb", help="Path to save the TensorFlow frozen graphdef", default=None)
        tf_args.add_argument("--save-tensorboard", help="[EXPERIMENTAL] Path to save a TensorBoard visualization", default=None)
        tf_args.add_argument("--save-timeline", help="[EXPERIMENTAL] Directory to save timeline JSON files for profiling inference (view at chrome://tracing)", default=None)
    if runtime:
        tf_args.add_argument("--gpu-memory-fraction", help="Maximum percentage of GPU memory TensorFlow can allocate per process", type=float, default=None)
        tf_args.add_argument("--allow-growth", help="Allow GPU memory allocated by TensorFlow to grow", action="store_true", default=None)
        tf_args.add_argument("--xla", help="[EXPERIMENTAL] Attempt to run graph with xla", action="store_true", default=None)
    tf_args.add_argument("--freeze-graph", help="[EXPERIMENTAL] Attempt to freeze the graph", action="store_true", default=None)
    if tftrt:
        tftrt_args = parser.add_argument_group("TensorFlow-TensorRT", "[UNTESTED] Options for TensorFlow-TensorRT Integration")
        tftrt_args.add_argument("--tftrt", help="[UNTESTED] Enable TF-TRT integration", action="store_true", default=None)
        tftrt_args.add_argument("--minimum-segment-size", help="Minimum length of a segment to convert to TensorRT", type=int, default=None)
        tftrt_args.add_argument("--dynamic-op", help="Enable dynamic mode (defers engine build until runtime)", action="store_true", default=None)


def add_onnx_args(parser, write=True, outputs=True, shape_inference_default=None):
    onnx_args = parser.add_argument_group("ONNX Options", "Options for ONNX")
    if write:
        onnx_args.add_argument("--save-onnx", help="Path to save the ONNX model", default=None)

    if shape_inference_default:
        onnx_args.add_argument("--no-shape-inference", help="Disable ONNX shape inference when loading the model", action="store_true", default=None)
    else:
        onnx_args.add_argument("--shape-inference", help="Enable ONNX shape inference when loading the model", action="store_true", default=None)

    if outputs:
        onnx_args.add_argument("--onnx-outputs", help="Name(s) of ONNX output(s). "
                               "Using '--onnx-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
        onnx_args.add_argument("--onnx-exclude-outputs", help="[EXPERIMENTAL] Name(s) of ONNX output(s) to unmark as outputs.", nargs="+", default=None)


def add_tf_onnx_args(parser):
    tf_onnx_args = parser.add_argument_group("TensorFlow-ONNX Options", "Options for TensorFlow-ONNX conversion")
    tf_onnx_args.add_argument("--opset", help="Opset to use when converting to ONNX", default=None, type=int)
    tf_onnx_args.add_argument("--no-const-folding", help="Do not fold constants in the TensorFlow graph prior to conversion", action="store_true", default=None)


def add_logger_args(parser):
    logging_args = parser.add_argument_group("Logging", "Options for logging and debug output")
    logging_args.add_argument("-v", "--verbose", help="Increase logging verbosity. Specify multiple times for higher verbosity", action="count", default=0)
    logging_args.add_argument("--silent", help="Disable all output", action="store_true", default=None)
    logging_args.add_argument("--log-format", help="Format for log messages: {{'timestamp': Include timestamp, 'line-info': Include file and line number, "
                              "'no-colors': Disable colors}}", choices=["timestamp", "line-info", "no-colors"], nargs="+", default=[])


def get(args, attr):
    """
    Gets a command-line argument if it exists, otherwise returns None.

    Args:
        args: The command-line arguments.
        attr (str): The name of the command-line argument.
    """
    if hasattr(args, attr):
        return getattr(args, attr)
    return None


def parse_meta(meta_args, includes_shape=True, includes_dtype=True):
    """
    Parses a list of tensor metadata arguments of the form "<name>,<shape>,<dtype>"
    `shape` and `dtype` are optional, but `dtype` must always come after `shape` if they are both enabled.

    Args:
        meta_args (List[str]): A list of tensor metadata arguments from the command-line.
        includes_shape (bool): Whether the arguments include shape information.
        includes_dtype (bool): Whether the arguments include dtype information.

    Returns:
        TensorMetadata: The parsed tensor metadata.
    """
    SEP = ","
    SHAPE_SEP = "x"
    meta = TensorMetadata()
    for orig_tensor_meta_arg in meta_args:
        tensor_meta_arg = orig_tensor_meta_arg

        def pop_meta(name):
            nonlocal tensor_meta_arg
            tensor_meta_arg, _, val = tensor_meta_arg.rpartition(SEP)
            if not tensor_meta_arg:
                G_LOGGER.critical("Could not parse {:} from argument: {:}. Is it separated by a comma "
                                    "(,) from the tensor name?".format(name, orig_tensor_meta_arg))
            if val.lower() == "auto":
                val = None
            return val


        def parse_dtype(dtype):
            if dtype is not None:
                if dtype not in misc.NP_TYPE_FROM_STR:
                    G_LOGGER.critical("Could not understand data type: {:}. Please use one of: {:} or `auto`"
                            .format(dtype, list(misc.NP_TYPE_FROM_STR.keys())))
                dtype = misc.NP_TYPE_FROM_STR[dtype]
            return dtype


        def parse_shape(shape):
            if shape is not None:
                def parse_shape_dim(buf):
                    try:
                        buf = int(buf)
                    except:
                        pass
                    return buf


                parsed_shape = []
                # Allow for quoted strings in shape dimensions
                in_quotes = False
                buf = ""
                for char in shape.lower():
                    if char in ["\"", "'"]:
                        in_quotes = not in_quotes
                    elif not in_quotes and char == SHAPE_SEP:
                        parsed_shape.append(parse_shape_dim(buf))
                        buf = ""
                    else:
                        buf += char
                # For the last dimension
                parsed_shape.append(parse_shape_dim(buf))
                shape = tuple(parsed_shape)
            return shape


        name = None
        dtype = None
        shape = None

        if includes_dtype:
            dtype = parse_dtype(pop_meta("data type"))

        if includes_shape:
            shape = parse_shape(pop_meta("shape"))

        name = tensor_meta_arg
        meta.add(name, dtype, shape)
    return meta


# shapes is a TensorMetadata describing the runtime input shapes.
# Returns (List[Tuple[OrderedDict[str, List[int]]])
def parse_profile_shapes(shapes, min_args, opt_args, max_args):
    def get_shapes(lst, idx):
        default_shapes = copy.copy(shapes)
        if idx < len(lst):
            default_shapes.update(parse_meta(lst[idx], includes_dtype=False))
        # Don't care about dtype, and need to override dynamic dimensions
        default_shapes = {name: misc.override_dynamic_shape(shape) for name, (_, shape) in default_shapes.items()}

        for name, (_, shape) in shapes.items():
            if tuple(default_shapes[name]) != tuple(shape):
                G_LOGGER.warning("Input tensor: {:} | For TensorRT profile, overriding shape: {:} to: {:}".format(name, shape, default_shapes[name]), mode=LogMode.ONCE)

        return default_shapes


    num_profiles = max(len(min_args), len(opt_args), len(max_args))

    # For cases where input shapes are provided, we have to generate a profile
    if not num_profiles and shapes:
        num_profiles = 1


    profiles = []
    for idx in range(num_profiles):
        min_shapes = get_shapes(min_args, idx)
        opt_shapes = get_shapes(opt_args, idx)
        max_shapes = get_shapes(max_args, idx)
        if sorted(min_shapes.keys()) != sorted(opt_shapes.keys()):
            G_LOGGER.critical("Mismatch in input names between minimum shapes ({:}) and optimum shapes "
                            "({:})".format(list(min_shapes.keys()), list(opt_shapes.keys())))
        elif sorted(opt_shapes.keys()) != sorted(max_shapes.keys()):
            G_LOGGER.critical("Mismatch in input names between optimum shapes ({:}) and maximum shapes "
                            "({:})".format(list(opt_shapes.keys()), list(max_shapes.keys())))

        profiles.append((min_shapes, opt_shapes, max_shapes))
    return profiles


def setup_logger(args):
    if args.verbose >= 4:
        G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE
    elif args.verbose == 3:
        G_LOGGER.severity = G_LOGGER.SUPER_VERBOSE
    elif args.verbose == 2:
        G_LOGGER.severity = G_LOGGER.EXTRA_VERBOSE
    elif args.verbose == 1:
        G_LOGGER.severity = G_LOGGER.VERBOSE

    if args.silent:
        G_LOGGER.severity = G_LOGGER.CRITICAL

    for fmt in args.log_format:
        if fmt == "no-colors":
            G_LOGGER.colors = False
        elif fmt == "timestamp":
            G_LOGGER.timestamp = True
        elif fmt == "line-info":
            G_LOGGER.line_info = True


def determine_model_type(args):
    if get(args, "model_type") is not None:
        return args.model_type.lower()

    if get(args, "model_file") is None:
        return None

    def use_ext(ext_mapping):
        file_ext = os.path.splitext(args.model_file)[-1]
        if file_ext in ext_mapping:
            return ext_mapping[file_ext]

    if get(args, "ckpt") or os.path.isdir(args.model_file):
        return "ckpt"
    elif "tf" in args.runners or "trt_legacy" in args.runners:
        if args.caffe_model:
            return "caffe"
        ext_mapping = {".hdf5": "keras", ".uff": "uff", ".prototxt": "caffe", ".onnx": "onnx", ".engine": "engine", ".plan": "engine"}
        return use_ext(ext_mapping) or "frozen"
    else:
        # When no framework is provided, some extensions can be ambiguous
        ext_mapping = {".hdf5": "keras", ".graphdef": "frozen", ".onnx": "onnx", ".uff": "uff", ".engine": "engine", ".plan": "engine"}
        model_type = use_ext(ext_mapping)
        if model_type:
            return model_type

    G_LOGGER.critical("Could not automatically determine model type for: {:}\n"
                      "Please explicitly specify the type with the --model-type option".format(
                          args.model_file))


def setup(args, unknown):
    """
    Prepares argument values for use.
    """
    def exist(names):
        return all([hasattr(args, name) for name in names])

    def process_output_arg(name):
        arg = get(args, name)
        if arg is not None and len(arg) == 2 and arg == ["mark", "all"]:
            arg = constants.MARK_ALL
        setattr(args, name, arg)


    if unknown:
        G_LOGGER.critical("Unrecognized Options: {:}".format(unknown))

    setup_logger(args)

    if not exist(["runners"]): # For when no runners are specified
        args.runners = []

    if get(args, "network_api") and not get(args, "gen_script"):
        G_LOGGER.critical("Cannot use the --network-api option if --gen/--gen-script is not being used.")
    elif get(args, "network_api") and "trt" not in args.runners:
        args.runners.append("trt")

    if get(args, "model_file"):
        G_LOGGER.verbose("Model: {:}".format(args.model_file))
        if not os.path.exists(args.model_file):
            G_LOGGER.warning("Model path does not exist: {:}".format(args.model_file))
        args.model_file = os.path.abspath(args.model_file)
    elif args.runners and get(args, "network_api") is None:
        G_LOGGER.critical("One or more runners was specified, but no model file was provided. Make sure you've specified the model path, "
                          "and also that it's not being consumed as an argument for another parameter")

    args.model_type = determine_model_type(args)

    if get(args, "inputs"):
        args.inputs = parse_meta(args.inputs, includes_dtype=False) # TensorMetadata
    else:
        args.inputs = TensorMetadata()

    if exist(["trt_min_shapes", "trt_opt_shapes", "trt_max_shapes"]):
        args.profiles = parse_profile_shapes(args.inputs, args.trt_min_shapes, args.trt_opt_shapes, args.trt_max_shapes)
    elif args.inputs:
        args.profiles = parse_profile_shapes(args.inputs, [], [], [])

    if exist(["workspace"]):
        args.workspace = int(args.workspace) if args.workspace is not None else args.workspace

    process_output_arg("tf_outputs")
    process_output_arg("trt_outputs")
    process_output_arg("onnx_outputs")

    return args
