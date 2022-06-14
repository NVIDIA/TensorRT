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
import contextlib

from polygraphy import config, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER, LogMode

trt = mod.lazy_import("tensorrt")
np = mod.lazy_import("numpy")


TRT_LOGGER = None


@mod.export()
def get_trt_logger():
    """
    Get the global TensorRT logger created by Polygraphy.

    Returns:
        trt.Logger: The TensorRT logger.
    """
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger()
    return TRT_LOGGER


def fail_unavailable(what):
    G_LOGGER.backtrace()
    G_LOGGER.critical("{:} is not available on TensorRT version {:}.".format(what, trt.__version__))


def check_onnx_parser_errors(parser, success):
    if parser.num_errors > 0:
        for index in range(parser.num_errors):
            G_LOGGER.error(parser.get_error(index))
        G_LOGGER.critical("Could not parse ONNX correctly")

    if not success:
        G_LOGGER.critical("Failed to parse ONNX model. Does the model file exist and contain a valid ONNX model?")


def get_layer_class_mapping():
    layer_class_mapping = {}

    def try_add(layer_type, layer_cls):
        try:
            layer_type = getattr(trt.LayerType, layer_type)
            layer_cls = getattr(trt, layer_cls)
        except AttributeError:
            if config.INTERNAL_CORRECTNESS_CHECKS:
                G_LOGGER.warning(
                    "Could not find layer type: {:} or layer class: {:}".format(layer_type, layer_cls)
                )
        else:
            layer_class_mapping[layer_type] = layer_cls

    try_add("CONVOLUTION", "IConvolutionLayer")
    try_add("FULLY_CONNECTED", "IFullyConnectedLayer")
    try_add("ACTIVATION", "IActivationLayer")
    try_add("POOLING", "IPoolingLayer")
    try_add("LRN", "ILRNLayer")
    try_add("SCALE", "IScaleLayer")
    try_add("SOFTMAX", "ISoftMaxLayer")
    try_add("DECONVOLUTION", "IDeconvolutionLayer")
    try_add("CONCATENATION", "IConcatenationLayer")
    try_add("ELEMENTWISE", "IElementWiseLayer")
    try_add("PLUGIN", "IPluginLayer")
    try_add("UNARY", "IUnaryLayer")
    try_add("PADDING", "IPaddingLayer")
    try_add("SHUFFLE", "IShuffleLayer")
    try_add("REDUCE", "IReduceLayer")
    try_add("TOPK", "ITopKLayer")
    try_add("GATHER", "IGatherLayer")
    try_add("MATRIX_MULTIPLY", "IMatrixMultiplyLayer")
    try_add("RAGGED_SOFTMAX", "IRaggedSoftMaxLayer")
    try_add("CONSTANT", "IConstantLayer")
    try_add("RNN", "IRNNLayer")
    try_add("RNN_V2", "IRNNv2Layer")
    try_add("IDENTITY", "IIdentityLayer")
    try_add("PLUGIN_V2", "IPluginV2Layer")
    try_add("SLICE", "ISliceLayer")
    try_add("SHAPE", "IShapeLayer")
    try_add("PARAMETRIC_RELU", "IParametricReLULayer")
    try_add("RESIZE", "IResizeLayer")
    try_add("TRIP_LIMIT", "ITripLimitLayer")
    try_add("RECURRENCE", "IRecurrenceLayer")
    try_add("ITERATOR", "IIteratorLayer")
    try_add("LOOP_OUTPUT", "ILoopOutputLayer")
    try_add("SELECT", "ISelectLayer")
    try_add("FILL", "IFillLayer")
    try_add("QUANTIZE", "IQuantizeLayer")
    try_add("DEQUANTIZE", "IDequantizeLayer")
    try_add("CONDITION", "IConditionLayer")
    try_add("CONDITIONAL_INPUT", "IIfConditionalInputLayer")
    try_add("CONDITIONAL_OUTPUT", "IIfConditionalOutputLayer")
    try_add("ASSERTION", "IAssertionLayer")
    try_add("SCATTER", "IScatterLayer")
    try_add("EINSUM", "IEinsumLayer")

    return layer_class_mapping


def np_dtype_from_trt(trt_dtype):
    _ = mod.has_mod(np)  # Force numpy to be imported
    return np.dtype(trt.nptype(trt_dtype))


def get_network_input_metadata(network):
    inputs = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.add(name=tensor.name, dtype=np_dtype_from_trt(tensor.dtype), shape=tensor.shape)
    return inputs


def get_network_output_metadata(network):
    outputs = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.add(name=tensor.name, dtype=np_dtype_from_trt(tensor.dtype), shape=tensor.shape)
    return outputs


def get_layer_input_metadata(layer):
    meta = TensorMetadata()
    for i in range(layer.num_inputs):
        inp = layer.get_input(i)
        if inp:
            meta.add(inp.name, np_dtype_from_trt(inp.dtype), inp.shape)
    return meta


def get_layer_output_metadata(layer):
    meta = TensorMetadata()
    for i in range(layer.num_outputs):
        outp = layer.get_output(i)
        if outp:
            meta.add(outp.name, np_dtype_from_trt(outp.dtype), outp.shape)
    return meta


def str_from_layer(layer, index):
    input_info = get_layer_input_metadata(layer)
    output_info = get_layer_output_metadata(layer)
    return util.str_from_layer("Layer", index, layer.name, layer.type, input_info, output_info)


def get_layer_attribute_names(layer):
    def is_special_attribute(attr):
        return attr.startswith("__") and attr.endswith("__")

    def is_valid_attribute(attr, layer):
        if (
            type(layer) == trt.IPoolingLayer
            or type(layer) == trt.IConvolutionLayer
            or type(layer) == trt.IDeconvolutionLayer
        ):
            if len(layer.get_input(0).shape) > 4:
                # 3D pooling uses padding_nd
                return attr not in ["padding", "stride", "window_size"]
        if type(layer) == trt.IResizeLayer:
            if layer.num_inputs > 1:
                return attr not in ["scales"]
        if type(layer) == trt.ISliceLayer:
            if layer.num_inputs > 1:
                return attr not in ["shape", "start", "stride"]
        return True

    return [
        attr
        for attr in dir(layer)
        if not is_special_attribute(attr) and not hasattr(trt.ILayer, attr) and is_valid_attribute(attr, layer)
    ]


def str_from_network(network, mode="full"):
    """
    Converts a TensorRT network to a human-readable representation

    Args:
        network (trt.INetworkDefinition): The network.
        mode (str): Controls what is displayed for each layer. Choices: ["none", "basic", "attrs", "full"]

    Returns:
        str
    """
    LAYER_TYPE_CLASS_MAPPING = get_layer_class_mapping()

    network_str = "Name: {:} | {:} Batch Network{:}\n".format(
        network.name,
        "Implicit"
        if hasattr(network, "has_implicit_batch_dimension") and network.has_implicit_batch_dimension
        else "Explicit",
        " with Explicit Precision "
        if hasattr(network, "has_explicit_precision") and network.has_explicit_precision
        else "",
    )
    network_str += "\n"

    input_metadata = get_network_input_metadata(network)
    network_str += "---- {:} Network Input(s) ----\n{:}\n\n".format(len(input_metadata), input_metadata)
    output_metadata = get_network_output_metadata(network)
    network_str += "---- {:} Network Output(s) ----\n{:}\n\n".format(len(output_metadata), output_metadata)
    network_str += "---- {:} Layer(s) ----\n".format(network.num_layers)
    if mode != "none":
        for index, layer in enumerate(network):
            if layer.type in LAYER_TYPE_CLASS_MAPPING:
                layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

            network_str += str_from_layer(layer, index)

            if mode in ["attrs", "full"]:
                # Exclude special attributes, as well as any attributes of the base layer class (those can be displayed above).
                attrs = get_layer_attribute_names(layer)
                if attrs:
                    network_str += util.indent_block("---- Attributes ----") + "\n"
                for attr in attrs:
                    with G_LOGGER.verbosity():
                        val = getattr(layer, attr)
                    if mode == "full" or not isinstance(val, np.ndarray):
                        attr_str = ""
                        if layer.name:
                            attr_str += "{:}.".format(layer.name)
                        network_str += util.indent_block("{:}{:} = {:}".format(attr_str, attr, val)) + "\n"
            network_str += "\n"

    return util.indent_block(network_str, level=0)


def _get_network_outputs(network):
    return [network.get_output(index).name for index in range(network.num_outputs)]


def check_outputs_not_found(not_found, available_outputs):
    if not_found:
        available_outputs = util.unique_list(available_outputs)
        G_LOGGER.critical(
            "The following outputs were not found: {:}.\n"
            "Note: Available tensors:\n\t{:}".format(not_found, "\n\t".join(available_outputs))
        )


def mark_outputs(network, outputs):
    """
    Mark the specified outputs as network outputs.

    Args:
        network (trt.INetworkDefinition): The network in which to mark outputs.
        outputs (Sequence[str]): The names of tensors to mark as outputs.
    """
    outputs = set(outputs)
    all_outputs = []
    for layer in network:
        for index in range(layer.num_outputs):
            tensor = layer.get_output(index)
            all_outputs.append(tensor.name)
            # Clear all old outputs
            if tensor.is_network_output:
                network.unmark_output(tensor)

            if tensor.name in outputs:
                if not tensor.is_network_output:
                    G_LOGGER.ultra_verbose("Marking {:} as an output".format(tensor.name))
                    network.mark_output(tensor)

    marked_outputs = set(_get_network_outputs(network))
    not_found = outputs - marked_outputs
    check_outputs_not_found(not_found, all_outputs)


def mark_layerwise(network):
    # Layers within loops cannot be marked as network outputs.
    LOOP_START_NAMES = ["TRIP_LIMIT", "ITERATOR", "RECURRENCE"]
    LOOP_END_NAMES = ["LOOP_OUTPUT"]
    LOOP_START_LAYERS = [getattr(trt.LayerType, attr) for attr in LOOP_START_NAMES if hasattr(trt.LayerType, attr)]
    LOOP_END_LAYERS = [getattr(trt.LayerType, attr) for attr in LOOP_END_NAMES if hasattr(trt.LayerType, attr)]
    EXCLUDE_LAYERS = [trt.LayerType.SHAPE, trt.LayerType.CONSTANT]
    outputs = []
    in_loop = False
    for layer in network:
        if layer.type in LOOP_START_LAYERS:
            G_LOGGER.warning(
                "Loop detected. Please ensure the network is topologically sorted so that layers within "
                "the loop body are not marked as network outputs in layerwise mode",
                mode=LogMode.ONCE,
            )
            in_loop = True
        elif layer.type in LOOP_END_LAYERS:
            in_loop = False

        should_mark_layer = not in_loop and layer.type not in EXCLUDE_LAYERS
        if should_mark_layer:
            for index in range(layer.num_outputs):
                tensor = layer.get_output(index)
                outputs.append(tensor.name)

    G_LOGGER.verbose("Marking {:} tensors as outputs".format(len(outputs)))
    mark_outputs(network, outputs)


def unmark_outputs(network, outputs):
    outputs = set(outputs)

    unmarked_outputs = set()
    for layer in network:
        for index in range(layer.num_outputs):
            tensor = layer.get_output(index)
            if tensor.is_network_output and tensor.name in outputs:
                network.unmark_output(tensor)
                unmarked_outputs.add(tensor.name)
    not_found = outputs - unmarked_outputs
    check_outputs_not_found(not_found, _get_network_outputs(network))


def str_from_config(config):
    config_str = "{:20} | {:} bytes ({:.2f} MiB)\n".format(
        "Workspace", config.max_workspace_size, config.max_workspace_size / (1024.0 ** 2)
    )
    config_str += "{:20} | ".format("Precision")
    with contextlib.suppress(AttributeError):
        config_str += "TF32: {:}, ".format(config.get_flag(trt.BuilderFlag.TF32))
    config_str += "FP16: {:}, INT8: {:}, Strict Types: {:}\n".format(
        config.get_flag(trt.BuilderFlag.FP16),
        config.get_flag(trt.BuilderFlag.INT8),
        config.get_flag(trt.BuilderFlag.STRICT_TYPES),
    )

    with contextlib.suppress(AttributeError):
        source_vals = [
            val.name for val in trt.TacticSource.__members__.values() if (1 << int(val)) & config.get_tactic_sources()
        ]
        config_str += "{:20} | {:}\n".format("Tactic Sources", source_vals)

    with contextlib.suppress(AttributeError):
        config_str += "{:20} | {:}\n".format("Safety Restricted", config.get_flag(trt.BuilderFlag.SAFETY_SCOPE))

    if config.int8_calibrator:
        config_str += "{:20} | {:}\n".format("Calibrator", config.int8_calibrator)
    config_str += "{:20} | {:} profile(s)".format("Profiles", config.num_optimization_profiles)
    return config_str


def check_profile(profile):
    if not bool(profile):
        G_LOGGER.critical("Profile is not valid, please provide profile data.\nNote: profile was: {:}".format(profile))
    return profile


def str_from_tensor(tensor, is_shape_tensor):
    ret = "Input "
    if is_shape_tensor:
        ret += "shape-tensor"
    else:
        ret += "tensor"
    ret += ": {:} (dtype={:}, shape={:})".format(tensor.name, tensor.dtype, tensor.shape)
    return ret


def get_input_metadata_from_profile(profile, network):
    """
    Returns metadata about the inputs based on the OPT values set in a profile.

    Args:
        profile (trt.IOptimizationProfile):
                The profile from which to retrieve input metada.
        network (trt.INetworkDefinition):
                The network the profile applies to.

    Returns:
        TensorMetadata:
                A mapping of input names to their types and shapes.
                Shapes are retrieved from the OPT values in the profile.
    """
    input_metadata = TensorMetadata()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.is_shape_tensor:
            shapes = profile.get_shape_input(tensor.name)
        else:
            shapes = profile.get_shape(tensor.name)

        if tuple(shapes[0]) != tuple(shapes[2]):
            G_LOGGER.warning(
                "Will use `opt` shapes from profile 0 for calibration. "
                "Note that even though `min` != `max` in this profile, calibration "
                "will use fixed input shapes (this is not necessarily an issue)."
            )
        # Always use opt shape
        input_metadata.add(name=tensor.name, dtype=np_dtype_from_trt(tensor.dtype), shape=shapes[1])
    return input_metadata


def add_binding_to_metadata(engine, binding, metadata, name_binding):
    # name_binding always comes from profile 0, since that's where we
    # get all binding names in the runner
    metadata.add(
        name=engine[name_binding],
        dtype=np_dtype_from_trt(engine.get_binding_dtype(binding)),
        shape=list(engine.get_binding_shape(binding)),
    )


def get_input_metadata_from_engine(engine, start_binding, end_binding):
    inputs = TensorMetadata()
    for index, binding in enumerate(range(start_binding, end_binding)):
        if engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, inputs, name_binding=index)
    return inputs


def get_output_metadata_from_engine(engine, start_binding, end_binding):
    outputs = TensorMetadata()
    for index, binding in enumerate(range(start_binding, end_binding)):
        if not engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, outputs, name_binding=index)
    return outputs


def str_from_engine(engine):
    bindings_per_profile = get_bindings_per_profile(engine)
    engine_str = "Name: {:} | {:}{:} Batch Engine ({:} layers)\n".format(
        engine.name,
        "Refittable " if engine.refittable else "",
        "Implicit"
        if hasattr(engine, "has_implicit_batch_dimension") and engine.has_implicit_batch_dimension
        else "Explicit",
        engine.num_layers,
    )
    engine_str += "\n"

    # Show metadata for the first profile (i.e. the dynamic shapes)
    input_metadata = get_input_metadata_from_engine(engine, 0, bindings_per_profile)
    engine_str += "---- {:} Engine Input(s) ----\n{:}\n\n".format(len(input_metadata), input_metadata)
    output_metadata = get_output_metadata_from_engine(engine, 0, bindings_per_profile)
    engine_str += "---- {:} Engine Output(s) ----\n{:}\n\n".format(len(output_metadata), output_metadata)

    engine_str += "---- Memory ----\nDevice Memory: {:} bytes\n\n".format(engine.device_memory_size)

    engine_str += "---- {:} Profile(s) ({:} Binding(s) Each) ----\n".format(
        engine.num_optimization_profiles, bindings_per_profile
    )
    for profile_index in range(engine.num_optimization_profiles):
        engine_str += "- Profile: {:}\n".format(profile_index)

        max_width = max([len(binding) for binding in engine]) + 8
        for offset in range(bindings_per_profile):
            binding = profile_index * bindings_per_profile + offset
            name = "[Name: {:}]".format(engine.get_binding_name(binding))
            engine_str += util.indent_block(
                "Binding Index: {:} {:} {:<{max_width}}".format(
                    binding, "(Input) " if engine.binding_is_input(binding) else "(Output)", name, max_width=max_width
                )
            )

            if engine.binding_is_input(binding):
                if engine.is_shape_binding(binding):
                    min_shape, opt_shape, max_shape = engine.get_profile_shape_input(profile_index, binding)
                else:
                    min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_index, binding)
                engine_str += " | Shapes: min={:}, opt={:}, max={:}\n".format(min_shape, opt_shape, max_shape)
            else:
                engine_str += " | Shape: {:}\n".format(engine.get_binding_shape(binding))
        engine_str += "\n"
    return util.indent_block(engine_str, level=0)


def get_bindings_per_profile(engine):
    return engine.num_bindings // engine.num_optimization_profiles


def get_active_profile_bindings(context):
    """
    Gets the start and end binding indices for the active optimization profile.

    Args:
        engine (trt.ICudaEngine): The engine in question.
        context (trt.IExecutionContext): The context where the profile is currently set.

    Returns:
        Tuple[int, int]: The start and end bindings indices, in that order
    """
    active_profile = context.active_optimization_profile
    bindings_per_profile = get_bindings_per_profile(context.engine)

    start_binding = bindings_per_profile * active_profile
    end_binding = start_binding + bindings_per_profile

    G_LOGGER.ultra_verbose(
        "Total # of Profiles: {:}, Bindings Per Profile: {:}, Active Profile: {:}, "
        "Start Binding: {:}, End Binding: {:}".format(
            context.engine.num_optimization_profiles, bindings_per_profile, active_profile, start_binding, end_binding
        )
    )
    return start_binding, end_binding
