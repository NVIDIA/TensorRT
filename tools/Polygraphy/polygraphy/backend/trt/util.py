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
import contextlib
import ctypes

import tensorrt as trt
from polygraphy.common import TensorMetadata
from polygraphy.common.constants import DEFAULT_SHAPE_VALUE
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc

TRT_LOGGER = trt.Logger()


def load_plugins(plugins):
    for plugin in plugins:
        G_LOGGER.info("Loading plugin library: {:}".format(plugin))
        ctypes.CDLL(plugin)


def check_onnx_parser_errors(parser):
    if parser.num_errors > 0:
        for index in range(parser.num_errors):
            G_LOGGER.error(parser.get_error(index))
        G_LOGGER.critical("Could not parse ONNX correctly")


def get_layer_class_mapping():
    return {
        trt.LayerType.CONVOLUTION: trt.IConvolutionLayer,
        trt.LayerType.FULLY_CONNECTED: trt.IFullyConnectedLayer,
        trt.LayerType.ACTIVATION: trt.IActivationLayer,
        trt.LayerType.POOLING: trt.IPoolingLayer,
        trt.LayerType.LRN: trt.ILRNLayer,
        trt.LayerType.SCALE: trt.IScaleLayer,
        trt.LayerType.SOFTMAX: trt.ISoftMaxLayer,
        trt.LayerType.DECONVOLUTION: trt.IDeconvolutionLayer,
        trt.LayerType.CONCATENATION: trt.IConcatenationLayer,
        trt.LayerType.ELEMENTWISE: trt.IElementWiseLayer,
        trt.LayerType.PLUGIN: trt.IPluginLayer,
        trt.LayerType.RNN: trt.IRNNLayer,
        trt.LayerType.UNARY: trt.IUnaryLayer,
        trt.LayerType.PADDING: trt.IPaddingLayer,
        trt.LayerType.SHUFFLE: trt.IShuffleLayer,
        trt.LayerType.REDUCE: trt.IReduceLayer,
        trt.LayerType.TOPK: trt.ITopKLayer,
        trt.LayerType.GATHER: trt.IGatherLayer,
        trt.LayerType.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer,
        trt.LayerType.RAGGED_SOFTMAX: trt.IRaggedSoftMaxLayer,
        trt.LayerType.CONSTANT: trt.IConstantLayer,
        trt.LayerType.RNN_V2: trt.IRNNv2Layer,
        trt.LayerType.IDENTITY: trt.IIdentityLayer,
        trt.LayerType.PLUGIN_V2: trt.IPluginV2Layer,
        trt.LayerType.SLICE: trt.ISliceLayer,
        trt.LayerType.SHAPE: trt.IShapeLayer,
        trt.LayerType.PARAMETRIC_RELU: trt.IParametricReLULayer,
        trt.LayerType.RESIZE: trt.IResizeLayer,
        trt.LayerType.TRIP_LIMIT: trt.ITripLimitLayer,
        trt.LayerType.RECURRENCE: trt.IRecurrenceLayer,
        trt.LayerType.ITERATOR: trt.IIteratorLayer,
        trt.LayerType.LOOP_OUTPUT: trt.ILoopOutputLayer,
        trt.LayerType.SELECT: trt.ISelectLayer,
        trt.LayerType.FILL: trt.IFillLayer,
    }


def get_input_metadata(network):
    inputs = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return inputs


def get_output_metadata(network):
    outputs = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return outputs


def str_from_network(network, mode="full"):
    """
    Converts a TensorRT network to a human-readable representation

    Args:
        network (trt.INetworkDefinition): The network.
        mode (str): Controls what is displayed for each layer. Choices: ["none", "basic", "attrs", "full"]

    Returns:
        str
    """
    import numpy as np

    try:
        LAYER_TYPE_CLASS_MAPPING = get_layer_class_mapping()
    except AttributeError:
        LAYER_TYPE_CLASS_MAPPING = {}

    def is_special_attribute(attr):
        return attr.startswith("__") and attr.endswith("__")

    def is_valid_attribute(attr, layer):
        if type(layer) == trt.IPoolingLayer or type(layer) == trt.IConvolutionLayer or type(layer) == trt.IDeconvolutionLayer:
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

    def get_layer_input_metadata(layer):
        meta = TensorMetadata()
        for i in range(layer.num_inputs):
            inp = layer.get_input(i)
            if inp:
                meta.add(inp.name, trt.nptype(inp.dtype), inp.shape)
        return meta

    def get_layer_output_metadata(layer):
        meta = TensorMetadata()
        for i in range(layer.num_outputs):
            outp = layer.get_output(i)
            if outp:
                meta.add(outp.name, trt.nptype(outp.dtype), outp.shape)
        return meta


    network_str = "Name: {:} | {:} Batch Network{:}\n".format(network.name,
                    "Implicit" if hasattr(network, "has_implicit_batch_dimension") and network.has_implicit_batch_dimension else "Explicit",
                    " with Explicit Precision " if hasattr(network, "has_explicit_precision") and network.has_explicit_precision else "")
    network_str += "\n"

    input_metadata = get_input_metadata(network)
    network_str += "---- {:} Network Inputs ----\n{:}\n\n".format(len(input_metadata), input_metadata)
    output_metadata = get_output_metadata(network)
    network_str += "---- {:} Network Outputs ----\n{:}\n\n".format(len(output_metadata), output_metadata)
    network_str += "---- {:} Layers ----\n".format(network.num_layers)
    if mode != "none":
        for index, layer in enumerate(network):
            if layer.type in LAYER_TYPE_CLASS_MAPPING:
                layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

            input_info = get_layer_input_metadata(layer)
            output_info = get_layer_output_metadata(layer)

            network_str += misc.str_from_layer("Layer", index, layer.name, layer.type, input_info, output_info)

            if mode in ["attrs", "full"]:
                # Exclude special attributes, as well as any attributes of the base layer class (those can be displayed above).
                attrs = [attr for attr in dir(layer) if not is_special_attribute(attr) and not hasattr(trt.ILayer, attr) and is_valid_attribute(attr, layer)]
                if attrs:
                    network_str += misc.indent_block("---- Attributes ----") + "\n"
                for attr in attrs:
                    val = getattr(layer, attr)
                    if mode == "full" or not isinstance(val, np.ndarray):
                        if layer.name:
                            network_str += "{:}.".format(layer.name)
                        network_str += misc.indent_block("{:} = {:}".format(attr, val)) + "\n"
            network_str += "\n"
    else:
        network_str += "(Use --mode to display)"

    return misc.indent_block(network_str, level=0)


def _get_network_outputs(network):
    return [network.get_output(index).name for index in range(network.num_outputs)]


def check_outputs_not_found(not_found, all_outputs):
    if not_found:
        all_outputs = misc.unique_list(all_outputs)
        G_LOGGER.critical("The following outputs: {:} were not found. "
                          "Note: Available tensors: {:}".format(not_found, all_outputs))


def mark_outputs(network, outputs):
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
                    network.mark_output(tensor)

    marked_outputs = set(_get_network_outputs(network))
    not_found = outputs - marked_outputs
    check_outputs_not_found(not_found, all_outputs)


def mark_layerwise(network):
    # Layers within loops cannot be marked as network outputs.
    LOOP_START_NAMES = ["TRIP_LIMIT", "ITERATOR"]
    LOOP_END_NAMES = ["LOOP_OUTPUT"]
    LOOP_START_LAYERS = [getattr(trt.LayerType, attr) for attr in LOOP_START_NAMES if hasattr(trt.LayerType, attr)]
    LOOP_END_LAYERS = [getattr(trt.LayerType, attr) for attr in LOOP_END_NAMES if hasattr(trt.LayerType, attr)]
    EXCLUDE_OUTPUT_LAYERS = [trt.LayerType.SHAPE, trt.LayerType.CONSTANT]
    num_tensors_marked = 0
    in_loop = False
    for layer in network:
        if layer.type in LOOP_START_LAYERS:
            G_LOGGER.warning("Loop detected. Please ensure the network is topologically sorted so that layers within "
                             "the loop body are not marked as network outputs in layerwise mode")
            in_loop = True
        elif layer.type in LOOP_END_LAYERS:
            in_loop = False

        def should_mark_layer():
            return not in_loop and layer.type not in EXCLUDE_OUTPUT_LAYERS

        if should_mark_layer():
            for index in range(layer.num_outputs):
                tensor = layer.get_output(index)
                if not tensor.is_network_output:
                    G_LOGGER.verbose("Marking {:} as an output".format(tensor.name))
                    network.mark_output(tensor)
                    num_tensors_marked += 1
    G_LOGGER.verbose("Marking {:} tensors as outputs".format(num_tensors_marked))


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
    config_str = "max_workspace_size={:} bytes ({:.2f} MB) | ".format(config.max_workspace_size, config.max_workspace_size / (1024.0 ** 2))
    with contextlib.suppress(AttributeError): config_str += "tf32={:}, ".format(config.get_flag(trt.BuilderFlag.TF32))
    config_str += "fp16={:}, int8={:}, strict_types={:} | {:} profiles".format(config.get_flag(trt.BuilderFlag.FP16),
                        config.get_flag(trt.BuilderFlag.INT8), config.get_flag(trt.BuilderFlag.STRICT_TYPES), config.num_optimization_profiles)
    return config_str


def check_profile(profile):
    if not bool(profile):
        G_LOGGER.critical("Profile is not valid, please provide profile data. Note: profile was: {:}".format(profile_shapes))
    return profile


def build_default_profile(builder, network, default_shape_value=None):
    default_shape_value = misc.default_value(default_shape_value, DEFAULT_SHAPE_VALUE)

    def override_shape(shape):
        return tuple([default_shape_value if misc.is_dimension_dynamic(dim) else dim for dim in shape])

    trt_profile = builder.create_optimization_profile()
    for idx in range(network.num_inputs):
        inp = network.get_input(idx)

        with G_LOGGER.verbosity(): # WAR for spam from TRT
            is_shape_tensor = inp.is_shape_tensor

        if is_shape_tensor:
            rank = inp.shape[0]
            shape = (default_shape_value, ) * rank
            G_LOGGER.warning("Input shape-tensor: {:32} | Adjusted dynamic shape-tensor contents to: {:}. If this is incorrect, please provide a profile "
                             "that sets the values for this input shape-tensor.".format(inp.name, shape, rank), mode=LogMode.ONCE)
            trt_profile.set_shape_input(inp.name, shape, shape, shape)
        else:
            shape = override_shape(inp.shape)
            if override_shape(inp.shape) != inp.shape:
                G_LOGGER.warning("Input tensor: {:32} | Adjusted dynamic shape: {:} to: {:}. If this is incorrect, please provide a profile "
                                 "that sets the shape for this input tensor.".format(inp.name, inp.shape, shape), mode=LogMode.ONCE)
            trt_profile.set_shape(inp.name, shape, shape, shape)
    return check_profile(trt_profile)


def build_profile(builder, network, profile):
    trt_profile = builder.create_optimization_profile()
    unused_keys = set(profile.keys())
    for idx in range(network.num_inputs):
        inp = network.get_input(idx)
        if inp.name in unused_keys:
            unused_keys.remove(inp.name)

        with G_LOGGER.verbosity(): # WAR for spam from TRT
            is_shape_tensor = inp.is_shape_tensor

        if is_shape_tensor:
            if inp.name in profile:
                shapes = profile[inp.name]
                trt_profile.set_shape_input(inp.name, shapes.min, shapes.opt, shapes.max)
                G_LOGGER.extra_verbose("Input shape-tensor: {:32} | Setting values to min: {:}, opt: {:}, max: {:}".format(inp.name, shapes.min, shapes.opt, shapes.max))
            else:
                G_LOGGER.warning("input shape-tensor: {:32} | No values provided. Assuming this is not a dynamic shape-tensor.".format(inp.name), mode=LogMode.ONCE)
        elif misc.is_shape_dynamic(inp.shape):
            shapes = profile[inp.name]
            trt_profile.set_shape(inp.name, shapes.min, shapes.opt, shapes.max)
            G_LOGGER.extra_verbose("Input tensor: {:32} | Setting shape to min: {:}, opt: {:}, max: {:}".format(inp.name, shapes.min, shapes.opt, shapes.max))

    if unused_keys:
        G_LOGGER.warning("Some inputs provided in the profile were unused: {:}".format(list(unused_keys)))

    return check_profile(trt_profile)


def get_input_metadata_from_profile(profile, network):
    """
    Returns metadata about the inputs based on a profile

    Args:
        profile (trt.IOptimizationProfile): The profile from which to retrieve input metada.
        network (trt.INetworkDefinition): The network

    Returns:
        TensorMetadata: A mapping of input names to their types and shapes.
    """
    input_metadata = TensorMetadata()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.is_shape_tensor:
            shapes = profile.get_shape_input(tensor.name)
        else:
            shapes = profile.get_shape(tensor.name)

        if tuple(shapes[0]) != tuple(shapes[1]):
            G_LOGGER.warning("In profile 0, min != max, using opt shapes for calibration")
        # Always use opt shape
        input_metadata.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=shapes[1])
    return input_metadata


def add_binding_to_metadata(engine, binding, metadata):
    metadata.add(
        name=engine[binding],
        dtype=trt.nptype(engine.get_binding_dtype(binding)),
        shape=list(engine.get_binding_shape(binding))
    )


def get_input_metadata_from_engine(engine, start_binding, end_binding):
    inputs = TensorMetadata()
    for binding in range(start_binding, end_binding):
        if engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, inputs)
    return inputs


def get_output_metadata_from_engine(engine, start_binding, end_binding):
    outputs = TensorMetadata()
    for binding in range(start_binding, end_binding):
        if not engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, outputs)
    return outputs


def str_from_engine(engine):
    bindings_per_profile = get_bindings_per_profile(engine)
    engine_str = "Name: {:} | {:}{:} Batch Engine ({:} layers)\n".format(engine.name,
                        "Refittable " if engine.refittable else "",
                        "Implicit" if hasattr(engine, "has_implicit_batch_dimension") and engine.has_implicit_batch_dimension else "Explicit",
                        engine.num_layers)
    engine_str += "\n"

    # Show metadata for the first profile (i.e. the dynamic shapes)
    input_metadata = get_input_metadata_from_engine(engine, 0, bindings_per_profile)
    engine_str += "---- {:} Engine Inputs ----\n{:}\n\n".format(len(input_metadata), input_metadata)
    output_metadata = get_output_metadata_from_engine(engine, 0, bindings_per_profile)
    engine_str += "---- {:} Engine Outputs ----\n{:}\n\n".format(len(output_metadata), output_metadata)

    engine_str += "---- Memory ----\nWorkspace Memory: {:} bytes\n\n".format(engine.max_workspace_size)

    engine_str += "---- {:} Profiles ({:} Bindings Each) ----\n".format(engine.num_optimization_profiles, bindings_per_profile)
    for profile_index in range(engine.num_optimization_profiles):
        engine_str += "- Profile: {:}\n".format(profile_index)

        max_width = max([len(binding) for binding in engine]) + 8
        for offset in range(bindings_per_profile):
            binding = profile_index * bindings_per_profile + offset
            name =  "[Name: {:}]".format(engine.get_binding_name(binding))
            engine_str += misc.indent_block("Binding Index: {:} {:} {:<{max_width}}".format(
                                binding, "(Input) " if engine.binding_is_input(binding) else "(Output)", name, max_width=max_width))

            if engine.binding_is_input(binding):
                if engine.is_shape_binding(binding):
                    min_shape, opt_shape, max_shape = engine.get_profile_shape_input(profile_index, binding)
                else:
                    min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_index, binding)
                engine_str += " | Shapes: min={:}, opt={:}, max={:}\n".format(min_shape, opt_shape, max_shape)
            else:
                engine_str += " | Shape: {:}".format(tuple(output_metadata[engine[offset]][1]))
        engine_str += "\n"
    return misc.indent_block(engine_str, level=0)


def get_bindings_per_profile(engine):
    return engine.num_bindings // engine.num_optimization_profiles


def get_active_profile_bindings(engine, context):
    """
    Gets the start and end binding indices for the active optimization profile.

    Args:
        engine (trt.ICudaEngine): The engine in question.
        context (trt.IExecutionContext): The context where the profile is currently set.

    Returns:
        Tuple[int, int]: The start and end bindings indices, in that order
    """
    active_profile = context.active_optimization_profile
    bindings_per_profile = get_bindings_per_profile(engine)

    start_binding = bindings_per_profile * active_profile
    end_binding = start_binding + bindings_per_profile

    G_LOGGER.ultra_verbose("Total # of Profiles: {:}, Bindings Per Profile: {:}, Active Profile: {:}, "
                           "Start Binding: {:}, End Binding: {:}".format(
                                engine.num_optimization_profiles, bindings_per_profile,
                                active_profile, start_binding, end_binding))
    return start_binding, end_binding
