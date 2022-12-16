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
import contextlib
import json
import os
import signal

from polygraphy import config, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.exception import PolygraphyException
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

    LoggerType = trt.Logger
    if mod.version(trt.__version__) >= mod.version("8.0"):

        class CustomTrtLogger(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, msg):
                try:
                    log_func = {
                        # This function cannot throw, so `critical` should not be used here!
                        trt.Logger.INTERNAL_ERROR: G_LOGGER.error,
                        trt.Logger.ERROR: G_LOGGER.error,
                        # Reduce warning spam from TRT.
                        trt.Logger.WARNING: lambda msg: G_LOGGER.warning(msg, mode=LogMode.ONCE),
                        trt.Logger.INFO: G_LOGGER.verbose,
                        trt.Logger.VERBOSE: G_LOGGER.extra_verbose,
                    }.get(severity, G_LOGGER.super_verbose)

                    log_func(msg)
                except KeyboardInterrupt:
                    # `log()` is `noexcept` so we need to convert exceptions to signals so that
                    # ctrl-C will work as expected.
                    os.kill(os.getpid(), signal.SIGTERM)

        LoggerType = CustomTrtLogger

    if TRT_LOGGER is None:
        TRT_LOGGER = LoggerType()
    return TRT_LOGGER


def _should_use_v3_api():
    return mod.version(trt.__version__) > mod.version("8.5.0.9")


def fail_unavailable(what):
    G_LOGGER.backtrace()
    G_LOGGER.critical(f"{what} is not available on TensorRT version {trt.__version__}.")


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
                G_LOGGER.warning(f"Could not find layer type: {layer_type} or layer class: {layer_cls}")
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
    try_add("GRID_SAMPLE", "IGridSampleLayer")
    try_add("ONE_HOT", "IOneHotLayer")
    try_add("NON_ZERO", "INonZeroLayer")
    try_add("NMS", "INMSLayer")
    try_add("REVERSE_SEQUENCE", "IReverseSequenceLayer")
    try_add("NORMALIZATION", "INormalizationLayer")
    try_add("CAST", "ICastLayer")

    return layer_class_mapping


def np_dtype_from_trt(trt_dtype):
    # trt.nptype uses NumPy, so to make autoinstall work, we need to trigger it before that.
    mod.autoinstall(np)
    return np.dtype(trt.nptype(trt_dtype))


def get_network_input_names_meta(network):
    names = []
    meta = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        names.append(tensor.name)
        meta.add(name=tensor.name, dtype=np_dtype_from_trt(tensor.dtype), shape=tensor.shape)
    return names, meta


def get_network_output_names_meta(network):
    names = []
    meta = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        names.append(tensor.name)
        meta.add(name=tensor.name, dtype=np_dtype_from_trt(tensor.dtype), shape=tensor.shape)
    return names, meta


def get_layer_input_names_meta(layer):
    names = []
    meta = TensorMetadata()
    for i in range(layer.num_inputs):
        inp = layer.get_input(i)
        if inp:
            names.append(inp.name)
            meta.add(inp.name, np_dtype_from_trt(inp.dtype), inp.shape)
    return names, meta


def get_layer_output_names_meta(layer):
    names = []
    meta = TensorMetadata()
    for i in range(layer.num_outputs):
        out = layer.get_output(i)
        if out:
            names.append(out.name)
            meta.add(out.name, np_dtype_from_trt(out.dtype), out.shape)
    return names, meta


def str_from_layer(layer, index):
    input_names, input_meta = get_layer_input_names_meta(layer)
    output_names, output_meta = get_layer_output_names_meta(layer)
    return util.str_from_layer(
        "Layer", index, layer.name, layer.type, input_names, input_meta, output_names, output_meta
    )


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


def str_from_network(network, show_layers=None, show_attrs=None, show_weights=None):
    """
    Converts a TensorRT network to a human-readable representation

    Args:
        network (trt.INetworkDefinition): The network.
        show_layers (bool): Whether to display per-layer information.
        show_attrs (bool): Whether to display per-layer attributes.
        show_weights (bool): Whether to display the value of weights.

    Returns:
        str
    """
    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)
    show_weights = util.default(show_weights, False)

    LAYER_TYPE_CLASS_MAPPING = get_layer_class_mapping()

    network_str = f"Name: {network.name} | {'Implicit' if hasattr(network, 'has_implicit_batch_dimension') and network.has_implicit_batch_dimension else 'Explicit'} Batch Network{' with Explicit Precision ' if hasattr(network, 'has_explicit_precision') and network.has_explicit_precision else ''}\n"
    network_str += "\n"

    _, input_metadata = get_network_input_names_meta(network)
    network_str += f"---- {len(input_metadata)} Network Input(s) ----\n{input_metadata}\n\n"
    _, output_metadata = get_network_output_names_meta(network)
    network_str += f"---- {len(output_metadata)} Network Output(s) ----\n{output_metadata}\n\n"
    network_str += f"---- {network.num_layers} Layer(s) ----\n"
    if show_layers:
        for index, layer in enumerate(network):
            if layer.type in LAYER_TYPE_CLASS_MAPPING:
                layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

            network_str += str_from_layer(layer, index)

            if show_attrs:
                # Exclude special attributes, as well as any attributes of the base layer class (those can be displayed above).
                attrs = get_layer_attribute_names(layer)
                if attrs:
                    network_str += util.indent_block("---- Attributes ----") + "\n"
                for attr in attrs:
                    with G_LOGGER.verbosity():
                        val = getattr(layer, attr)
                    if show_weights or not isinstance(val, np.ndarray):
                        attr_str = ""
                        if layer.name:
                            attr_str += f"{layer.name}."
                        network_str += util.indent_block(f"{attr_str}{attr} = {val}") + "\n"
            network_str += "\n"

    return util.indent_block(network_str, level=0)


def get_all_tensors(network):
    all_tensors = set()
    for layer in network:
        for i in range(layer.num_inputs):
            all_tensors.add(layer.get_input(i))
        for i in range(layer.num_outputs):
            all_tensors.add(layer.get_output(i))
    # Optional tensors that are omitted are reported as `None`s, so we need to exclude them.
    return {t.name: t for t in all_tensors if t is not None}


def mark_outputs(network, outputs):
    """
    Mark the specified outputs as network outputs.

    Args:
        network (trt.INetworkDefinition): The network in which to mark outputs.
        outputs (Sequence[str]): The names of tensors to mark as outputs.
    """
    outputs = util.unique_list(outputs)

    tensor_map = get_all_tensors(network)
    util.check_sequence_contains(
        tensor_map.keys(), outputs, name="the network", items_name="outputs", check_extra=False
    )

    for tensor in tensor_map.values():
        # Clear all old outputs
        if tensor.is_network_output:
            network.unmark_output(tensor)

    for name in outputs:
        G_LOGGER.ultra_verbose(f"Marking {name} as an output")
        network.mark_output(tensor_map[name])


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
                if tensor is not None:
                    outputs.append(tensor.name)

    G_LOGGER.verbose(f"Marking {len(outputs)} tensors as outputs")
    mark_outputs(network, outputs)


def unmark_outputs(network, outputs):
    outputs = util.unique_list(outputs)

    tensor_map = get_all_tensors(network)
    util.check_sequence_contains(
        tensor_map.keys(), outputs, name="the network", items_name="outputs", check_extra=False
    )

    for name in outputs:
        tensor = tensor_map[name]
        if tensor.is_network_output:
            network.unmark_output(tensor)


def str_from_config(config):
    # Check the default device type so that we can trigger this from the tests.
    # On non-DLA platforms, config.DLA_core can never be set to anything other than -1,
    # but default_device_type can be set to DLA..
    using_dla = config.DLA_core >= 0 or config.default_device_type == trt.DeviceType.DLA

    lines = []

    def str_from_list(lst):
        return "[" + ", ".join(lst) + "]"

    def add_line(title, line):
        lines.append((f"{title:{22}} | " + line).strip())

    def get_enabled_enum_vals(EnumType, is_enabled):
        # is_enabled is a Callable[[enum_val], bool] which reports whether to include the enum value.
        return [name for name, enum_val in EnumType.__members__.items() if is_enabled(enum_val)]

    # Flags
    enabled_builder_flags = get_enabled_enum_vals(trt.BuilderFlag, lambda flag: config.get_flag(flag))
    add_line("Flags", f"{str_from_list(enabled_builder_flags)}")

    # Engine Capability
    with contextlib.suppress(AttributeError):
        add_line("Engine Capability", str(config.engine_capability))

    # Memory Pools
    with contextlib.suppress(AttributeError):
        mem_pool_limits = [
            f"{name}: {config.get_memory_pool_limit(pool_type) / float(1<<20):.2f} MiB"
            for name, pool_type in trt.MemoryPoolType.__members__.items()
            # Only show DLA memory pools when DLA is in use
            if (not name.startswith("DLA") or using_dla)
        ]
        add_line("Memory Pools", f"{str_from_list(mem_pool_limits)}")

    # Tactic Sources
    with contextlib.suppress(AttributeError):
        source_vals = get_enabled_enum_vals(trt.TacticSource, lambda val: (1 << int(val)) & config.get_tactic_sources())
        add_line("Tactic Sources", f"{str_from_list(source_vals)}")

    # DLA
    if using_dla:
        add_line("DLA", f"Default Device Type: {config.default_device_type}, Core: {config.DLA_core}")

    # Profiling Verbosity
    with contextlib.suppress(AttributeError):
        add_line("Profiling Verbosity", f"{config.profiling_verbosity}")

    # Optimization Profiles
    if config.num_optimization_profiles > 1:  # Not particularly interesting unless there are multiple.
        add_line("Optimization Profiles", f"{config.num_optimization_profiles} profile(s)")

    # Preview Features
    with contextlib.suppress(AttributeError):
        feature_vals = get_enabled_enum_vals(trt.PreviewFeature, lambda val: config.get_preview_feature(val))
        if feature_vals:
            add_line("Preview Features", f"{str_from_list(feature_vals)}")

    # Calibrator
    if config.int8_calibrator:
        add_line("Calibrator", f"{config.int8_calibrator}")

    return "\n".join(lines)


def check_profile(profile):
    if not bool(profile):
        G_LOGGER.critical(f"Profile is not valid, please provide profile data.\nNote: profile was: {profile}")
    return profile


def str_from_tensor(tensor, is_shape_tensor):
    ret = "Input "
    if is_shape_tensor:
        ret += "shape-tensor"
    else:
        ret += "tensor"
    ret += f": {tensor.name} (dtype={tensor.dtype}, shape={tensor.shape})"
    return ret


def get_input_metadata_from_network(network, profile):
    """
    Returns metadata about the inputs of a network, referring to the values
    set in a profile for dynamic shapes.

    Args:
        network (trt.INetworkDefinition):
                The network the profile applies to.


        profile (trt.IOptimizationProfile):
                The profile from which to retrieve input metadata.

    Returns:
        TensorMetadata:
                A mapping of input names to their types and shapes.
                Shapes are retrieved from the OPT values in the profile.

    Raises:
        PolygraphyException:
                If the network has dynamic shapes or shape tensor inputs but no profile
                was provided.
    """
    input_metadata = TensorMetadata()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        # Only access the profile if we actually need to.
        # This way, this method works with static networks even without a profile set.
        shape = tensor.shape
        min_shape = None
        max_shape = None
        if tensor.is_shape_tensor or util.is_shape_dynamic(tensor.shape):
            if tensor.is_shape_tensor:
                min_shape, _, max_shape = profile.get_shape_input(tensor.name)
            else:
                min_shape, _, max_shape = profile.get_shape(tensor.name)

        input_metadata.add(
            name=tensor.name,
            dtype=np_dtype_from_trt(tensor.dtype),
            shape=shape,
            min_shape=min_shape,
            max_shape=max_shape,
        )
    return input_metadata


# calib_profile parameter is used to bypass `get_calibration_profile()` to make this work on TRT 7.0 and older.
def try_setup_polygraphy_calibrator(config, network, calib_profile=None):
    """
    Tries to call setup methods specific to Polygraphy calibrators.
    Returns early if there is no calibrator or if it is not a Polygraphy calibrator.
    """
    calibrator = config.int8_calibrator
    if calibrator is None or not (
        hasattr(calibrator, "is_polygraphy_calibrator") and calibrator.is_polygraphy_calibrator
    ):
        # No calibrator or not a Polygraphy calibrator.
        return

    if calib_profile is None:
        try:
            calib_profile = config.get_calibration_profile()
        except AttributeError:
            G_LOGGER.extra_verbose("Cannot get calibration profile on TensorRT 7.0 and older.")
            # Return early so we don't emit extraneous warnings on TRT 7.0 and older.
            return

    try:
        input_metadata = get_input_metadata_from_network(network, calib_profile)
    except PolygraphyException as err:
        G_LOGGER.warning(
            "Could not determine input_metadata to provide to the calibrator because no calibration profile is set. "
            "Please either set a calibration profile in the config or call `calibrator.set_input_metadata()` manually. "
            f"\nNote: Error was:\n{err}",
            mode=LogMode.ONCE,
        )
    else:
        calibrator.set_input_metadata(input_metadata)


def get_metadata_from_engine(engine, mode):
    meta = TensorMetadata()
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(name) != mode:
            continue

        meta.add(name=name, dtype=np_dtype_from_trt(engine.get_tensor_dtype(name)), shape=engine.get_tensor_shape(name))
    return meta


def str_from_engine(engine, show_layers=None, show_attrs=None):
    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)

    if _should_use_v3_api():
        num_io_tensors = engine.num_io_tensors
    else:
        num_io_tensors = get_bindings_per_profile(engine)

    engine_str = f"Name: {engine.name} | {'Refittable ' if engine.refittable else ''}{'Implicit' if hasattr(engine, 'has_implicit_batch_dimension') and engine.has_implicit_batch_dimension else 'Explicit'} Batch Engine\n"
    engine_str += "\n"

    # Show metadata for the first profile (i.e. the dynamic shapes)
    if _should_use_v3_api():
        input_metadata = get_metadata_from_engine(engine, mode=trt.TensorIOMode.INPUT)
        output_metadata = get_metadata_from_engine(engine, mode=trt.TensorIOMode.OUTPUT)
    else:
        input_metadata = get_input_metadata_from_engine(engine, 0, num_io_tensors)
        output_metadata = get_output_metadata_from_engine(engine, 0, num_io_tensors)

    engine_str += f"---- {len(input_metadata)} Engine Input(s) ----\n{input_metadata}\n\n"

    engine_str += f"---- {len(output_metadata)} Engine Output(s) ----\n{output_metadata}\n\n"

    engine_str += f"---- Memory ----\nDevice Memory: {engine.device_memory_size} bytes\n\n"

    engine_str += f"---- {engine.num_optimization_profiles} Profile(s) ({num_io_tensors} Tensor(s) Each) ----\n"
    for profile_index in range(engine.num_optimization_profiles):
        engine_str += f"- Profile: {profile_index}\n"

        if _should_use_v3_api():
            max_width = max([len(engine.get_tensor_name(idx)) for idx in range(engine.num_io_tensors)]) + 8
        else:
            max_width = max([len(binding) for binding in engine]) + 8

        for idx in range(num_io_tensors):
            if _should_use_v3_api():
                name = engine.get_tensor_name(idx)
                binding_type = " (Input)" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "(Output)"
                engine_str += util.indent_block(f"Tensor: {name:<{max_width}} {binding_type}, Index: {idx}")
            else:
                binding = profile_index * num_io_tensors + idx
                name = f"[Name: {engine.get_binding_name(binding)}]"
                binding_type = "(Input) " if engine.binding_is_input(binding) else "(Output)"
                engine_str += util.indent_block(f"Binding Index: {binding} {binding_type} {name:<{max_width}}")

            if _should_use_v3_api():
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(name, profile_index)
                    engine_str += f" | Shapes: min={min_shape}, opt={opt_shape}, max={max_shape}\n"
                else:
                    engine_str += f" | Shape: {engine.get_tensor_shape(name)}\n"
            else:
                if engine.binding_is_input(binding):
                    if engine.is_shape_binding(binding):
                        min_shape, opt_shape, max_shape = engine.get_profile_shape_input(profile_index, binding)
                    else:
                        min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_index, binding)
                    engine_str += f" | Shapes: min={min_shape}, opt={opt_shape}, max={max_shape}\n"
                else:
                    engine_str += f" | Shape: {engine.get_binding_shape(binding)}\n"
        engine_str += "\n"

    layers_per_profile = engine.num_layers // engine.num_optimization_profiles
    engine_str += (
        f"---- {layers_per_profile} Layer(s){' Per Profile' if engine.num_optimization_profiles > 1 else ''} ----\n"
    )
    if show_layers:
        try:
            inspector = engine.create_engine_inspector()
        except AttributeError:
            G_LOGGER.warning(
                f"Cannot show layer information because IEngineInspector is not available in this version of TensorRT ({trt.__version__})"
            )
        else:
            for profile_idx in range(engine.num_optimization_profiles):
                indent_level = 0
                if engine.num_optimization_profiles >= 1:
                    indent_level = 1
                    engine_str += f"- Profile: {profile_idx}\n"

                offset = profile_idx * layers_per_profile
                for index in range(layers_per_profile):
                    layer_info = json.loads(
                        inspector.get_layer_information(offset + index, trt.LayerInformationFormat.JSON)
                    )

                    op = "Unknown"
                    input_names, input_meta = [], TensorMetadata()
                    output_names, output_meta = [], TensorMetadata()
                    origin = "Unknown"
                    tactic = "Unknown"
                    if engine.profiling_verbosity == trt.ProfilingVerbosity.DETAILED:
                        name = layer_info.get("Name", "Unknown")
                        op = layer_info.get("LayerType", "Unknown")

                        def names_meta_from_inspector(key):
                            names = []
                            meta = TensorMetadata()
                            info = layer_info.get(key)
                            if info is None:
                                return meta
                            for elem in info:
                                names.append(elem["Name"])
                                meta.add(name=elem["Name"], dtype=None, shape=elem["Dimensions"])
                            return names, meta

                        input_names, input_meta = names_meta_from_inspector("Inputs")
                        output_names, output_meta = names_meta_from_inspector("Outputs")
                        origin = layer_info.get("Origin", "Unknown")
                        tactic = layer_info.get("TacticValue", "Unknown")
                    else:
                        G_LOGGER.warning(
                            f"This engine was created with a profiling verbosity of: {engine.profiling_verbosity}. Some layer information may be missing. Try setting a higher profiling verbosity to see more detailed layer information. ",
                            mode=LogMode.ONCE,
                        )
                        name = layer_info

                    engine_str += (
                        util.indent_block(
                            util.str_from_layer(
                                "Layer", index, name, op, input_names, input_meta, output_names, output_meta
                            ),
                            indent_level,
                        )
                        + "\n"
                    )

                    if show_attrs:
                        engine_str += util.indent_block("---- Attributes ----", indent_level + 1) + "\n"
                        engine_str += util.indent_block(f"Origin = {origin}", indent_level + 1) + "\n"
                        engine_str += util.indent_block(f"Tactic = {tactic}", indent_level + 1) + "\n"

                    engine_str += "\n"

    return util.indent_block(engine_str, level=0)



# V2 APIs
def add_binding_to_metadata(engine, binding, metadata, name_binding):
    if _should_use_v3_api():
        G_LOGGER.internal_error("This function should not be called when using the V3 API")

    # name_binding always comes from profile 0, since that's where we
    # get all binding names in the runner
    metadata.add(
        name=engine[name_binding],
        dtype=np_dtype_from_trt(engine.get_binding_dtype(binding)),
        shape=list(engine.get_binding_shape(binding)),
    )


def get_input_metadata_from_engine(engine, start_binding, end_binding):
    if _should_use_v3_api():
        G_LOGGER.internal_error("This function should not be called when using the V3 API")

    inputs = TensorMetadata()
    for index, binding in enumerate(range(start_binding, end_binding)):
        if engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, inputs, name_binding=index)
    return inputs


def get_output_metadata_from_engine(engine, start_binding, end_binding):
    if _should_use_v3_api():
        G_LOGGER.internal_error("This function should not be called when using the V3 API")

    outputs = TensorMetadata()
    for index, binding in enumerate(range(start_binding, end_binding)):
        if not engine.binding_is_input(binding):
            add_binding_to_metadata(engine, binding, outputs, name_binding=index)
    return outputs


def get_bindings_per_profile(engine):
    if _should_use_v3_api():
        G_LOGGER.internal_error("This function should not be called when using the V3 API")

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
    if _should_use_v3_api():
        G_LOGGER.internal_error("This function should not be called when using the V3 API")

    active_profile = context.active_optimization_profile
    if active_profile < 0:
        G_LOGGER.critical(
            f"Cannot determine profile bindings since the optimization profile for this context is set to: {active_profile}"
        )

    bindings_per_profile = get_bindings_per_profile(context.engine)

    start_binding = bindings_per_profile * active_profile
    end_binding = start_binding + bindings_per_profile

    G_LOGGER.ultra_verbose(
        f"Total # of Profiles: {context.engine.num_optimization_profiles}, Bindings Per Profile: {bindings_per_profile}, "
        f"Active Profile: {active_profile}, Start Binding: {start_binding}, End Binding: {end_binding}"
    )
    return start_binding, end_binding
