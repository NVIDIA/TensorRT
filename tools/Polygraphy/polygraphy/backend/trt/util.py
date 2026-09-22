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
import contextlib
import fnmatch
import json
import os
import re
import signal
import math
from polygraphy import config, mod, util, cuda
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.common import TensorMetadata
from polygraphy.common import FormattedArray
from polygraphy.datatype import DataType
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.json import load_json
from polygraphy.comparator import RunResults

trt = lazy_import_trt()
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

    if TRT_LOGGER is not None:
        return TRT_LOGGER

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
                    trt.Logger.WARNING: lambda msg: G_LOGGER.warning(
                        msg, mode=LogMode.ONCE
                    ),
                    trt.Logger.INFO: G_LOGGER.verbose,
                    trt.Logger.VERBOSE: G_LOGGER.extra_verbose,
                }.get(severity, G_LOGGER.super_verbose)

                log_func(msg)
            except KeyboardInterrupt:
                # `log()` is `noexcept` so we need to convert exceptions to signals so that
                # ctrl-C will work as expected.
                os.kill(os.getpid(), signal.SIGTERM)

    TRT_LOGGER = CustomTrtLogger()
    return TRT_LOGGER


def fail_unavailable(what):
    G_LOGGER.backtrace()
    G_LOGGER.critical(f"{what} is not available on TensorRT version {trt.__version__}.")


def check_onnx_parser_errors(parser, success):
    if parser.num_errors > 0:
        for index in range(parser.num_errors):
            G_LOGGER.error(parser.get_error(index))
        G_LOGGER.critical("Could not parse ONNX correctly")

    if not success:
        G_LOGGER.critical(
            "Failed to parse ONNX model. Does the model file exist and contain a valid ONNX model?"
        )


def get_layer_class_mapping():
    layer_class_mapping = {}

    def try_add(layer_type, layer_cls):
        try:
            layer_type = getattr(trt.LayerType, layer_type)
            layer_cls = getattr(trt, layer_cls)
        except AttributeError:
            if config.INTERNAL_CORRECTNESS_CHECKS:
                G_LOGGER.warning(
                    f"Could not find layer type: {layer_type} or layer class: {layer_cls}"
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
    try_add("GRID_SAMPLE", "IGridSampleLayer")
    try_add("ONE_HOT", "IOneHotLayer")
    try_add("NON_ZERO", "INonZeroLayer")
    try_add("NMS", "INMSLayer")
    try_add("REVERSE_SEQUENCE", "IReverseSequenceLayer")
    try_add("NORMALIZATION", "INormalizationLayer")
    try_add("CAST", "ICastLayer")
    try_add("SQUEEZE", "ISqueezeLayer")
    try_add("UNSQUEEZE", "IUnsqueezeLayer")
    try_add("CUMULATIVE", "ICumulativeLayer")
    try_add("DYNAMIC_QUANTIZE", "IDynamicQuantizeLayer")
    try_add("ATTENTION_INPUT", "IAttentionInputLayer")
    try_add("ATTENTION_OUTPUT", "IAttentionOutputLayer")
    try_add("ROTARY_EMBEDDING", "IRotaryEmbeddingLayer")
    try_add("KV_CACHE_UPDATE", "IKVCacheUpdateLayer")
    try_add("MOE", "IMoELayer")
    try_add("DIST_COLLECTIVE", "IDistCollectiveLayer")

    return layer_class_mapping


def get_network_input_names_meta(network):
    names = []
    meta = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        names.append(tensor.name)
        meta.add(
            name=tensor.name,
            dtype=DataType.from_dtype(tensor.dtype, "tensorrt"),
            shape=tensor.shape,
        )
    return names, meta


def get_network_output_names_meta(network):
    names = []
    meta = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        names.append(tensor.name)
        meta.add(
            name=tensor.name,
            dtype=DataType.from_dtype(tensor.dtype, "tensorrt"),
            shape=tensor.shape,
        )
    return names, meta


def get_layer_input_names_meta(layer):
    names = []
    meta = TensorMetadata()
    for i in range(layer.num_inputs):
        inp = layer.get_input(i)
        if inp:
            names.append(inp.name)
            meta.add(inp.name, DataType.from_dtype(inp.dtype, "tensorrt"), inp.shape)
    return names, meta


def get_layer_output_names_meta(layer):
    names = []
    meta = TensorMetadata()
    for i in range(layer.num_outputs):
        out = layer.get_output(i)
        if out:
            names.append(out.name)
            meta.add(out.name, DataType.from_dtype(out.dtype, "tensorrt"), out.shape)
    return names, meta


def str_from_layer(layer, index):
    input_names, input_meta = get_layer_input_names_meta(layer)
    output_names, output_meta = get_layer_output_names_meta(layer)
    return util.str_from_layer(
        "Layer",
        index,
        layer.name,
        layer.type,
        input_names,
        input_meta,
        output_names,
        output_meta,
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
        if not is_special_attribute(attr)
        and not hasattr(trt.ILayer, attr)
        and is_valid_attribute(attr, layer)
    ]


def str_from_network(network, show_layers=None, show_attrs=None, show_weights=None):
    """
    Converts a TensorRT network to a human-readable representation.

    Args:
        network (trt.INetworkDefinition): The network.
        show_layers (bool): Whether to display per-layer information.
        show_attrs (bool): Whether to display per-layer attributes.
        show_weights (bool): Whether to display the value of weights.

    Returns:
        str
    """
    from polygraphy.tools.inspect.subtool.model.extractors import (
        graph_data_from_trt_network,
    )
    from polygraphy.tools.inspect.subtool.model.text import str_from_graph_data

    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)
    show_weights = util.default(show_weights, False)

    graph_data = graph_data_from_trt_network(network, show_weights=show_weights)
    return str_from_graph_data(
        graph_data,
        show_layers=show_layers,
        show_attrs=show_attrs,
        show_weights=show_weights,
    )


def get_all_tensors(network):
    all_tensors = set()
    for layer in network:
        for i in range(layer.num_inputs):
            all_tensors.add(layer.get_input(i))
        for i in range(layer.num_outputs):
            all_tensors.add(layer.get_output(i))
    # Optional tensors that are omitted are reported as `None`s, so we need to exclude them.
    return {t.name: t for t in all_tensors if t is not None}


def get_all_io_tensors(network):
    all_io_tensors = set()
    for i in range(network.num_inputs):
        all_io_tensors.add(network.get_input(i))
    for i in range(network.num_outputs):
        all_io_tensors.add(network.get_output(i))
    return {t.name: t for t in all_io_tensors if t is not None}


def _expand_wildcard_patterns(names, candidates):
    """Expand fnmatch wildcard patterns in `names` against `candidates`."""
    result = []
    for name in names:
        if any(c in name for c in ("*", "?", "[", "]")):
            matches = fnmatch.filter(candidates, name)
            if not matches:
                G_LOGGER.warning(f"No tensors matched wildcard pattern: '{name}'")
            result.extend(matches)
        else:
            result.append(name)
    return result


def mark_outputs(network, outputs):
    """
    Mark the specified outputs as network outputs.

    Args:
        network (trt.INetworkDefinition): The network in which to mark outputs.
        outputs (Sequence[str]): The names of tensors to mark as outputs.
            Supports fnmatch wildcard patterns (e.g. ``"conv_*"``).
    """
    tensor_map = get_all_tensors(network)
    outputs = _expand_wildcard_patterns(outputs, list(tensor_map.keys()))
    outputs = util.unique_list(outputs)
    if not outputs:
        G_LOGGER.critical(
            "No outputs were selected. Please check your wildcard patterns."
        )

    util.check_sequence_contains(
        tensor_map.keys(),
        outputs,
        name="the network",
        items_name="outputs",
        check_extra=False,
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
    LOOP_START_LAYERS = [
        getattr(trt.LayerType, attr)
        for attr in LOOP_START_NAMES
        if hasattr(trt.LayerType, attr)
    ]
    LOOP_END_LAYERS = [
        getattr(trt.LayerType, attr)
        for attr in LOOP_END_NAMES
        if hasattr(trt.LayerType, attr)
    ]
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


def mark_by_layer_type(network, layer_types):
    """
    Mark outputs of all layers whose type matches any of the given TRT layer type names (case-insensitive).

    Args:
        network (trt.INetworkDefinition): The TensorRT network.
        layer_types (Sequence[str]): Layer type names to match against ``trt.LayerType`` enum values.
            If a name does not match any enum value, a critical error is raised with suggestions.
    """
    import difflib

    all_type_names = [
        attr
        for attr in dir(trt.LayerType)
        if not attr.startswith("_")
        and isinstance(getattr(trt.LayerType, attr), trt.LayerType)
    ]
    lower_to_canonical = {t.lower(): t for t in all_type_names}

    outputs = []
    for requested in layer_types:
        canonical = lower_to_canonical.get(requested.lower())
        if canonical is None:
            similar = difflib.get_close_matches(
                requested.upper(), all_type_names, n=5, cutoff=0.6
            )
            hint = (
                f" Did you mean one of: {similar}?"
                if similar
                else f" Available layer types: {sorted(all_type_names)}"
            )
            G_LOGGER.critical(f"No TensorRT layer type '{requested}' was found.{hint}")
        else:
            target_type = getattr(trt.LayerType, canonical)
            for layer in network:
                if layer.type == target_type:
                    for i in range(layer.num_outputs):
                        tensor = layer.get_output(i)
                        if tensor is not None:
                            outputs.append(tensor.name)

    if outputs:
        mark_outputs(network, outputs)


def unmark_outputs(network, outputs):
    tensor_map = get_all_tensors(network)
    outputs = _expand_wildcard_patterns(outputs, list(tensor_map.keys()))
    outputs = util.unique_list(outputs)

    util.check_sequence_contains(
        tensor_map.keys(),
        outputs,
        name="the network",
        items_name="outputs",
        check_extra=False,
    )

    for name in outputs:
        tensor = tensor_map[name]
        if tensor.is_network_output:
            network.unmark_output(tensor)


def str_from_config(config, network):
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
        return [
            name
            for name, enum_val in EnumType.__members__.items()
            if is_enabled(enum_val)
        ]

    # Flags
    enabled_builder_flags = get_enabled_enum_vals(
        trt.BuilderFlag, lambda flag: config.get_flag(flag)
    )
    enabled_builder_flags += get_enabled_enum_vals(
        trt.NetworkDefinitionCreationFlag,
        lambda flag: hasattr(network, "get_flag") and network.get_flag(flag),
    )
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
        source_vals = get_enabled_enum_vals(
            trt.TacticSource, lambda val: (1 << int(val)) & config.get_tactic_sources()
        )
        add_line("Tactic Sources", f"{str_from_list(source_vals)}")

    # DLA
    if using_dla:
        add_line(
            "DLA",
            f"Default Device Type: {config.default_device_type}, Core: {config.DLA_core}",
        )

    # Profiling Verbosity
    with contextlib.suppress(AttributeError):
        add_line("Profiling Verbosity", f"{config.profiling_verbosity}")

    # Optimization Profiles
    if (
        config.num_optimization_profiles > 1
    ):  # Not particularly interesting unless there are multiple.
        add_line(
            "Optimization Profiles", f"{config.num_optimization_profiles} profile(s)"
        )

    # Preview Features
    with contextlib.suppress(AttributeError):
        feature_vals = get_enabled_enum_vals(
            trt.PreviewFeature, lambda val: config.get_preview_feature(val)
        )
        if feature_vals:
            add_line("Preview Features", f"{str_from_list(feature_vals)}")

    # Calibrator
    if hasattr(config, "int8_calibrator") and config.int8_calibrator:
        add_line("Calibrator", f"{config.int8_calibrator}")

    # Quantization Flags
    with contextlib.suppress(AttributeError):
        quantization_flags = get_enabled_enum_vals(
            trt.QuantizationFlag, lambda val: config.get_quantization_flag(val)
        )
        if quantization_flags:
            add_line("Quantization Flags", f"{str_from_list(quantization_flags)}")

    return "\n".join(lines)


def check_profile(profile):
    if not bool(profile):
        G_LOGGER.critical(
            f"Profile is not valid, please provide profile data.\nNote: profile was: {profile}"
        )
    return profile


def str_from_tensor(tensor, is_shape_tensor):
    ret = "Input "
    if is_shape_tensor:
        ret += "shape-tensor"
    else:
        ret += "tensor"
    ret += f": {tensor.name} (dtype={tensor.dtype}, shape={tensor.shape})"
    return ret


# Note: When `force_opt_shapes=True` this method is treated as being specific to calibration.
def get_input_metadata_from_network(network, profile, force_opt_shapes=None):
    """
    Returns metadata about the inputs of a network, referring to the values
    set in a profile for dynamic shapes.

    Args:
        network (trt.INetworkDefinition):
                The network the profile applies to.
        profile (trt.IOptimizationProfile):
                The profile from which to retrieve input metadata.

        force_opt_shapes (bool):
                Whether to ignore the minimum and maximum shapes in the profile
                and always use OPT shapes.
                Defaults to False.

    Returns:
        TensorMetadata:
                A mapping of input names to their types and shapes.
                Shapes are retrieved from the OPT values in the profile.

    Raises:
        PolygraphyException:
                If the network has dynamic shapes or shape tensor inputs but no profile
                was provided.
    """
    force_opt_shapes = util.default(force_opt_shapes, False)

    input_metadata = TensorMetadata()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        # Only access the profile if we actually need to.
        # This way, this method works with static networks even without a profile set.
        min_shape = None
        max_shape = None
        opt_shape = tensor.shape
        if tensor.is_shape_tensor or util.is_shape_dynamic(tensor.shape):
            if tensor.is_shape_tensor:
                min_shape, opt_shape, max_shape = profile.get_shape_input(tensor.name)
            else:
                min_shape, opt_shape, max_shape = profile.get_shape(tensor.name)

            if force_opt_shapes and tuple(min_shape) != tuple(max_shape):
                G_LOGGER.warning(
                    "TensorRT does not currently support using dynamic shapes during calibration. "
                    "The `OPT` shapes from the calibration profile will be used for tensors with dynamic shapes. "
                    "Calibration data is expected to conform to those shapes. ",
                    mode=LogMode.ONCE,
                )

        input_metadata.add(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=opt_shape if force_opt_shapes else tensor.shape,
            min_shape=None if force_opt_shapes else min_shape,
            max_shape=None if force_opt_shapes else max_shape,
        )
    return input_metadata


# calib_profile parameter is used to bypass `get_calibration_profile()` to make this work on TRT 7.0 and older.
def try_setup_polygraphy_calibrator(config, network, calib_profile=None):
    """
    Tries to call setup methods specific to Polygraphy calibrators.
    Returns early if there is no calibrator or if it is not a Polygraphy calibrator.
    """
    try:
        calibrator = config.int8_calibrator
    except AttributeError:
        return
    if calibrator is None or not (
        hasattr(calibrator, "is_polygraphy_calibrator")
        and calibrator.is_polygraphy_calibrator
    ):
        # No calibrator or not a Polygraphy calibrator.
        return

    if calib_profile is None:
        try:
            calib_profile = config.get_calibration_profile()
        except AttributeError:
            G_LOGGER.extra_verbose(
                "Cannot get calibration profile on TensorRT 7.0 and older."
            )
            # Return early so we don't emit extraneous warnings on TRT 7.0 and older.
            return

    try:
        # TensorRT does not currently support shapes other than the OPT shape.
        input_metadata = get_input_metadata_from_network(
            network, calib_profile, force_opt_shapes=True
        )
    except PolygraphyException as err:
        G_LOGGER.warning(
            "Could not determine input_metadata to provide to the calibrator because no calibration profile is set. "
            "Please either set a calibration profile in the config or call `calibrator.set_input_metadata()` manually. "
            f"\nNote: Error was:\n{err}",
            mode=LogMode.ONCE,
        )
    else:
        calibrator.set_input_metadata(input_metadata)


def get_tensor_format(engine, context, name):
    try:
        return engine.get_tensor_format(name, context.active_optimization_profile)
    except TypeError:
        return engine.get_tensor_format(name)


def get_hwc_shape_from_chw(shape):
    # Move 3rd-to-last dim to last, shifting last 2 dims earlier
    return shape[:-3] + shape[-2:] + (shape[-3],)


def get_chw_shape_from_hwc(shape):
    # Move last dim to 3rd-to-last, shifting other dims later
    return shape[:-3] + (shape[-1],) + shape[-3:-1]


def get_metadata_from_engine(engine, context, mode):
    meta = TensorMetadata()
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(name) != mode:
            continue

        shape = engine.get_tensor_shape(name)
        # If the input format is HWC, make sure the input is shaped accordingly
        if get_tensor_format(engine, context, name) == trt.TensorFormat.HWC:
            shape = get_hwc_shape_from_chw(shape)

        meta.add(
            name=name,
            dtype=DataType.from_dtype(engine.get_tensor_dtype(name), "tensorrt"),
            shape=shape,
        )
    return meta


class TensorInfo:
    def __init__(self, json_path: str = None):
        self.tensors = {}
        if json_path:
            self.load_json(json_path)

    def load_json(self, json_path: str) -> None:
        data = load_json(json_path)
        if isinstance(data, RunResults):
            # Handle RunResults format
            for runner_name, iterations in data.items():
                if not iterations:
                    G_LOGGER.warning(f"No iterations found for runner: {runner_name}")
                    continue

                if len(iterations) > 1:
                    G_LOGGER.warning(
                        f"Found {len(iterations)} iterations in tensor info file, only using the first one"
                    )

                iter_data = iterations[0]
                for name, tensor in iter_data.items():
                    if not isinstance(tensor, np.ndarray):
                        tensor = np.array(tensor)

                    self.tensors[name] = {
                        "min": float(np.min(tensor)),
                        "max": float(np.max(tensor)),
                        "avg": float(np.mean(tensor)),
                    }
                break  # Only use first runner
        else:
            G_LOGGER.warning(f"Unsupported tensor info format: {json_path}")

    def get_tensor_statistics(self, tensor_name: str) -> str:
        tensor = self.tensors.get(tensor_name)
        if not tensor:
            return ""
        return f", min={tensor['min']:.2f}, max={tensor['max']:.2f}, avg={tensor['avg']:.2f}"


def str_from_engine(
    engine, context, show_layers=None, show_attrs=None, combine_tensor_info=None
):
    from polygraphy.tools.inspect.subtool.model.extractors import (
        graph_data_from_trt_engine,
    )
    from polygraphy.tools.inspect.subtool.model.text import str_from_graph_data

    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)

    graph_data = graph_data_from_trt_engine(
        engine, context, combine_tensor_info=combine_tensor_info
    )
    return str_from_graph_data(
        graph_data,
        show_layers=show_layers,
        show_attrs=show_attrs,
    )


def _get_array_on_gpu(arr, name, device_buffers, stream=None):
    """
    Copies the provided array to GPU memory if needed and returns a pointer
    to the GPU memory. If sufficient GPU memory has not been allocated for
    the array in ``device_buffers``, this function will allocate new memory.

    Args:
        arr (Union[DeviceView, numpy.ndarray, torch.Tensor]): The array.
        name (str): The name of the array.
        device_buffers (Dict[str, DeviceArray]):
                A mapping of names to DeviceArrays.
        stream (cuda.Stream): The CUDA stream to use.

    Returns:
        int: A pointer to the GPU memory.
    """
    if util.array.is_on_gpu(arr):
        return util.array.data_ptr(arr)

    arr = util.array.make_contiguous(arr)

    shape = (util.array.nbytes(arr),)
    if name not in device_buffers:
        # We intentionally don't set the shape here so that it's treated as a scalar and therefore has
        # some memory allocated. Otherwise, if there's an empty tensor, we won't allocate anything
        # and the device pointer will be 0 (i.e. nullptr), which TensorRT will complain about.
        device_buffers[name] = cuda.DeviceArray.raw()

    device_buffers[name].resize(shape)
    device_buffers[name].copy_from(util.array.view(arr, DataType.UINT8, shape), stream)
    return device_buffers[name].ptr


def inherit_and_extend_docstring(parent_method):
    """
    Decorator to inherit and extend docstrings from parent class methods.

    Combines the parent method's description and Args with the child method's
    description and Args, preserving proper formatting for Sphinx documentation.

    Args:
        parent_method: The parent method to inherit docstring from

    Returns:
        Decorator function that combines parent and child docstrings
    """

    def decorator(child_method):
        parent_doc = parent_method.__doc__ or ""
        child_doc = child_method.__doc__ or ""

        if not parent_doc:
            return child_method
        if not child_doc:
            child_method.__doc__ = parent_doc
            return child_method

        def extract_description_and_args(docstring):
            """Extract description and Args section from a docstring."""
            desc = re.split(r"\n\s*Args:", docstring, 1)[0].strip()
            args_match = re.search(
                r"\n\s*Args:\s*\n(.*?)(?=\n\s*[A-Z][a-z]*:|\Z)", docstring, re.DOTALL
            )
            args = args_match.group(1).rstrip() if args_match else ""
            return desc, args

        # Extract components from both docstrings
        parent_desc, parent_args = extract_description_and_args(parent_doc)
        child_desc, child_args = extract_description_and_args(child_doc)

        # Combine descriptions
        combined_desc = f"{parent_desc}\n\n{child_desc}" if child_desc else parent_desc

        # Combine Args sections
        args_parts = [
            args for args in [parent_args, child_args] if args
        ]  # Filter for non-empty argument strings
        combined_doc = (
            f"{combined_desc}\n\nArgs:\n" + "\n".join(args_parts)
            if args_parts
            else combined_desc
        )

        child_method.__doc__ = combined_doc
        return child_method

    return decorator


def convert_linear_to_vectorized_format(context, name, input_arr):
    dims = context.get_tensor_shape(name)
    vector_dim = context.engine.get_tensor_vectorized_dim(name)
    spv = context.engine.get_tensor_components_per_element(name)

    # For DLA, format is typically (N, C / spv, H, W, spv).
    # Can be performed by reshape then transpose.
    # If vector dim doesn't divide evenly into channel dimension, we need to pad.

    assert (
        vector_dim == 1 and len(dims) == 4
    ), "Only 4D channel vectorization is supported"

    if dims[vector_dim] < spv or dims[vector_dim] % spv != 0:
        pad_value = spv - (dims[vector_dim] % spv)
        input_arr = util.array.pad(input_arr, [(0, pad_value)], [vector_dim])
        dims = input_arr.shape

    # Reshape to vectorized dimensions then transpose.
    vectorized_dims = [dims[0], math.ceil(dims[1] / spv), spv, dims[2], dims[3]]

    vectorized_arr = util.array.reshape(input_arr, vectorized_dims)
    vectorized_arr = util.array.transpose(vectorized_arr, (0, 1, 3, 4, 2))
    return FormattedArray(vectorized_arr, shape=vectorized_dims)


def convert_vectorized_to_linear_format(context, name, raw_array):
    dims = context.get_tensor_shape(name)
    vector_dim = context.engine.get_tensor_vectorized_dim(name)
    spv = context.engine.get_tensor_components_per_element(name)

    # Raw array is provided as bytes, so convert to target datatype.
    dtype = DataType.to_dtype(
        DataType.from_dtype(
            context.engine.get_tensor_dtype(name), source_module="tensorrt"
        ),
        "numpy",
    )

    assert (
        vector_dim == 1 and len(dims) == 4
    ), "Only 4D channel vectorization is supported"

    # Get view of raw-byte array as specified dtype. Note view_arr is flattened at this point.
    view_arr = util.array.view(raw_array, dtype, len(raw_array) // dtype.itemsize)

    # Original vectorized dimensions
    vectorized_dims = [dims[0], math.ceil(dims[1] / spv), dims[2], dims[3], spv]

    if dims[vector_dim] % spv != 0:
        # Figure out how much padding was added
        padded_dims = list(dims)
        padded_dims[vector_dim] = spv * math.ceil(dims[vector_dim] / spv)

        # Reshape to vectorized dimensions, transpose, and reshape to padded dimensions
        linear_arr = util.array.reshape(view_arr, vectorized_dims)
        linear_arr = util.array.transpose(linear_arr, (0, 1, 4, 2, 3))
        linear_arr = util.array.reshape(linear_arr, padded_dims)
        # We need to slice off the padding.
        slice_end = spv - (dims[1] % spv)
        linear_arr = linear_arr[:, 0:-slice_end, :, :]
        return linear_arr

    # There's no padding so we can just reshape / transpose directly.
    else:
        linear_arr = util.array.reshape(view_arr, dims)
        linear_arr = util.array.reshape(linear_arr, vectorized_dims)
        linear_arr = util.array.transpose(linear_arr, (0, 1, 4, 2, 3))
        linear_arr = util.array.reshape(linear_arr, dims)
        return linear_arr
