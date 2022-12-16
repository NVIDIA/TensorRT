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
import ctypes
import time

from polygraphy import constants, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.config import CreateConfig
from polygraphy.logger import G_LOGGER

trt = mod.lazy_import("tensorrt")
gs = mod.lazy_import("onnx_graphsurgeon")
np = mod.lazy_import("numpy")


@mod.export(funcify=True)
class LoadPlugins(BaseLoader):
    """
    A passthrough loader that loads plugins from the specified paths.
    Passthrough here means that it can be used to wrap any other loader. The purpose of wrapping
    another loader is that you can control the order of execution when lazily evaluating.

    For immediate evaluation, use `load_plugins` instead:
    ::

        load_plugins(plugins=["/path/to/my/plugin.so", "/path/to/my/other_plugin.so"])
    """

    def __init__(self, plugins=None, obj=None):
        """
        Loads plugins from the specified paths.

        Args:
            plugins (List[str]):
                    A list of paths to plugin libraries to load before inference.
            obj :
                    An object or callable to return or call respectively.
                    If ``obj`` is callable, extra parameters will be forwarded to ``obj``.
                    If ``obj`` is not callable, it will be returned.
        """
        self.plugins = util.default(plugins, [])
        self.obj = obj

    def call_impl(self, *args, **kwargs):
        """
        Returns:
            object:
                    The provided ``obj`` argument, or its return value if it is
                    callable. Returns ``None`` if ``obj`` was not set.
        """
        for plugin in self.plugins:
            G_LOGGER.info(f"Loading plugin library: {plugin}")
            ctypes.CDLL(plugin)

        ret, _ = util.invoke_if_callable(self.obj, *args, **kwargs)
        return ret


@mod.export(funcify=True)
class CreateNetwork(BaseLoader):
    """
    Functor that creates an empty TensorRT network.
    """

    def __init__(self, explicit_batch=None):
        """
        Creates an empty TensorRT network.

        Args:
            explicit_batch (bool):
                    Whether to create the network with explicit batch mode.
                    Defaults to True.
        """
        self.explicit_batch = util.default(explicit_batch, True)

    def call_impl(self):
        """
        Returns:
            (trt.Builder, trt.INetworkDefinition): The builder and empty network.
        """
        with util.FreeOnException([trt.Builder(trt_util.get_trt_logger())]) as (builder,):
            network_flags = 0
            if self.explicit_batch:
                network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags=network_flags)
            if network is None:
                G_LOGGER.critical("Invalid network. See logging output above for details.")
            return builder, network


class BaseNetworkFromOnnx(BaseLoader):
    def __init__(self, explicit_batch=None):
        """
        Args:
            explicit_batch (bool):
                    Whether to create the network with explicit batch mode.
                    Defaults to True.
        """
        self.explicit_batch = util.default(explicit_batch, True)

    def call_impl(self):
        with util.FreeOnException(create_network(explicit_batch=self.explicit_batch)) as (builder, network):
            parser = trt.OnnxParser(network, trt_util.get_trt_logger())
            return builder, network, parser


@mod.export(funcify=True)
class NetworkFromOnnxBytes(BaseNetworkFromOnnx):
    """
    Functor that parses an ONNX model to create a trt.INetworkDefinition.
    """

    def __init__(self, model_bytes):
        """
        Parses an ONNX model.

        Args:
            model_bytes (Union[bytes, Callable() -> bytes]):
                    A serialized ONNX model or a callable that returns one.
        """
        super().__init__()
        self._model_bytes = model_bytes

    def call_impl(self):
        """
        Returns:
            (trt.IBuilder, trt.INetworkDefinition, trt.OnnxParser):
                    A TensorRT network, as well as the builder used to create it, and the parser
                    used to populate it.
        """
        with util.FreeOnException(super().call_impl()) as (builder, network, parser):
            success = parser.parse(util.invoke_if_callable(self._model_bytes)[0])
            trt_util.check_onnx_parser_errors(parser, success)
            return builder, network, parser


@mod.export(funcify=True)
class NetworkFromOnnxPath(BaseNetworkFromOnnx):
    """
    Functor that parses an ONNX model to create a trt.INetworkDefinition.
    This loader supports models with weights stored in an external location.
    """

    def __init__(self, path):
        """
        Parses an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.
        """
        super().__init__()
        self.path = path

    def call_impl(self):
        """
        Returns:
            (trt.IBuilder, trt.INetworkDefinition, trt.OnnxParser):
                    A TensorRT network, as well as the builder used to create it, and the parser
                    used to populate it.
        """
        path = util.invoke_if_callable(self.path)[0]
        if mod.version(trt.__version__) >= mod.version("7.1"):
            with util.FreeOnException(super().call_impl()) as (builder, network, parser):
                # We need to use parse_from_file for the ONNX parser to keep track of the location of the ONNX file for
                # potentially parsing any external weights.
                success = parser.parse_from_file(path)
                trt_util.check_onnx_parser_errors(parser, success)
                return builder, network, parser
        else:
            from polygraphy.backend.common import bytes_from_path

            return network_from_onnx_bytes(bytes_from_path(path))


@mod.export(funcify=True)
class PostprocessNetwork(BaseLoader):
    """
    [EXPERIMENTAL] Functor that applies a given post-processing function to a TensorRT ``INetworkDefinition``.
    """

    def __init__(self, network, func, name=None):
        """
        Applies a given post-processing function to a TensorRT ``INetworkDefinition``.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.
            func (Callable[[trt.INetworkDefinition], None])
                    A callable which accepts a named `network` argument.  `PostprocessNetwork` will pass in the parsed network via this argument, which can then be modified by the callable.
            name (Optional[str])
                    The name of this postprocessing step, used for logging purposes.
        """

        self._network = network

        # Sanity-check that the function passed in is callable
        if not callable(func):
            G_LOGGER.critical(f"Object {func} (of type {type(func)}) is not a callable.")

        try:
            func_name = func.__name__
        except:
            func_name = str(func)

        self._func = func
        self.name = util.default(name, func_name)

    def call_impl(self):
        """
        Returns:
            Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]:
                    The modified network along with the builder and parser if provided.
        """
        ret, owns_network = util.invoke_if_callable(self._network)
        builder, network, parser = util.unpack_args(ret, num=3)

        G_LOGGER.verbose(f"Executing postprocessing step [{self.name}]")

        with contextlib.ExitStack() as stack:
            if owns_network:
                stack.enter_context(util.FreeOnException([builder, network, parser]))

            self._func(network=network)

            if parser is None:
                return builder, network
            return builder, network, parser


@mod.export(funcify=True)
class ModifyNetworkOutputs(PostprocessNetwork):
    """
    Functor that modifies outputs in a TensorRT ``INetworkDefinition``.
    """

    @staticmethod
    def _apply(network, outputs, exclude_outputs):
        if outputs == constants.MARK_ALL:
            trt_util.mark_layerwise(network)
        elif outputs is not None:
            trt_util.mark_outputs(network, outputs)
        if exclude_outputs is not None:
            trt_util.unmark_outputs(network, exclude_outputs)
        return network

    def __init__(self, network, outputs=None, exclude_outputs=None):
        """
        Modifies outputs in a TensorRT ``INetworkDefinition``.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.

            outputs (Sequence[str]):
                    Names of tensors to mark as outputs. If provided, this will override the outputs
                    already marked in the network.
                    If a value of `constants.MARK_ALL` is used instead of a list, all tensors in the network are marked.
            exclude_outputs (Sequence[str]):
                    Names of tensors to exclude as outputs. This can be useful in conjunction with
                    ``outputs=constants.MARK_ALL`` to omit outputs.
        """
        func = lambda network: ModifyNetworkOutputs._apply(network, outputs, exclude_outputs)
        super().__init__(network, func, "ModifyNetworkOutputs")


@mod.export(funcify=True)
class SetLayerPrecisions(PostprocessNetwork):
    """
    Functor that sets layer precisions in a TensorRT ``INetworkDefinition``.
    """

    @staticmethod
    def _apply(network, layer_precisions):
        util.check_sequence_contains(
            [layer.name for layer in network],
            layer_precisions.keys(),
            name="the network",
            items_name="layers",
            check_extra=False,
        )

        for layer in network:
            if layer.name in layer_precisions:
                layer.precision = layer_precisions[layer.name]
        return network


    def __init__(self, network, layer_precisions):
        """
        Sets layer precisions in a TensorRT ``INetworkDefinition``.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.
            layer_precisions (Dict[str, trt.DataType]):
                    A mapping of layer names to their desired compute precision.
        """
        func = lambda network: SetLayerPrecisions._apply(network, layer_precisions)
        super().__init__(network, func, "SetLayerPrecisions")


@mod.export(funcify=True)
class SetTensorDatatypes(PostprocessNetwork):
    """
    Functor that sets tensor datatypes in a TensorRT ``INetworkDefinition``.
    """

    @staticmethod
    def _apply(network, tensor_datatypes):
        tensor_map = trt_util.get_all_tensors(network)
        util.check_sequence_contains(
            tensor_map.keys(),
            tensor_datatypes.keys(),
            name="the network",
            items_name="tensors",
            check_extra=False,
        )

        for name, dtype in tensor_datatypes.items():
            tensor_map[name].dtype = dtype
        return network

    def __init__(self, network, tensor_datatypes):
        """
        Sets tensor datatypes in a TensorRT ``INetworkDefinition``.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.
            tensor_datatypes (Dict[str, trt.DataType]):
                    A mapping of tensor names to their desired data types.
        """
        func = lambda network: SetTensorDatatypes._apply(network, tensor_datatypes)
        super().__init__(network, func, "SetTensorDatatypes")


@mod.export(funcify=True)
class SetTensorFormats(PostprocessNetwork):
    """
    Functor that sets tensor formats in a TensorRT ``INetworkDefinition``.
    """

    @staticmethod
    def _apply(network, tensor_formats):
        tensor_map = trt_util.get_all_tensors(network)
        util.check_sequence_contains(
            tensor_map.keys(),
            tensor_formats.keys(),
            name="the network",
            items_name="tensors",
            check_extra=False,
        )

        for name, formats in tensor_formats.items():
            mask = 0
            for format in formats:
                mask |= 1 << int(format)
            tensor_map[name].allowed_formats = mask
        return network



    def __init__(self, network, tensor_formats):
        """
        Sets tensor formats in a TensorRT ``INetworkDefinition``.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.
            tensor_formats (Dict[str, List[trt.TensorFormat]]):
                    A mapping of tensor names to their allowed formats.
        """
        func = lambda network: SetTensorFormats._apply(network, tensor_formats)
        super().__init__(network, func, "SetTensorFormats")


@mod.export(funcify=True)
class EngineBytesFromNetwork(BaseLoader):
    """
    Functor that uses a TensorRT ``INetworkDefinition`` to build a serialized engine.
    """

    def __init__(self, network, config=None, save_timing_cache=None):
        """
        Builds and serializes TensorRT engine.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.


            config (Callable(trt.Builder, trt.INetworkDefinition) -> trt.IBuilderConfig):
                    A TensorRT builder configuration or a callable that returns one. If not supplied,
                    a `CreateConfig` instance with default parameters is used.
            save_timing_cache (Union[str, file-like]):
                    A path or file-like object at which to save a tactic timing cache.
                    Any existing cache will be appended to.
                    If a path is provided, the file will be locked for exclusive access to prevent
                    multiple processes from attempting to update the timing cache at the same time.
        """
        self._network = network
        self._config = util.default(config, CreateConfig())
        self.timing_cache_path = save_timing_cache

    def call_impl(self):
        """
        Returns:
            bytes: The serialized engine that was created.
        """
        # If network is a callable, then we own its return value
        ret, owns_network = util.invoke_if_callable(self._network)
        builder, network, parser = util.unpack_args(ret, num=3)

        if builder is None or network is None:
            G_LOGGER.critical(
                f"Expected to recevie a (builder, network) tuple for the `network` parameter, but received: ({builder}, {network})"
            )

        with contextlib.ExitStack() as stack:
            if owns_network:
                stack.enter_context(builder)
                stack.enter_context(network)
                if parser is not None:
                    stack.enter_context(parser)
            else:
                provided = "Builder and Network" if parser is None else "Builder, Network, and Parser"
                G_LOGGER.verbose(
                    f"{provided} were provided directly instead of via a Callable. This loader will not assume ownership. Please ensure that they are freed."
                )

            config, owns_config = util.invoke_if_callable(self._config, builder, network)
            if owns_config:
                stack.enter_context(config)
            else:
                G_LOGGER.verbose(
                    "Builder configuration was provided directly instead of via a Callable. This loader will not assume "
                    "ownership. Please ensure it is freed."
                )

            trt_util.try_setup_polygraphy_calibrator(config, network)

            G_LOGGER.super_verbose(
                lambda: (
                    "Displaying TensorRT Network:\n"
                    + trt_util.str_from_network(
                        network,
                        show_layers=True,
                        show_attrs=True,
                        show_weights=G_LOGGER.module_severity.get(G_LOGGER.module_path(__file__))
                        <= G_LOGGER.ULTRA_VERBOSE,
                    )
                )
            )

            G_LOGGER.start(f"Building engine with configuration:\n{trt_util.str_from_config(config)}")

            start_time = time.time()
            try:
                engine_bytes = builder.build_serialized_network(network, config)
            except AttributeError:
                engine = builder.build_engine(network, config)
                if not engine:
                    G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")
                stack.enter_context(engine)
                engine_bytes = engine.serialize()
            end_time = time.time()

            if not engine_bytes:
                G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")

            G_LOGGER.finish(f"Finished engine building in {end_time - start_time:.3f} seconds")

            if self.timing_cache_path:
                try:
                    timing_cache = config.get_timing_cache()
                except AttributeError:
                    trt_util.fail_unavailable("save_timing_cache in EngineBytesFromNetwork")

                with util.LockFile(self.timing_cache_path):
                    try:
                        prev_cache = config.create_timing_cache(util.load_file(self.timing_cache_path))
                    except:
                        prev_cache = None

                    if timing_cache:
                        if prev_cache is not None:
                            combine_success = timing_cache.combine(prev_cache, ignore_mismatch=True)
                            if not combine_success:
                                G_LOGGER.warning("Could not combine old timing cache into current timing cache")

                        with timing_cache.serialize() as buffer:
                            util.save_file(buffer, self.timing_cache_path, description="tactic timing cache")

            return engine_bytes


@mod.export(funcify=True)
class EngineFromNetwork(EngineBytesFromNetwork):
    """
    Similar to EngineBytesFromNetwork, but returns an ICudaEngine instance
    instead of a serialized engine.
    """

    def call_impl(self):
        """
        Returns:
            trt.ICudaEngine: The engine that was created.
        """
        # We do not invoke super().call_impl here because we would otherwise be responsible
        # for freeing it's return values.
        return engine_from_bytes(super().call_impl)


@mod.export(funcify=True)
class EngineFromBytes(BaseLoader):
    """
    Functor that deserializes an engine from a buffer.
    """

    def __init__(self, serialized_engine):
        """
        Deserializes an engine from a buffer.

        Args:
            serialized_engine (Union[Union[str, bytes], Callable() -> Union[str, bytes]]):
                    The serialized engine bytes  or a callable that returns them.
        """
        self._serialized_engine = serialized_engine

    def call_impl(self):
        """
        Returns:
            trt.ICudaEngine: The deserialized engine.
        """
        buffer, owns_buffer = util.invoke_if_callable(self._serialized_engine)

        trt.init_libnvinfer_plugins(trt_util.get_trt_logger(), "")
        with contextlib.ExitStack() as stack, trt.Runtime(trt_util.get_trt_logger()) as runtime:
            if owns_buffer:
                try:
                    buffer.__enter__  # IHostMemory is freed only in __exit__
                except AttributeError:
                    pass
                else:
                    stack.enter_context(buffer)

            engine = runtime.deserialize_cuda_engine(buffer)
            if not engine:
                G_LOGGER.critical("Could not deserialize engine. See log for details.")
            return engine


@mod.export(funcify=True)
class BytesFromEngine(BaseLoader):
    """
    Functor that serializes an engine.
    """

    def __init__(self, engine):
        """
        Serializes an engine.

        Args:
            engine (Union[trt.ICudaEngine, Callable() -> trt.ICudaEngine]):
                    An engine or a callable that returns one.
        """
        self._engine = engine

    def call_impl(self):
        """
        Returns:
            bytes: The serialized engine.
        """
        engine, owns_engine = util.invoke_if_callable(self._engine)

        with contextlib.ExitStack() as stack:
            if owns_engine:
                stack.enter_context(util.FreeOnException([engine]))

            with engine.serialize() as buffer:
                return bytes(buffer)


@mod.export(funcify=True)
class SaveEngine(BaseLoader):
    """
    Functor that saves an engine to the provided path.
    """

    def __init__(self, engine, path):
        """
        Saves an engine to the provided path.

        Args:
            engine (Union[trt.ICudaEngine, Callable() -> trt.ICudaEngine]):
                    An engine or a callable that returns one.


            path (str): The path at which to save the engine.
        """
        self._engine = engine
        self.path = path

    def call_impl(self):
        """
        Returns:
            trt.ICudaEngine: The engine that was saved.
        """
        engine, owns_engine = util.invoke_if_callable(self._engine)

        with contextlib.ExitStack() as stack:
            if owns_engine:
                stack.enter_context(util.FreeOnException([engine]))

            util.save_file(contents=bytes_from_engine(engine), dest=self.path, description="engine")
            return engine


@mod.export(funcify=True)
class OnnxLikeFromNetwork(BaseLoader):
    """
    Functor that creates an ONNX-like, but **not** valid ONNX, model based on a TensorRT network.
    """

    def __init__(self, network) -> None:
        """
        [HIGHLY EXPERIMENTAL] Creates an ONNX-like, but **not** valid ONNX, model from a TensorRT network.
        This uses the ONNX format, but generates nodes that are **not** valid ONNX operators.
        Hence, this should be used **only** for visualization or debugging purposes.

        The resulting model does **not** include enough information to faithfully reconstruct the TensorRT network,
        but does preserve the structure of the network and many of the layer parameters.

        Args:
            network (Union[Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]], Callable() -> Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser or a callable that returns one.
                    To omit the parser, return a tuple containing just the builder and network.
        """
        self._network = network

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX-like, but **not** valid ONNX, representation of the TensorRT network.
        """
        ret, owns_network = util.invoke_if_callable(self._network)
        builder, network, parser = util.unpack_args(ret, num=3)

        if builder is None or network is None:
            G_LOGGER.critical(
                f"Expected to recevie a (builder, network) tuple for the `network` parameter, but received: ({builder}, {network})"
            )

        with contextlib.ExitStack() as stack:
            if owns_network:
                stack.enter_context(builder)
                stack.enter_context(network)
                if parser is not None:
                    stack.enter_context(parser)

            tensor_map = {}

            def tensors_from_names_meta(names, meta):
                nonlocal tensor_map
                tensors = []
                for name in names:
                    if name not in tensor_map:
                        dtype, shape = meta[name]
                        tensor_map[name] = gs.Variable(name=name, dtype=dtype, shape=shape)
                    tensors.append(tensor_map[name])
                return tensors

            nodes = []
            graph_inputs = tensors_from_names_meta(*trt_util.get_network_input_names_meta(network))
            graph_outputs = tensors_from_names_meta(*trt_util.get_network_output_names_meta(network))

            LAYER_TYPE_CLASS_MAPPING = trt_util.get_layer_class_mapping()

            for layer in network:
                op_name = layer.type.name
                if layer.type in LAYER_TYPE_CLASS_MAPPING:
                    layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

                node_inputs = tensors_from_names_meta(*trt_util.get_layer_input_names_meta(layer))
                node_outputs = tensors_from_names_meta(*trt_util.get_layer_output_names_meta(layer))
                attrs = {}
                attr_names = trt_util.get_layer_attribute_names(layer)
                for name in attr_names:
                    with G_LOGGER.verbosity():
                        attr = getattr(layer, name)

                    if util.is_sequence(attr) or any(isinstance(attr, cls) for cls in [trt.Dims, trt.Permutation]):
                        try:
                            attr = list(attr)
                        except ValueError:  # Invalid dims
                            attr = []

                    if hasattr(attr, "__entries"):  # TensorRT Enums
                        attr = attr.name

                    if isinstance(attr, trt.ILoop):
                        attr = attr.name

                    VALID_TYPES = [np.ndarray, list, int, str, bool, float]
                    if not any(isinstance(attr, cls) for cls in VALID_TYPES):
                        G_LOGGER.internal_error(
                            f"Unknown type: {type(attr)} for layer attribute: {attr}.\nNote: Layer was: {layer}"
                        )
                        try:
                            attr = str(attr)
                        except:
                            attr = "<error during conversion>"

                    attrs[name] = attr

                nodes.append(
                    gs.Node(name=layer.name, op=op_name, attrs=attrs, inputs=node_inputs, outputs=node_outputs)
                )

            graph = gs.Graph(name=network.name, inputs=graph_inputs, outputs=graph_outputs, nodes=nodes)

            return gs.export_onnx(graph)
