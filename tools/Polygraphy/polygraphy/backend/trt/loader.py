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
import copy
import ctypes
import time

from polygraphy import constants, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.profile import Profile
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
            obj (object):
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
            G_LOGGER.info("Loading plugin library: {:}".format(plugin))
            ctypes.CDLL(plugin)

        ret, _ = util.invoke_if_callable(self.obj, *args, **kwargs)
        return ret


@mod.export(funcify=True)
class CreateNetwork(BaseLoader):
    """
    Functor that creates an empty TensorRT network.
    """

    def __init__(self, explicit_precision=None, explicit_batch=None):
        """
        Creates an empty TensorRT network.

        Args:
            explicit_precision (bool):
                    Whether to create the network with explicit precision enabled. Defaults to False
            explicit_batch (bool):
                    Whether to create the network with explicit batch mode. Defaults to True.
        """
        self.explicit_precision = util.default(explicit_precision, False)
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
            if self.explicit_precision:
                network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
            network = builder.create_network(flags=network_flags)
            if network is None:
                G_LOGGER.critical("Invalid network. See logging output above for details.")
            return builder, network


class BaseNetworkFromOnnx(BaseLoader):
    def __init__(self, explicit_precision, explicit_batch=None):
        """
        Args:
            explicit_precision (bool): Whether to create the network with explicit precision enabled.
        """
        self.explicit_precision = util.default(explicit_precision, False)
        self.explicit_batch = util.default(explicit_batch, True)

    def call_impl(self):
        with util.FreeOnException(
            create_network(explicit_precision=self.explicit_precision, explicit_batch=self.explicit_batch)
        ) as (builder, network):
            parser = trt.OnnxParser(network, trt_util.get_trt_logger())
            return builder, network, parser


@mod.export(funcify=True)
class NetworkFromOnnxBytes(BaseNetworkFromOnnx):
    """
    Functor that parses an ONNX model to create a trt.INetworkDefinition.
    """

    def __init__(self, model_bytes, explicit_precision=None):
        """
        Parses an ONNX model.

        Args:
            model_bytes (Union[bytes, Callable() -> bytes]):
                    A serialized ONNX model or a callable that returns one.
            explicit_precision (bool): Whether to construct the TensorRT network with explicit precision enabled.
        """
        super().__init__(explicit_precision)
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

    def __init__(self, path, explicit_precision=None):
        """
        Parses an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.
        """
        super().__init__(explicit_precision)
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

            return network_from_onnx_bytes(bytes_from_path(path), self.explicit_precision)


@mod.export(funcify=True)
class ModifyNetworkOutputs(BaseLoader):
    """
    Functor that modifies outputs in a TensorRT ``INetworkDefinition``.
    """

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
        self._network = network
        self.outputs = outputs
        self.exclude_outputs = exclude_outputs

    def call_impl(self):
        """
        Returns:
            trt.INetworkDefinition: The modified network.
        """
        ret, owns_network = util.invoke_if_callable(self._network)
        builder, network, parser = util.unpack_args(ret, num=3)

        with contextlib.ExitStack() as stack:
            if owns_network:
                stack.enter_context(util.FreeOnException([builder, network, parser]))

            if self.outputs == constants.MARK_ALL:
                trt_util.mark_layerwise(network)
            elif self.outputs is not None:
                trt_util.mark_outputs(network, self.outputs)

            if self.exclude_outputs is not None:
                trt_util.unmark_outputs(network, self.exclude_outputs)

            if parser is None:
                return builder, network
            return builder, network, parser


@mod.export(funcify=True)
class CreateConfig(BaseLoader):
    """
    Functor that creates a TensorRT IBuilderConfig.
    """

    def __init__(
        self,
        max_workspace_size=None,
        tf32=None,
        fp16=None,
        int8=None,
        profiles=None,
        calibrator=None,
        strict_types=None,
        load_timing_cache=None,
        algorithm_selector=None,
        sparse_weights=None,
        tactic_sources=None,
        restricted=None,
        use_dla=None,
        allow_gpu_fallback=None,
    ):
        """
        Creates a TensorRT IBuilderConfig that can be used by EngineFromNetwork.

        Args:
            max_workspace_size (int):
                    The maximum workspace size, in bytes, when building the engine.
                    Defaults to 16 MiB.
            tf32 (bool):
                    Whether to build the engine with TF32 precision enabled.
                    Defaults to False.
            fp16 (bool):
                    Whether to build the engine with FP16 precision enabled.
                    Defaults to False.
            int8 (bool):
                    Whether to build the engine with INT8 precision enabled.
                    Defaults to False.
            profiles (List[Profile]):
                    A list of optimization profiles to add to the configuration. Only needed for
                    networks with dynamic input shapes. If this is omitted for a network with
                    dynamic shapes, a default profile is created, where dynamic dimensions are
                    replaced with Polygraphy's DEFAULT_SHAPE_VALUE (defined in constants.py).
                    A partially populated profile will be automatically filled using values from ``Profile.fill_defaults()``
                    See ``Profile`` for details.
            calibrator (trt.IInt8Calibrator):
                    An int8 calibrator. Only required in int8 mode when
                    the network does not have explicit precision. For networks with
                    dynamic shapes, the last profile provided (or default profile if
                    no profiles are provided) is used during calibration.
            strict_types (bool):
                    Whether to enable strict types in the builder. This will constrain the builder from
                    using data types other than those specified in the network.
                    Defaults to False.
            load_timing_cache (Union[str, file-like]):
                    A path or file-like object from which to load a tactic timing cache.
                    Providing a tactic timing cache can speed up the engine building process.
                    Caches can be generated while building an engine with, for example, EngineFromNetwork.
            algorithm_selector (trt.IAlgorithmSelector):
                    An algorithm selector. Allows the user to control how tactics are selected
                    instead of letting TensorRT select them automatically.
            sparse_weights (bool):
                    Whether to enable optimizations for sparse weights.
                    Defaults to False.
            tactic_sources (List[trt.TacticSource]):
                    The tactic sources to enable. This controls which libraries (e.g. cudnn, cublas, etc.)
                    TensorRT is allowed to load tactics from.
                    Use an empty list to disable all tactic sources.
                    Defaults to TensorRT's default tactic sources.
            restricted (bool):
                    Whether to enable safety scope checking in the builder. This will check if the network
                    and builder configuration are compatible with safety scope.
                    Defaults to False.
            use_dla (bool):
                    [EXPERIMENTAL] Whether to enable DLA as the default device type.
                    Defaults to False.
            allow_gpu_fallback (bool):
                    [EXPERIMENTAL] When DLA is enabled, whether to allow layers to fall back to GPU if they cannot be run on DLA.
                    Has no effect if DLA is not enabled.
                    Defaults to False.
        """
        self.max_workspace_size = util.default(max_workspace_size, 1 << 24)
        self.tf32 = util.default(tf32, False)
        self.fp16 = util.default(fp16, False)
        self.int8 = util.default(int8, False)
        self.profiles = util.default(profiles, [Profile()])
        self.calibrator = calibrator
        self.strict_types = util.default(strict_types, False)
        self.restricted = util.default(restricted, False)
        self.timing_cache_path = load_timing_cache
        self.algorithm_selector = algorithm_selector
        self.sparse_weights = util.default(sparse_weights, False)
        self.tactic_sources = tactic_sources
        self.use_dla = util.default(use_dla, False)
        self.allow_gpu_fallback = util.default(allow_gpu_fallback, False)

        if self.calibrator is not None and not self.int8:
            G_LOGGER.warning(
                "A calibrator was provided to `CreateConfig`, but int8 mode was not enabled. "
                "Did you mean to set `int8=True` to enable building with int8 precision?"
            )

    def call_impl(self, builder, network):
        """
        Args:
            builder (trt.Builder):
                    The TensorRT builder to use to create the configuration.
            network (trt.INetworkDefinition):
                    The TensorRT network for which to create the config. The network is used to
                    automatically create a default optimization profile if none are provided.

        Returns:
            trt.IBuilderConfig: The TensorRT builder configuration.
        """
        with util.FreeOnException([builder.create_builder_config()]) as (config,):

            def try_run(func, name):
                try:
                    return func()
                except AttributeError:
                    trt_util.fail_unavailable("{:} in CreateConfig".format(name))

            def try_set_flag(flag_name):
                return try_run(lambda: config.set_flag(getattr(trt.BuilderFlag, flag_name)), flag_name.lower())

            with G_LOGGER.indent():
                G_LOGGER.verbose("Setting TensorRT Optimization Profiles")
                profiles = copy.deepcopy(self.profiles)
                for profile in profiles:
                    # Last trt_profile is used for set_calibration_profile.
                    trt_profile = profile.fill_defaults(network).to_trt(builder, network)
                    config.add_optimization_profile(trt_profile)
                G_LOGGER.info("Configuring with profiles: {:}".format(profiles))

            config.max_workspace_size = int(self.max_workspace_size)

            if self.strict_types:
                try_set_flag("STRICT_TYPES")

            if self.restricted:
                try_set_flag("SAFETY_SCOPE")

            if self.tf32:
                try_set_flag("TF32")
            else:  # TF32 is on by default
                with contextlib.suppress(AttributeError):
                    config.clear_flag(trt.BuilderFlag.TF32)

            if self.fp16:
                try_set_flag("FP16")

            if self.int8:
                try_set_flag("INT8")
                if not network.has_explicit_precision:
                    if self.calibrator is not None:
                        input_metadata = trt_util.get_input_metadata_from_profile(trt_profile, network)
                        with contextlib.suppress(AttributeError):  # Polygraphy calibrator has a reset method
                            self.calibrator.reset(input_metadata)
                        config.int8_calibrator = self.calibrator
                        try:
                            config.set_calibration_profile(trt_profile)
                        except:
                            G_LOGGER.extra_verbose("Cannot set calibration profile on TensorRT 7.0 and older.")
                    else:
                        G_LOGGER.warning(
                            "Network does not have explicit precision and no calibrator was provided. Please ensure "
                            "that tensors in the network have dynamic ranges set, or provide a calibrator in order to use int8 mode."
                        )

            if self.sparse_weights:
                try_set_flag("SPARSE_WEIGHTS")

            if self.use_dla:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = 0

            if self.allow_gpu_fallback:
                try_set_flag("GPU_FALLBACK")

            if self.tactic_sources is not None:
                tactic_sources_flag = 0
                for source in self.tactic_sources:
                    tactic_sources_flag |= 1 << int(source)
                try_run(lambda: config.set_tactic_sources(tactic_sources_flag), name="tactic_sources")

            try:
                if self.timing_cache_path:
                    timing_cache_data = util.load_file(self.timing_cache_path, description="tactic timing cache")
                    cache = config.create_timing_cache(timing_cache_data)
                else:
                    # Create an empty timing cache by default so it will be populated during engine build.
                    # This way, consumers of CreateConfig have the option to use the cache later.
                    cache = config.create_timing_cache(b"")
            except AttributeError:
                if self.timing_cache_path:
                    trt_util.fail_unavailable("load_timing_cache in CreateConfig")
            else:
                config.set_timing_cache(cache, ignore_mismatch=False)

            if self.algorithm_selector is not None:

                def set_algo_selector():
                    config.algorithm_selector = self.algorithm_selector

                try_run(set_algo_selector, "algorithm_selector")

            return config


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
                    Any existing cache will be overwritten. Note that if the provided config includes a tactic
                    timing cache, the data from that cache will be copied into the new cache.
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
                "Expected to recevie a (builder, network) tuple for the `network` parameter, "
                "but received: ({:}, {:})".format(builder, network)
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
                    "{:} were provided directly instead of via a Callable. This loader will not assume ownership. "
                    "Please ensure that they are freed.".format(provided)
                )

            config, owns_config = util.invoke_if_callable(self._config, builder, network)
            if owns_config:
                stack.enter_context(config)
            else:
                G_LOGGER.verbose(
                    "Builder configuration was provided directly instead of via a Callable. This loader will not assume "
                    "ownership. Please ensure it is freed."
                )

            try:
                config.int8_calibrator.__enter__  # Polygraphy calibrator frees device buffers on exit.
            except AttributeError:
                pass
            else:
                stack.enter_context(config.int8_calibrator)

            network_log_mode = "full" if G_LOGGER.severity <= G_LOGGER.ULTRA_VERBOSE else "attrs"
            G_LOGGER.super_verbose(
                lambda: ("Displaying TensorRT Network:\n" + trt_util.str_from_network(network, mode=network_log_mode))
            )

            G_LOGGER.start("Building engine with configuration:\n{:}".format(trt_util.str_from_config(config)))

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

            G_LOGGER.finish("Finished engine building in {:.3f} seconds".format(end_time - start_time))

            try:
                timing_cache = config.get_timing_cache()
            except AttributeError:
                if self.timing_cache_path:
                    trt_util.fail_unavailable("save_timing_cache in EngineBytesFromNetwork")
            else:
                if timing_cache and self.timing_cache_path:
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
                "Expected to recevie a (builder, network) tuple for the `network` parameter, "
                "but received: ({:}, {:})".format(builder, network)
            )

        with contextlib.ExitStack() as stack:
            if owns_network:
                stack.enter_context(builder)
                stack.enter_context(network)
                if parser is not None:
                    stack.enter_context(parser)

            tensor_map = {}

            def tensors_from_meta(meta):
                nonlocal tensor_map
                tensors = []
                for name, (dtype, shape) in meta.items():
                    if name not in tensor_map:
                        tensor_map[name] = gs.Variable(name=name, dtype=dtype, shape=shape)
                    tensors.append(tensor_map[name])
                return tensors

            nodes = []
            graph_inputs = tensors_from_meta(trt_util.get_network_input_metadata(network))
            graph_outputs = tensors_from_meta(trt_util.get_network_output_metadata(network))

            LAYER_TYPE_CLASS_MAPPING = trt_util.get_layer_class_mapping()

            for layer in network:
                op_name = layer.type.name
                if layer.type in LAYER_TYPE_CLASS_MAPPING:
                    layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]

                node_inputs = tensors_from_meta(trt_util.get_layer_input_metadata(layer))
                node_outputs = tensors_from_meta(trt_util.get_layer_output_metadata(layer))
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
                            "Unknown type: {:} for layer attribute: {:}.\n"
                            "Note: Layer was: {:}".format(type(attr), attr, layer)
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
