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
from collections import OrderedDict

import tensorrt as trt
from polygraphy.backend.base import BaseLoadModel
from polygraphy.backend.trt import util as trt_util
from polygraphy.common import constants
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc


class LoadPlugins(BaseLoadModel):
    def __init__(self, obj=None, plugins=None):
        """
        A passthrough loader that loads plugins from the specified paths.
        Passthrough here means that it can be used to wrap any other loader. The purpose of wrapping
        another loader is that you can control the order of execution when lazily evaluating.

        For immediate evaluation, call the loader. For example:

        ::
            # Note the `()` at the end
            LoadPlugins(plugins=["/path/to/my/plugin.so", "/path/to/my/other_plugin.so"])()

        Args:
            obj (object):
                    Any object to pass through this loader.
                    If ``obj`` is callable, parameters to __call__ will be forwarded
                    to ``obj`` when this loader is called.
            plugins (List[str]):
                    A list of paths to plugin libraries to load before inference.
        """
        self.obj = obj
        self.plugins = plugins


    def __call__(self, *args, **kwargs):
        if self.plugins:
            trt_util.load_plugins(self.plugins)

        ret, _ = misc.try_call(self.obj, *args, **kwargs)
        return ret


class CreateNetwork(BaseLoadModel):
    def __init__(self, explicit_precision=None, explicit_batch=None):
        """
        Functor that creates an empty TensorRT network.

        Args:
            explicit_precision (bool):
                    Whether to create the network with explicit precision enabled. Defaults to False
            explicit_batch (bool):
                    Whether to create the network with explicit batch mode. Defaults to True.
        """
        self.explicit_precision = misc.default_value(explicit_precision, False)
        self.explicit_batch = misc.default_value(explicit_batch, True)


    def __call__(self):
        """
        Creates an empty TensorRT network.

        Returns:
            (trt.Builder, trt.INetworkDefinition): The builder and empty network.
        """
        builder = trt.Builder(trt_util.TRT_LOGGER)
        network_flags = 0
        if self.explicit_batch:
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if self.explicit_precision:
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        network = builder.create_network(flags=network_flags)
        if network is None:
            G_LOGGER.critical("Invalid network. See logging output above for details.")
        return builder, network


class BaseNetworkFromOnnx(BaseLoadModel):
    def __init__(self, explicit_precision, explicit_batch=None):
        """
        Args:
            explicit_precision (bool): Whether to create the network with explicit precision enabled.
        """
        self.explicit_precision = misc.default_value(explicit_precision, False)
        self.explicit_batch = misc.default_value(explicit_batch, True)


    def __call__(self):
        builder, network = CreateNetwork(explicit_precision=self.explicit_precision, explicit_batch=self.explicit_batch)()
        parser = trt.OnnxParser(network, trt_util.TRT_LOGGER)
        return builder, network, parser


class NetworkFromOnnxBytes(BaseNetworkFromOnnx):
    def __init__(self, model_bytes, explicit_precision=None):
        """
        Functor that parses an ONNX model to create a trt.INetworkDefinition.

        Args:
            model_bytes (Callable() -> bytes): A loader that can supply a serialized ONNX model.
        """
        super().__init__(explicit_precision)
        self._model_bytes = model_bytes


    def __call__(self):
        """
        Parses an ONNX model.

        Returns:
            (trt.IBuilder, trt.INetworkDefinition, trt.OnnxParser):
                    A TensorRT network, as well as the builder used to create it, and the parser
                    used to populate it.
        """
        builder, network, parser = super().__call__()
        parser.parse(misc.try_call(self._model_bytes)[0])
        trt_util.check_onnx_parser_errors(parser)
        return builder, network, parser


if misc.version(trt.__version__) >= misc.version("7.1"):
    class NetworkFromOnnxPath(BaseNetworkFromOnnx):
        def __init__(self, path, explicit_precision=None):
            """
            Functor that parses an ONNX model to create a trt.INetworkDefinition.
            This loader supports models with weights stored in an external location.

            Args:
                path (str): The path from which to load the model.
            """
            super().__init__(explicit_precision)
            self.path = path


        def __call__(self):
            """
            Parses an ONNX model from a file.

            Returns:
                (trt.IBuilder, trt.INetworkDefinition, trt.OnnxParser):
                        A TensorRT network, as well as the builder used to create it, and the parser
                        used to populate it.
            """
            builder, network, parser = super().__call__()
            # We need to use parse_from_file for the ONNX parser to keep track of the location of the ONNX file for
            # potentially parsing any external weights.
            parser.parse_from_file(misc.try_call(self.path)[0])
            trt_util.check_onnx_parser_errors(parser)
            return builder, network, parser
else:
    class NetworkFromOnnxPath(NetworkFromOnnxBytes):
        def __init__(self, path, explicit_precision=None):
            """
            Functor that parses an ONNX model to create a trt.INetworkDefinition.
            This loader supports models with weights stored in an external location.

            Args:
                path (str): The path from which to load the model.
            """
            from polygraphy.backend.common import BytesFromPath
            load_model = BytesFromPath(misc.try_call(self.path)[0](self.path))
            super().__init__(load_model, explicit_precision)


class ModifyNetwork(BaseLoadModel):
    def __init__(self, network, outputs=None, exclude_outputs=None):
        """
        Functor that modifies a TensorRT ``INetworkDefinition``.

        Args:
            network (Callable() -> trt.Builder, trt.INetworkDefinition):
                    A callable capable of returning a TensorRT Builder and INetworkDefinition. The callable may
                    have at most 3 return values if another object needs to be kept alive for the duration of the network,
                    e.g., in the case of a parser. The first and second return values must
                    always be the builder and network respectively. ModifyNetwork will never take ownership of these.

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


    def __call__(self):
        """
        Modifies a TensorRT ``INetworkDefinition``.

        Returns:
            trt.INetworkDefinition: The modified network.
        """
        ret, _ = misc.try_call(self._network)
        builder, network, parser = misc.unpack_args(ret, num=3)

        if self.outputs == constants.MARK_ALL:
            trt_util.mark_layerwise(network)
        elif self.outputs is not None:
            trt_util.mark_outputs(network, self.outputs)

        if self.exclude_outputs is not None:
            trt_util.unmark_outputs(network, self.exclude_outputs)

        if parser is not None:
            return builder, network, parser
        else:
            return builder, network


class ShapeTuple(object):
    def __init__(self, min, opt, max):
        """
        Represents a set of shapes for a single binding in a profile

        Args:
            min (Tuple[int]): The minimum shape that the profile will support.
            opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
            max (Tuple[int]): The maximum shape that the profile will support.
        """
        self.min = min
        self.opt = opt
        self.max = max


    def __str__(self):
        return "(min={:}, opt={:}, max={:})".format(self.min, self.opt, self.max)


    def __repr__(self):
        return type(self).__name__ + self.__str__()


class Profile(OrderedDict):
    """
    An ordered dictionary that represents a single optimization profile that
    can be used to build an engine.

    More specifically, this is a OrderedDict[str, ShapeTuple] which maps binding
    names to a set of min/opt/max shapes.
    """
    def add(self, name, min, opt, max):
        """
        A convenience function to add shapes for a single binding.

        Args:
            name (str): The name of the binding.
            min (Tuple[int]): The minimum shape that the profile will support.
            opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
            max (Tuple[int]): The maximum shape that the profile will support.

        Returns:
            self:
                Which allows this function to be easily chained to add multiple bindings,
                e.g., Profile().add(...).add(...)
        """
        self[name] = ShapeTuple(min, opt, max)
        return self

    def __getitem__(self, key):
        """
        Retrieves the shapes registered for a given input name.

        Returns:
            ShapeTuple:
                    A named tuple include ``min``, ``opt``, and ``max`` members for the shapes
                    corresponding to the input.
        """
        if key not in self:
            G_LOGGER.critical("Binding: {:} does not have shapes set in this profile".format(key))
        return super().__getitem__(key)


class CreateConfig(BaseLoadModel):
    def __init__(self, max_workspace_size=None, tf32=None, fp16=None, int8=None, profiles=None, calibrator=None, strict_types=None):
        """
        Functor that creates a TensorRT IBuilderConfig.

        Args:
            max_workspace_size (int): The maximum workspace size, in bytes, when building the engine.
            tf32 (bool): Whether to build the engine with TF32 precision enabled. Defaults to False.
            fp16 (bool): Whether to build the engine with FP16 precision enabled. Defaults to False.
            int8 (bool): Whether to build the engine with INT8 precision enabled. Defaults to False.
            profiles (List[Profile]):
                    A list of optimization profiles to add to the configuration. Only needed for
                    networks with dynamic input shapes. If this is omitted for a network with
                    dynamic shapes, a default profile is created, where dynamic dimensions are
                    replaced with Polygraphy's DEFAULT_SHAPE_VALUE  (defined in util/constants.py).
                    See `Profile` for details.
            calibrator (trt.IInt8Calibrator):
                    An int8 calibrator. Only required in int8 mode when
                    the network does not have explicit precision. For networks with
                    dynamic shapes, the last profile provided (or default profile if
                    no profiles are provided) is used during calibration.
        """
        self.max_workspace_size = misc.default_value(max_workspace_size, 1 << 24)
        self.tf32 = misc.default_value(tf32, False)
        self.fp16 = misc.default_value(fp16, False)
        self.int8 = misc.default_value(int8, False)
        self.profiles = misc.default_value(profiles, [])
        self.calibrator = calibrator
        self.strict_types = misc.default_value(strict_types, False)

        if self.calibrator is not None and not self.int8:
            G_LOGGER.warning("A calibrator was provided to `CreateConfig`, but int8 mode was not enabled. "
                             "Did you mean to set `int8=True` to enable building with int8 precision?")


    def __call__(self, builder, network):
        """
        Creates a TensorRT IBuilderConfig that can be used by the EngineFromNetwork.

        Args:
            builder (trt.Builder):
                    The TensorRT builder to use to create the configuration.
            network (trt.INetworkDefinition):
                    The TensorRT network for which to create the config. The network is used to
                    automatically create a default optimization profile if none are provided.

        Returns:
            trt.IBuilderConfig: The TensorRT builder configuration.
        """
        config = builder.create_builder_config()

        calibration_profile = None
        for profile in self.profiles:
            calibration_profile = trt_util.build_profile(builder, network, profile)
            config.add_optimization_profile(calibration_profile)
        if not self.profiles:
            calibration_profile = trt_util.build_default_profile(builder, network)
            config.add_optimization_profile(calibration_profile)

        if self.profiles:
            G_LOGGER.info("Configuring with profiles: {:}".format(self.profiles))

        config.max_workspace_size = int(self.max_workspace_size)

        if self.strict_types:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if not self.tf32:
            with contextlib.suppress(AttributeError):
                config.clear_flag(trt.BuilderFlag.TF32)
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if not network.has_explicit_precision:
                if self.calibrator is not None:
                    input_metadata = trt_util.get_input_metadata_from_profile(calibration_profile, network)
                    with contextlib.suppress(AttributeError):
                        self.calibrator.reset(input_metadata)
                    config.int8_calibrator = self.calibrator
                else:
                    G_LOGGER.warning("Network does not have explicit precision and no calibrator was provided. Please ensure "
                                     "that tensors in the network have dynamic ranges set, or provide a calibrator in order to use int8 mode.")
        return config


class EngineFromNetwork(BaseLoadModel):
    def __init__(self, network, config=None):
        """
        Functor that uses a TensorRT ``INetworkDefinition`` to build an engine.

        Args:
            network (Callable() -> trt.Builder, trt.INetworkDefinition):
                    A callable capable of returning a TensorRT Builder and INetworkDefinition. The returned builder
                    and network are owned by EngineFromNetwork and should not be freed manually. The callable may
                    have at most 3 return values if another object needs to be kept alive for the duration of the network,
                    e.g., in the case of a parser. EngineFromNetwork will take ownership of the third return value, and,
                    like the network, it should not be freed by the callable. The first and second return values must
                    always be the builder and network respectively.
                    If instead of a loader, the network, builder, and optional parser arguments are provided directly,
                    then EngineFromNetwork will *not* deallocate them.


            config (Callable(trt.Builder, trt.INetworkDefinition) -> trt.IBuilderConfig):
                    A callable that returns a TensorRT builder configuration. If not supplied,
                    a `CreateConfig` instance with default parameters is used.
        """
        self._network = network
        self._config = misc.default_value(config, CreateConfig())


    def __call__(self):
        """
        Builds a TensorRT engine.

        Returns:
            trt.ICudaEngine: The engine that was created.
        """
        # If network is a callable, then we own its return value
        ret, owning = misc.try_call(self._network)
        builder, network, parser = misc.unpack_args(ret, num=3)

        with contextlib.ExitStack() as stack:
            provided = "Builder and Network" if parser is None else "Builder, Network, and Parser"
            if owning:
                stack.enter_context(builder)
                stack.enter_context(network)
                if parser is not None:
                    stack.enter_context(parser)
            else:
                G_LOGGER.verbose("{:} were provided directly instead of via a Callable. This loader will not assume ownership. "
                               "Please ensure that they are freed.".format(provided))

            network_log_mode = "full" if G_LOGGER.severity <= G_LOGGER.ULTRA_VERBOSE else "attrs"
            G_LOGGER.super_verbose(lambda: ("Displaying TensorRT Network:\n" + trt_util.str_from_network(network, mode=network_log_mode)))

            config, _ = misc.try_call(self._config, builder, network)
            G_LOGGER.info("Building engine with configuration: {:}".format(trt_util.str_from_config(config)))
            engine = builder.build_engine(network, config)
            if not engine:
                G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")

            if hasattr(config.int8_calibrator, "free"):
                config.int8_calibrator.free()

            return engine


class EngineFromBytes(BaseLoadModel):
    def __init__(self, serialized_engine):
        """
        Functor that deserializes an engine from a buffer.

        Args:
            serialized_engine (Callable() -> Union[str, bytes]):
                    Either a loader that can supply a memory buffer, or a memory buffer itself.
                    For example, use a lambda: `lambda: open("/path/to/trt.engine", "rb").read()` for lazy evaluation,
                    or the contents of the file directly: `open("/path/to/trt.engine", "rb").read()` for immediate evaluation.
        """
        self._serialized_engine = serialized_engine


    def __call__(self):
        """
        Deserializes an engine from a buffer.

        Returns:
            trt.ICudaEngine: The deserialized engine.
        """
        buffer, _ = misc.try_call(self._serialized_engine)

        trt.init_libnvinfer_plugins(trt_util.TRT_LOGGER, "")
        with trt.Runtime(trt_util.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(buffer)
            if not engine:
                G_LOGGER.critical("Could not load engine")
                G_LOGGER.ultra_verbose(lambda: "Note: serialized_engine was: {:}".format(buffer))
        return engine


class SaveEngine(BaseLoadModel):
    def __init__(self, engine, path=None):
        """
        Functor that saves an engine to the provided path.

        Args:
            engine (Callable() -> trt.ICudaEngine):
                    A callable that can supply a TensorRT engine.


            path (str): The path at which to save the engine.
        """
        self._engine = engine
        self.path = path


    def __call__(self):
        """
        Saves an engine to the provided path.

        Returns:
            trt.ICudaEngine: The engine that was saved.
        """
        engine, _ = misc.try_call(self._engine)
        misc.lazy_write(contents=lambda: engine.serialize(), path=self.path)
        return engine
