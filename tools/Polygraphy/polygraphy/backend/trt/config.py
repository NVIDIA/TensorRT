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
import copy

from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.profile import Profile
from polygraphy.logger import G_LOGGER

trt = mod.lazy_import("tensorrt")


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
        precision_constraints=None,
        strict_types=None,
        load_timing_cache=None,
        algorithm_selector=None,
        sparse_weights=None,
        tactic_sources=None,
        restricted=None,
        use_dla=None,
        allow_gpu_fallback=None,
        profiling_verbosity=None,
        memory_pool_limits=None,
        refittable=None,
        preview_features=None,
        engine_capability=None,
        direct_io=None,
        builder_optimization_level=None,
        fp8=None,
        hardware_compatibility_level=None,
    ):
        """
        Creates a TensorRT IBuilderConfig that can be used by EngineFromNetwork.

        Args:
            max_workspace_size (int):
                    [DEPRECATED - use memory_pool_limits]
                    The maximum workspace size, in bytes, when building the engine.
                    Defaults to None.
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
            precision_constraints (Optional[str]):
                    If set to "obey", require that layers execute in specified precisions.
                    If set to "prefer", prefer that layers execute in specified precisions but allow TRT to fall back to
                    other precisions if no implementation exists for the requested precision.
                    Otherwise, precision constraints are ignored.
                    Defaults to None.
            strict_types (bool):
                    [DEPRECATED] If True, prefer that layers execute in specified precisions and avoid I/O reformatting.
                    Fall back to ignoring the preferences if such an engine cannot be built.
                    precision_constraints is recommended instead.
                    Defaults to False.
            load_timing_cache (Union[str, file-like]):
                    A path or file-like object from which to load a tactic timing cache.
                    Providing a tactic timing cache can speed up the engine building process.
                    Caches can be generated while building an engine with, for example, EngineFromNetwork.
                    If a path is provided, the file will be locked for exclusive access so that other processes
                    cannot update the cache while it is being read.
                    If the file specified by the path does not exist, CreateConfig will emit a warning and fall back
                    to using an empty timing cache.
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
            profiling_verbosity (trt.ProfilingVerbosity):
                    The verbosity of NVTX annotations in the generated engine.
                    Higher verbosity allows you to determine more information about the engine.
                    Defaults to ``trt.ProfilingVerbosity.VERBOSE``.
            memory_pool_limits (Dict[trt.MemoryPoolType, int]):
                    Limits for different memory pools.
                    This should be a mapping of pool types to their respective limits in bytes.
            refittable (bool):
                    Enables the engine to be refitted with new weights after it is built.
                    Defaults to False.
            preview_features (List[trt.PreviewFeature]):
                    The preview features to enable.
                    Use an empty list to disable all preview features.
                    Defaults to TensorRT's default preview features.
            engine_capability (trt.EngineCapability):
                    The engine capability to build for.
                    Defaults to the default TensorRT engine capability.
            direct_io (bool):
                    Whether to disallow reformatting layers at network input/output tensors with
                    user-specified formats.
                    Defaults to False.
            builder_optimization_level (int):
                    The builder optimization level. A higher optimization level allows the optimizer to spend more time
                    searching for optimization opportunities. The resulting engine may have better performance compared
                    to an engine built with a lower optimization level.
                    Refer to the TensorRT API documentation for details.
                    Defaults to TensorRT's default optimization level.
            fp8  (bool):
                    Whether to build the engine with FP8 precision enabled.
                    Defaults to False.
            hardware_compatibility_level (trt.HardwareCompatibilityLevel):
                    The hardware compatibiliity level. This allows engines built on one GPU architecture to work on GPUs
                    of other architectures.
                    Defaults to TensorRT's default hardware compatibility level.
        """
        self.max_workspace_size = max_workspace_size
        if max_workspace_size is not None:
            mod.warn_deprecated("max_workspace_size", use_instead="memory_pool_limits", remove_in="0.45.0")

        self.tf32 = util.default(tf32, False)
        self.fp16 = util.default(fp16, False)
        self.int8 = util.default(int8, False)
        self.fp8 = util.default(fp8, False)
        self.profiles = util.default(profiles, [Profile()])
        self.calibrator = calibrator
        self.precision_constraints = precision_constraints
        self.strict_types = util.default(strict_types, False)
        self.restricted = util.default(restricted, False)
        self.refittable = util.default(refittable, False)
        self.timing_cache_path = load_timing_cache
        self.algorithm_selector = algorithm_selector
        self.sparse_weights = util.default(sparse_weights, False)
        self.tactic_sources = tactic_sources
        self.use_dla = util.default(use_dla, False)
        self.allow_gpu_fallback = util.default(allow_gpu_fallback, False)
        self.profiling_verbosity = profiling_verbosity
        self.memory_pool_limits = memory_pool_limits
        self.preview_features = preview_features
        self.engine_capability = engine_capability
        self.direct_io = util.default(direct_io, False)
        self.builder_optimization_level = builder_optimization_level
        self.hardware_compatibility_level = hardware_compatibility_level

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
                    trt_util.fail_unavailable(f"{name} in CreateConfig")

            def try_set_flag(flag_name):
                return try_run(lambda: config.set_flag(getattr(trt.BuilderFlag, flag_name)), flag_name.lower())

            if self.preview_features is not None:
                for preview_feature in trt.PreviewFeature.__members__.values():
                    try_run(
                        lambda: config.set_preview_feature(preview_feature, preview_feature in self.preview_features),
                        "preview_features",
                    )

            with G_LOGGER.indent():
                G_LOGGER.verbose("Setting TensorRT Optimization Profiles")
                profiles = copy.deepcopy(self.profiles)
                for profile in profiles:
                    # Last profile is used for set_calibration_profile.
                    calib_profile = profile.fill_defaults(network).to_trt(builder, network)
                    config.add_optimization_profile(calib_profile)
                G_LOGGER.info(f"Configuring with profiles: {profiles}")

            if self.max_workspace_size is not None:
                config.max_workspace_size = int(self.max_workspace_size)

            layer_with_precisions = {
                layer.name: layer.precision.name
                for layer in network
                if layer.precision_is_set and not layer.type == trt.LayerType.SHAPE
            }
            if self.precision_constraints == "obey":
                try_set_flag("OBEY_PRECISION_CONSTRAINTS")
            elif self.precision_constraints == "prefer":
                try_set_flag("PREFER_PRECISION_CONSTRAINTS")
            elif layer_with_precisions:
                G_LOGGER.warning(
                    "It looks like some layers in the network have compute precision set, but precision constraints were not enabled. "
                    "\nPrecision constraints must be set to 'prefer' or 'obey' for layer compute precision to take effect. "
                    f"\nNote: Layers and their requested precisions were: {layer_with_precisions}"
                )

            if self.strict_types:
                mod.warn_deprecated("strict_types", use_instead="precision_constraints", remove_in="0.45.0")
                try_set_flag("STRICT_TYPES")

            if self.restricted:
                try_set_flag("SAFETY_SCOPE")

            if self.refittable:
                try_set_flag("REFIT")

            if self.direct_io:
                try_set_flag("DIRECT_IO")

            if self.tf32:
                try_set_flag("TF32")
            else:  # TF32 is on by default
                with contextlib.suppress(AttributeError):
                    config.clear_flag(trt.BuilderFlag.TF32)

            if self.fp16:
                try_set_flag("FP16")


            if self.fp8:
                try_set_flag("FP8")

            if self.int8:
                try_set_flag("INT8")
                if not network.has_explicit_precision:
                    if self.calibrator is not None:
                        config.int8_calibrator = self.calibrator
                        try:
                            config.set_calibration_profile(calib_profile)
                        except AttributeError:
                            G_LOGGER.extra_verbose("Cannot set calibration profile on TensorRT 7.0 and older.")

                        trt_util.try_setup_polygraphy_calibrator(config, network, calib_profile=calib_profile)
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

            if self.profiling_verbosity is not None:

                def set_profiling_verbosity():
                    config.profiling_verbosity = self.profiling_verbosity

                try_run(set_profiling_verbosity, name="profiling_verbosity")
            else:
                try:
                    config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
                except AttributeError:
                    pass

            if self.memory_pool_limits is not None:
                for pool_type, pool_size in self.memory_pool_limits.items():
                    try_run(lambda: config.set_memory_pool_limit(pool_type, pool_size), name="memory_pool_limits")

            if self.tactic_sources is not None:
                tactic_sources_flag = 0
                for source in self.tactic_sources:
                    tactic_sources_flag |= 1 << int(source)
                try_run(lambda: config.set_tactic_sources(tactic_sources_flag), name="tactic_sources")

            try:
                cache = None
                if self.timing_cache_path:
                    try:
                        with util.LockFile(self.timing_cache_path):
                            timing_cache_data = util.load_file(
                                self.timing_cache_path, description="tactic timing cache"
                            )
                            cache = config.create_timing_cache(timing_cache_data)
                    except FileNotFoundError:
                        G_LOGGER.warning(
                            "Timing cache file {} not found, falling back to empty timing cache.".format(
                                self.timing_cache_path
                            )
                        )
                if cache is None:
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

                try_run(set_algo_selector, name="algorithm_selector")

                if not self.timing_cache_path:
                    G_LOGGER.warning("Disabling tactic timing cache because algorithm selector is enabled.")
                    try_set_flag("DISABLE_TIMING_CACHE")

            if self.engine_capability is not None:

                def set_engine_cap():
                    config.engine_capability = self.engine_capability

                try_run(set_engine_cap, "engine_capability")

            if self.builder_optimization_level is not None:

                def set_builder_optimization_level():
                    config.builder_optimization_level = self.builder_optimization_level

                try_run(set_builder_optimization_level, "builder_optimization_level")

            if self.hardware_compatibility_level is not None:

                def set_hardware_compatibility_level():
                    config.hardware_compatibility_level = self.hardware_compatibility_level

                try_run(set_hardware_compatibility_level, "hardware_compatibility_level")

            return config


@mod.export(funcify=True)
class PostprocessConfig(BaseLoader):
    """
    [EXPERIMENTAL] Functor that applies a given post-processing function to a TensorRT ``IBuilderConfig``.
    """

    def __init__(self, config, func):
        """
        Applies a given post-processing function to a TensorRT ``IBuilderConfig``.

        Args:
            config (Union[trt.IBuilderConfig, Callable[[trt.Builder, trt.INetworkDefinition], trt.IBuilderConfig]):
                    A TensorRT IBuilderConfig or a callable that accepts a TensorRT builder and network and returns a config.
            func (Callable[[trt.Builder, trt.INetworkDefinition, trt.IBuilderConfig], None])
                    A callable which takes a builder, network, and config parameter and modifies the config in place.
        """

        self._config = config

        # Sanity-check that the function passed in is callable
        if not callable(func):
            G_LOGGER.critical(f"Object {func} (of type {type(func)}) is not a callable.")

        self._func = func

    def call_impl(self, builder, network):
        """
        Args:
            builder (trt.Builder):
                    The TensorRT builder to use to create the configuration.
            network (trt.INetworkDefinition):
                    The TensorRT network for which to create the config. The network is used to
                    automatically create a default optimization profile if none are provided.

        Returns:
            trt.IBuilderConfig:
                    The modified builder configuration.
        """
        config, owns_config = util.invoke_if_callable(self._config, builder, network)

        with contextlib.ExitStack() as stack:
            if owns_config:
                stack.enter_context(util.FreeOnException([config]))

            self._func(builder, network, config)


            return config
