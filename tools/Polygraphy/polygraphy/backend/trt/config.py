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
import copy
import re

from polygraphy import config as polygraphy_config, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.profile import Profile
from polygraphy.backend.trt.util import inherit_and_extend_docstring
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER

trt = lazy_import_trt()


class _CreateConfigCommon(BaseLoader):
    """
    Generic TensorRT IBuilderConfig.
    """

    def __init__(
        self,
        profiles=None,
        precision_constraints=None,
        load_timing_cache=None,
        algorithm_selector=None,
        sparse_weights=None,
        tactic_sources=None,
        restricted=None,
        profiling_verbosity=None,
        memory_pool_limits=None,
        refittable=None,
        strip_plan=None,
        preview_features=None,
        engine_capability=None,
        direct_io=None,
        builder_optimization_level=None,
        hardware_compatibility_level=None,
        max_aux_streams=None,
        version_compatible=None,
        exclude_lean_runtime=None,
        quantization_flags=None,
        error_on_timing_cache_miss=None,
        disable_compilation_cache=None,
        progress_monitor=None,
        weight_streaming=None,
        runtime_platform=None,
        tiling_optimization_level=None,
    ):
        """
        Creates an IBuilderConfig that can be used by EngineFromNetwork.

        Args:
            profiles (List[Profile]):
                    A list of optimization profiles to add to the configuration. Only needed for
                    networks with dynamic input shapes. If this is omitted for a network with
                    dynamic shapes, a default profile is created, where dynamic dimensions are
                    replaced with Polygraphy's DEFAULT_SHAPE_VALUE (defined in constants.py).
                    A partially populated profile will be automatically filled using values from ``Profile.fill_defaults()``
                    See ``Profile`` for details.
            precision_constraints (Optional[str]):
                    If set to "obey", require that layers execute in specified precisions.
                    If set to "prefer", prefer that layers execute in specified precisions but allow TRT to fall back to
                    other precisions if no implementation exists for the requested precision.
                    Otherwise, precision constraints are ignored.
                    Defaults to None.
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
            strip_plan (bool):
                    Strips the refittable weights from the engine plan file.
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
            hardware_compatibility_level (trt.HardwareCompatibilityLevel):
                    The hardware compatibility level. This allows engines built on one GPU architecture to work on GPUs
                    of other architectures.
                    Defaults to TensorRT's default hardware compatibility level.
            max_aux_streams (int):
                    The maximum number of auxiliary streams that TensorRT is allowed to use. If the network contains
                    operators that can run in parallel, TRT can execute them using auxiliary streams in addition to the
                    one provided to the IExecutionContext::enqueueV3() call.
                    The default maximum number of auxiliary streams is determined by the heuristics in TensorRT on
                    whether enabling multi-stream would improve the performance.
            version_compatible (bool):
                    Whether to build an engine that is version compatible.
            exclude_lean_runtime (bool):
                    Whether to exclude the lean runtime in version compatible engines.
                    Requires that version compatibility is enabled.
            quantization_flags (List[trt.QuantizationFlag]):
                    The quantization flags to enable.
                    Use an empty list to disable all quantization flags.
                    Defaults to TensorRT's default quantization flags.
            error_on_timing_cache_miss (bool):
                    Emit error when a tactic being timed is not present in the timing cache.
                    This flag has an effect only when IBuilderConfig has an associated ITimingCache.
                    Defaults to False.
            disable_compilation_cache (bool):
                    Whether to disable caching JIT-compiled code.
                    Defaults to False.
            progress_monitor (trt.IProgressMonitor):
                    A progress monitor. Allow users to view engine building progress through CLI.
            weight_streaming (bool):
                    TWhether to enable weight streaming for the TensorRT Engine.
            runtime_platform (trt.RuntimePlatform):
                    Describes the intended runtime platform (operating system and CPU architecture) for the execution of the TensorRT engine. 
                    TensorRT provides support for cross-platform engine compatibility when the target runtime platform is different from the build platform.
                    Defaults to TensorRT's default runtime platform.
            tiling_optimization_level (trt.TilingOptimizationLevel):
                    The tiling optimization level. Setting a higher optimization level allows TensorRT to spend more building time for more tiling strategies.
                    Defaults to TensorRT's default tiling optimization level. Refer to the TensorRT API documentation for details.
        """
        self.profiles = util.default(profiles, [Profile()])
        self.precision_constraints = precision_constraints
        self.restricted = util.default(restricted, False)
        self.refittable = util.default(refittable, False)
        self.strip_plan = util.default(strip_plan, False)
        self.timing_cache_path = load_timing_cache
        self.algorithm_selector = algorithm_selector
        self.sparse_weights = util.default(sparse_weights, False)
        self.tactic_sources = tactic_sources
        self.profiling_verbosity = profiling_verbosity
        self.memory_pool_limits = memory_pool_limits
        self.preview_features = preview_features
        self.engine_capability = engine_capability
        self.direct_io = util.default(direct_io, False)
        self.builder_optimization_level = builder_optimization_level
        self.hardware_compatibility_level = hardware_compatibility_level
        self.max_aux_streams = max_aux_streams
        self.version_compatible = version_compatible
        self.exclude_lean_runtime = exclude_lean_runtime
        self.quantization_flags = quantization_flags
        self.error_on_timing_cache_miss = util.default(
            error_on_timing_cache_miss, False
        )
        self.disable_compilation_cache = util.default(disable_compilation_cache, False)
        self.progress_monitor = progress_monitor
        self.weight_streaming = weight_streaming
        self.runtime_platform = runtime_platform
        self.tiling_optimization_level = tiling_optimization_level

    @util.check_called_by("__call__")
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
        config = builder.create_builder_config()

        def try_run(func, name):
            try:
                return func()
            except AttributeError:
                trt_util.fail_unavailable(f"{name} in {self.__class__.__name__}")

        def try_set_flag(flag_name):
            return try_run(
                lambda: config.set_flag(getattr(trt.BuilderFlag, flag_name)),
                flag_name.lower(),
            )

        if self.preview_features is not None:
            for preview_feature in trt.PreviewFeature.__members__.values():
                try_run(
                    lambda: config.set_preview_feature(
                        preview_feature, preview_feature in self.preview_features
                    ),
                    "preview_features",
                )

        G_LOGGER.verbose("Setting TensorRT Optimization Profiles")
        profiles = copy.deepcopy(self.profiles)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network)
            config.add_optimization_profile(calib_profile.to_trt(builder, network))
        newline = "\n"
        sep = ",\n"
        G_LOGGER.info(
            f"Configuring with profiles:[\n"
            f"{util.indent_block(sep.join([f'Profile {index}:{newline}{util.indent_block(profile)}' for index, profile in enumerate(profiles)]))}\n]"
        )

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
        if self.restricted:
            try_set_flag("SAFETY_SCOPE")

        if self.refittable:
            try_set_flag("REFIT")

        if self.strip_plan:
            try_set_flag("STRIP_PLAN")

        if self.direct_io:
            try_set_flag("DIRECT_IO")

        if self.sparse_weights:
            try_set_flag("SPARSE_WEIGHTS")

        if self.profiling_verbosity is not None:

            def set_profiling_verbosity():
                config.profiling_verbosity = self.profiling_verbosity

            try_run(set_profiling_verbosity, name="profiling_verbosity")
        else:
            try:
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            except AttributeError:
                pass

        if self.memory_pool_limits is not None:
            for pool_type, pool_size in self.memory_pool_limits.items():
                try_run(
                    lambda: config.set_memory_pool_limit(pool_type, pool_size),
                    name="memory_pool_limits",
                )

        if self.tactic_sources is not None:
            tactic_sources_flag = 0
            for source in self.tactic_sources:
                tactic_sources_flag |= 1 << int(source)
            try_run(
                lambda: config.set_tactic_sources(tactic_sources_flag),
                name="tactic_sources",
            )

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
                trt_util.fail_unavailable(f"load_timing_cache in {self.__class__.__name__}")
        else:
            config.set_timing_cache(cache, ignore_mismatch=False)

        if self.algorithm_selector is not None:

            def set_algo_selector():
                config.algorithm_selector = self.algorithm_selector

            try_run(set_algo_selector, name="algorithm_selector")

            if not self.timing_cache_path:
                G_LOGGER.warning(
                    "Disabling tactic timing cache because algorithm selector is enabled."
                )
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

        if self.version_compatible:
            try_set_flag("VERSION_COMPATIBLE")

        if self.exclude_lean_runtime:
            if not self.version_compatible:
                G_LOGGER.critical(
                    f"Cannot set EXCLUDE_LEAN_RUNTIME if version compatibility is not enabled. "
                )
            try_set_flag("EXCLUDE_LEAN_RUNTIME")

        if self.hardware_compatibility_level is not None or self.version_compatible:
            G_LOGGER.info(
                "Version or hardware compatibility was enabled. "
                "If you are using an ONNX model, please set the NATIVE_INSTANCENORM ONNX parser flag, e.g. `--onnx-flags NATIVE_INSTANCENORM`"
            )

        if self.max_aux_streams is not None:

            def set_max_aux_streams():
                config.max_aux_streams = self.max_aux_streams

            try_run(set_max_aux_streams, "max_aux_streams")

        if self.quantization_flags is not None:
            for quantization_flag in trt.QuantizationFlag.__members__.values():
                if quantization_flag in self.quantization_flags:
                    try_run(
                        lambda: config.set_quantization_flag(quantization_flag),
                        "quantization_flag",
                    )
                else:
                    try_run(
                        lambda: config.clear_quantization_flag(quantization_flag),
                        "quantization_flag",
                    )

        if self.error_on_timing_cache_miss:
            try_set_flag("ERROR_ON_TIMING_CACHE_MISS")

        if self.disable_compilation_cache:
            try_set_flag("DISABLE_COMPILATION_CACHE")

        if self.progress_monitor is not None:

            def set_progress_monitor():
                config.progress_monitor = self.progress_monitor

            try_run(set_progress_monitor, name="progress_monitor")

        if self.weight_streaming:
            try_set_flag("WEIGHT_STREAMING")
        
        if self.runtime_platform is not None:

            def set_runtime_platform():
                config.runtime_platform = self.runtime_platform

            try_run(set_runtime_platform, "runtime_platform")

        if self.tiling_optimization_level is not None:

            def set_tiling_optimization_level():
                config.tiling_optimization_level = self.tiling_optimization_level

            try_run(set_tiling_optimization_level, "tiling_optimization_level")

        return config


@mod.export(funcify=True)
class CreateConfig(_CreateConfigCommon):
    """
    Functor that creates an IBuilderConfig with TensorRT features.
    """

    @inherit_and_extend_docstring(_CreateConfigCommon.__init__)
    def __init__(
        self,
        tf32=None,
        fp16=None,
        int8=None,
        fp8=None,
        bf16=None,
        calibrator=None,
        use_dla=None,
        allow_gpu_fallback=None,
        **kwargs
    ):
        """
        Creates an IBuilderConfig with TensorRT-specific features.

        Args:
            tf32 (bool):
                    Whether to enable TF32 precision. Defaults to False.
            fp16 (bool):
                    Whether to enable FP16 precision. Defaults to False.
            int8 (bool):
                    Whether to enable INT8 precision. Defaults to False.
            fp8 (bool):
                    Whether to enable FP8 precision. Defaults to False.
            bf16 (bool):
                    Whether to enable BF16 precision. Defaults to False.
            calibrator (trt.IInt8Calibrator):
                    An int8 calibrator. Only required in int8 mode when
                    the network does not have explicit precision. For networks with
                    dynamic shapes, the last profile provided (or default profile if
                    no profiles are provided) is used during calibration.
            use_dla (bool):
                    [EXPERIMENTAL] Whether to enable DLA as the default device type.
                    Defaults to False.
            allow_gpu_fallback (bool):
                    [EXPERIMENTAL] When DLA is enabled, whether to allow layers to fall back to GPU if they cannot be run on DLA.
                    Has no effect if DLA is not enabled.
                    Defaults to False.
            **kwargs: All other arguments from _CreateConfigCommon.
        """
        super().__init__(**kwargs)
        self.tf32 = util.default(tf32, False)
        self.fp16 = util.default(fp16, False)
        self.bf16 = util.default(bf16, False)
        self.int8 = util.default(int8, False)
        self.fp8 = util.default(fp8, False)
        self.calibrator = calibrator
        self.use_dla = util.default(use_dla, False)
        self.allow_gpu_fallback = util.default(allow_gpu_fallback, False)

        if self.calibrator is not None and not self.int8:
            G_LOGGER.warning(
                "A calibrator was provided to `CreateConfig`, but int8 mode was not enabled. "
                "Did you mean to set `int8=True` to enable building with int8 precision?"
            )

        # Print a message to tell users that TF32 can be enabled to improve perf with minor accuracy differences.
        if not self.tf32:
            G_LOGGER.info(
                "TF32 is disabled by default. Turn on TF32 for better performance with minor accuracy differences."
            )

        self._validator()

    def _validator(self):
        """
        Validates initialization parameters for TensorRT-specific features.
        """
        # Validate that TensorRT-RTX specific flags are not used in regular TensorRT mode
        if polygraphy_config.USE_TENSORRT_RTX:
            if self.fp16 or self.int8 or self.bf16 or self.fp8:
                G_LOGGER.critical("Precision flags (fp16, int8, bf16, fp8) are not supported with USE_TENSORRT_RTX=1.")
            if self.use_dla:
                G_LOGGER.critical("DLA is not supported with USE_TENSORRT_RTX=1.")
            if self.calibrator is not None:
                G_LOGGER.critical("Custom calibrator is not supported with USE_TENSORRT_RTX=1.")

    def _configure_flags(self, builder, network, config):
        """
        Validates and configures TensorRT-specific features.

        Args:
            builder (trt.Builder): The TensorRT builder
            network (trt.INetworkDefinition): The TensorRT network
            config (trt.IBuilderConfig): The TensorRT builder config to modify
        """
        def try_run(func, name):
            try:
                return func()
            except AttributeError:
                trt_util.fail_unavailable(f"{name} in CreateConfig")

        def try_set_flag(flag_name):
            return try_run(
                lambda: config.set_flag(getattr(trt.BuilderFlag, flag_name)),
                flag_name.lower(),
            )

        # Add precision-related logic
        if self.tf32:
            try_set_flag("TF32")
        else:  # TF32 is on by default
            with contextlib.suppress(AttributeError):
                config.clear_flag(trt.BuilderFlag.TF32)

        if self.fp16:
            try_set_flag("FP16")

        if self.bf16:
            try_set_flag("BF16")

        if self.fp8:
            try_set_flag("FP8")

        if self.int8:
            try_set_flag("INT8")

        if self.int8:
            # No Q/DQ layers means that we will need to calibrate.
            if not any(
                layer.type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]
                for layer in network
            ):
                if self.calibrator is not None:
                    config.int8_calibrator = self.calibrator
                    try:
                        profiles = copy.deepcopy(self.profiles)
                        calib_profile = profiles[-1].fill_defaults(network)
                        config.set_calibration_profile(
                            calib_profile.to_trt(builder, network)
                        )
                        G_LOGGER.info(f"Using calibration profile: {calib_profile}")
                    except AttributeError:
                        G_LOGGER.extra_verbose(
                            "Cannot set calibration profile on TensorRT 7.0 and older."
                        )

                    trt_util.try_setup_polygraphy_calibrator(
                        config,
                        network,
                        calib_profile=calib_profile.to_trt(builder, network),
                    )
                else:
                    G_LOGGER.warning(
                        "Network does not have explicit precision and no calibrator was provided. Please ensure "
                        "that tensors in the network have dynamic ranges set, or provide a calibrator in order to use int8 mode."
                    )

        if self.use_dla:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0

        if self.allow_gpu_fallback:
            try_set_flag("GPU_FALLBACK")

    @util.check_called_by("__call__")
    def call_impl(self, builder, network):
        """
        Callable implementation that creates and configures the IBuilderConfig with TensorRT features.
        """
        config = super().call_impl(builder, network)

        self._configure_flags(builder, network, config)

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
            G_LOGGER.critical(
                f"Object {func} (of type {type(func)}) is not a callable."
            )

        self._func = func

    @util.check_called_by("__call__")
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
        config, _ = util.invoke_if_callable(self._config, builder, network)

        self._func(builder, network, config)

        return config
