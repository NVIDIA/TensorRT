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
import copy
import os

from polygraphy import constants, mod, util
from polygraphy.common import TensorMetadata
from polygraphy import config as polygraphy_config
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.mod.trt_importer import tensorrt_module_and_version_string
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.trt.helper import make_trt_enum_val
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.comparator.data_loader import DataLoaderArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.script import (
    inline,
    inline_identifier,
    make_invocable,
    make_invocable_if_nondefault,
    safe,
)


def parse_profile_shapes(default_shapes, min_args, opt_args, max_args):
    """
    Parses TensorRT profile options from command-line arguments.

    Args:
        default_shapes (TensorMetadata): The inference input shapes.

    Returns:
        List[OrderedDict[str, Tuple[Shape]]]:
            A list of profiles where each profile is a dictionary that maps
            input names to a tuple of (min, opt, max) shapes.
    """

    def get_shapes(lst, idx):
        # Overwrite a copy of default_shapes with the shapes for min, opt, or max (if applicable)
        nonlocal default_shapes
        default_shapes = copy.copy(default_shapes)
        if idx < len(lst):
            default_shapes.update(args_util.parse_meta(lst[idx], includes_dtype=False))

        # Don't care about dtype, and need to override dynamic dimensions
        shapes = {
            name: util.override_dynamic_shape(shape)
            for name, (_, shape) in default_shapes.items()
        }

        for name, shape in shapes.items():
            if tuple(default_shapes[name].shape) != tuple(shape):
                G_LOGGER.warning(
                    f"Input tensor: {name} | For TensorRT profile, overriding dynamic shape: {default_shapes[name].shape} to: {shape}",
                    mode=LogMode.ONCE,
                )

        return shapes

    num_profiles = max(len(min_args), len(opt_args), len(max_args))

    # For cases where input shapes are provided, we have to generate a profile
    if not num_profiles and default_shapes:
        num_profiles = 1

    profiles = []
    for idx in range(num_profiles):
        min_shapes = get_shapes(min_args, idx)
        opt_shapes = get_shapes(opt_args, idx)
        max_shapes = get_shapes(max_args, idx)
        if sorted(min_shapes.keys()) != sorted(opt_shapes.keys()):
            G_LOGGER.critical(
                f"Mismatch in input names between minimum shapes ({list(min_shapes.keys())}) and optimum shapes ({list(opt_shapes.keys())})"
            )
        elif sorted(opt_shapes.keys()) != sorted(max_shapes.keys()):
            G_LOGGER.critical(
                f"Mismatch in input names between optimum shapes ({list(opt_shapes.keys())}) and maximum shapes ({list(max_shapes.keys())})"
            )

        profile = {
            name: (min_shapes[name], opt_shapes[name], max_shapes[name])
            for name in min_shapes.keys()
        }
        profiles.append(profile)
    return profiles


@mod.export()
class TrtConfigArgs(BaseArgs):
    """
    TensorRT Builder Configuration: creating the TensorRT BuilderConfig.

    Depends on:

        - DataLoaderArgs
        - ModelArgs: if allow_custom_input_shapes == True
    """

    def __init__(
        self,
        precision_constraints_default: bool = None,
        allow_random_data_calib_warning: bool = None,
        allow_custom_input_shapes: bool = None,
        allow_engine_capability: bool = None,
        allow_tensor_formats: bool = None,
        allow_compute_capabilities: bool = None,
    ):
        """
        Args:
            precision_constraints_default (str):
                    The default value to use for the precision constraints option.
                    Defaults to "none".
            allow_random_data_calib_warning (bool):
                    Whether to issue a warning when randomly generated data is being used for calibration.
                    Defaults to True.
            allow_custom_input_shapes (bool):
                    Whether to allow custom input shapes when randomly generating data.
                    Defaults to True.
            allow_engine_capability (bool):
                    Whether to allow engine capability to be specified.
                    Defaults to False.
            allow_tensor_formats (bool):
                    Whether to allow tensor formats and related options to be set.
                    Defaults to False.
            allow_compute_capabilities (bool):
                    Whether to allow compute capabilities options to be set.
                    Defaults to False.
        """
        super().__init__()
        self._precision_constraints_default = util.default(
            precision_constraints_default, "none"
        )
        self._allow_random_data_calib_warning = util.default(
            allow_random_data_calib_warning, True
        )
        self._allow_custom_input_shapes = util.default(allow_custom_input_shapes, True)
        self._allow_engine_capability = util.default(allow_engine_capability, False)
        self._allow_tensor_formats = util.default(allow_tensor_formats, False)
        self._allow_compute_capabilities = util.default(allow_compute_capabilities, False)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--trt-min-shapes",
            action="append",
            help="The minimum shapes the optimization profile(s) will support. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-min-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )
        self.group.add_argument(
            "--trt-opt-shapes",
            action="append",
            help="The shapes for which the optimization profile(s) will be most performant. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-opt-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )
        self.group.add_argument(
            "--trt-max-shapes",
            action="append",
            help="The maximum shapes the optimization profile(s) will support. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-max-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )

        self.group.add_argument(
            "--tf32",
            help="Enable tf32 precision in TensorRT",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--fp16",
            help="Enable fp16 precision in TensorRT",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--bf16",
            help="Enable bf16 precision in TensorRT",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--fp8",
            help="Enable fp8 precision in TensorRT",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--int8",
            help="Enable int8 precision in TensorRT. "
            "If calibration is required but no calibration cache is provided, this option will cause TensorRT to run "
            "int8 calibration using the Polygraphy data loader to provide calibration data. "
            "If calibration is run and the model has dynamic shapes, the last optimization profile will be "
            "used as the calibration profile. ",
            action="store_true",
            default=None,
        )

        precision_constraints_group = self.group.add_mutually_exclusive_group()
        precision_constraints_group.add_argument(
            "--precision-constraints",
            help=f"If set to `prefer`, TensorRT will restrict available tactics to layer precisions specified in the network unless no implementation exists with the preferred layer constraints, in which case it will issue a warning and use the fastest available implementation. If set to `obey`, TensorRT will instead fail to build the network if no implementation exists with the preferred layer constraints. Defaults to `{self._precision_constraints_default}`",
            choices=("prefer", "obey", "none"),
            default=self._precision_constraints_default,
        )
        self.group.add_argument(
            "--sparse-weights",
            help="Enable optimizations for sparse weights in TensorRT",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--version-compatible",
            help="Builds an engine designed to be forward TensorRT version compatible.",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--exclude-lean-runtime",
            help="Exclude the lean runtime from the plan when version compatibility is enabled. ",
            action="store_true",
            default=None,
        )

        self.group.add_argument(
            "--calibration-cache",
            help="Path to load/save a calibration cache. "
            "Used to store calibration scales to speed up the process of int8 calibration. "
            "If the provided path does not yet exist, int8 calibration scales will be calculated and written to it during engine building. "
            "If the provided path does exist, it will be read and int8 calibration will be skipped during engine building. ",
            default=None,
        )
        self.group.add_argument(
            "--calib-base-cls",
            "--calibration-base-class",
            dest="calibration_base_class",
            help="The name of the calibration base class to use. For example, 'IInt8MinMaxCalibrator'. ",
            default=None,
        )
        self.group.add_argument(
            "--quantile",
            type=float,
            help="The quantile to use for IInt8LegacyCalibrator. Has no effect for other calibrator types.",
            default=None,
        )
        self.group.add_argument(
            "--regression-cutoff",
            type=float,
            help="The regression cutoff to use for IInt8LegacyCalibrator. Has no effect for other calibrator types.",
            default=None,
        )

        self.group.add_argument(
            "--load-timing-cache",
            help="Path to load tactic timing cache. "
            "Used to cache tactic timing information to speed up the engine building process. "
            "If the file specified by --load-timing-cache does not exist, Polygraphy will emit a warning and fall back to "
            "using an empty timing cache.",
            default=None,
        )

        self.group.add_argument(
            "--error-on-timing-cache-miss",
            help="Emit error when a tactic being timed is not present in the timing cache.",
            action="store_true",
            default=None,
        )

        self.group.add_argument(
            "--disable-compilation-cache",
            help="Disable caching JIT-compiled code",
            action="store_true",
            default=None,
        )

        replay_group = self.group.add_mutually_exclusive_group()
        replay_group.add_argument(
            "--save-tactics",
            "--save-tactic-replay",
            help="Path to save a Polygraphy tactic replay file. "
            "Details about tactics selected by TensorRT will be recorded and stored at this location as a JSON file. ",
            dest="save_tactics",
            default=None,
        )
        replay_group.add_argument(
            "--load-tactics",
            "--load-tactic-replay",
            help="Path to load a Polygraphy tactic replay file, such as one created by --save-tactics. "
            "The tactics specified in the file will be used to override TensorRT's default selections. ",
            dest="load_tactics",
            default=None,
        )

        self.group.add_argument(
            "--tactic-sources",
            help="Tactic sources to enable. This controls which libraries "
            "(e.g. cudnn, cublas, etc.) TensorRT is allowed to load tactics from. "
            "Values come from the names of the values in the trt.TacticSource enum and are case-insensitive. "
            "If no arguments are provided, e.g. '--tactic-sources', then all tactic sources are disabled."
            "Defaults to TensorRT's default tactic sources.",
            nargs="*",
            default=None,
        )

        self.group.add_argument(
            "--trt-config-script",
            help="Path to a Python script that defines a function that creates a "
            "TensorRT IBuilderConfig. The function should take a builder and network as parameters and return a "
            "TensorRT builder configuration. When this option is specified, all other config arguments are ignored. "
            "By default, Polygraphy looks for a function called `load_config`. You can specify a custom function name "
            "by separating it with a colon. For example: `my_custom_script.py:my_func`",
            default=None,
        )
        self.group.add_argument(
            "--trt-config-func-name",
            help="[DEPRECATED - function name can be specified with --trt-config-script like so: `my_custom_script.py:my_func`]"
            "When using a trt-config-script, this specifies the name of the function "
            "that creates the config. Defaults to `load_config`. ",
            default=None,
        )
        self.group.add_argument(
            "--trt-config-postprocess-script",
            "--trt-cpps",
            help="[EXPERIMENTAL] Path to a Python script that defines a function that modifies a TensorRT IBuilderConfig. "
            "This function will be called after Polygraphy has finished created the builder configuration and should take a builder, "
            "network, and config as parameters and modify the config in place. "
            "Unlike `--trt-config-script`, all other config arguments will be reflected in the config passed to the function."
            "By default, Polygraphy looks for a function called `postprocess_config`. You can specify a custom function name "
            "by separating it with a colon. For example: `my_custom_script.py:my_func`",
            default=None,
        )
        self.group.add_argument(
            "--trt-safety-restricted",
            help="Enable safety scope checking in TensorRT",
            action="store_true",
            default=None,
            dest="restricted",
        )
        self.group.add_argument(
            "--refittable",
            help="Enable the engine to be refitted with new weights after it is built.",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--strip-plan",
            help="Builds the engine with the refittable weights stripped.",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--use-dla",
            help="[EXPERIMENTAL] Use DLA as the default device type",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--allow-gpu-fallback",
            help="[EXPERIMENTAL] Allow layers unsupported on the DLA to fall back to GPU. Has no effect if --use-dla is not set.",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--pool-limit",
            "--memory-pool-limit",
            dest="memory_pool_limit",
            help="Memory pool limits. Memory pool names come from the names of values in the `trt.MemoryPoolType` enum and are case-insensitive"
            "Format: `--pool-limit <pool_name>:<pool_limit> ...`. For example, `--pool-limit dla_local_dram:1e9 workspace:16777216`. "
            "Optionally, use a `K`, `M`, or `G` suffix to indicate KiB, MiB, or GiB respectively. "
            "For example, `--pool-limit workspace:16M` is equivalent to `--pool-limit workspace:16777216`. ",
            nargs="+",
            default=None,
        )
        self.group.add_argument(
            "--preview-features",
            dest="preview_features",
            help="Preview features to enable. Values come from the names of the values "
            "in the trt.PreviewFeature enum, and are case-insensitive."
            "If no arguments are provided, e.g. '--preview-features', then all preview features are disabled. "
            "Defaults to TensorRT's default preview features.",
            nargs="*",
            default=None,
        )

        self.group.add_argument(
            "--builder-optimization-level",
            help="The builder optimization level. Setting a higher optimization "
            "level allows the optimizer to spend more time searching for optimization opportunities. "
            "The resulting engine may have better performance compared to an engine built with a lower optimization level. "
            "Refer to the TensorRT API documentation for details. ",
            type=int,
            default=None,
        )

        self.group.add_argument(
            "--hardware-compatibility-level",
            help="The hardware compatibility level to use for the engine. This allows engines built on one GPU architecture to work on GPUs "
            "of other architectures. Values come from the names of values in the `trt.HardwareCompatibilityLevel` enum and are case-insensitive. "
            "For example, `--hardware-compatibility-level ampere_plus` ",
            default=None,
        )

        self.group.add_argument(
            "--max-aux-streams",
            help="The maximum number of auxiliary streams that TensorRT is allowed to use. If the network contains "
            "operators that can run in parallel, TRT can execute them using auxiliary streams in addition to the one "
            "provided to the IExecutionContext.execute_async_v3() call. "
            "The default maximum number of auxiliary streams is determined by the heuristics in TensorRT on "
            "whether enabling multi-stream would improve the performance. "
            "Refer to the TensorRT API documentation for details.",
            type=int,
            default=None,
        )

        self.group.add_argument(
            "--quantization-flags",
            dest="quantization_flags",
            help="Int8 quantization flags to enable. Values come from the names of values "
            "in the trt.QuantizationFlag enum, and are case-insensitive. "
            "If no arguments are provided, e.g. '--quantization-flags', then all quantization flags are disabled. "
            "Defaults to TensorRT's default quantization flags.",
            nargs="*",
            default=None,
        )

        self.group.add_argument(
            "--profiling-verbosity",
            help="The verbosity of NVTX annotations in the generated engine."
            "Values come from the names of values in the `trt.ProfilingVerbosity` enum and are case-insensitive. "
            "For example, `--profiling-verbosity detailed`. "
            "Defaults to 'detailed'.",
            default=None,
        )

        self.group.add_argument(
            "--weight-streaming",
            help="Build a weight streamable engine. Must be set with --strongly-typed. The weight streaming amount can be set with --weight-streaming-budget.",
            action="store_true",
            default=None,
        )

        self.group.add_argument(
            "--runtime-platform",
            help="The target runtime platform (operating system and CPU architecture) for the execution of the TensorRT engine. "
            "TensorRT provides support for cross-platform engine compatibility when the target runtime platform is different from the build platform. "
            "Values come from the names of values in the `trt.RuntimePlatform` enum and are case-insensitive. "
            "For example, `--runtime-platform same_as_build`, `--runtime-platform windows_amd64` ",
            default=None,
        )

        if self._allow_engine_capability:
            self.group.add_argument(
                "--engine-capability",
                help="The desired engine capability. "
                "Possible values come from the names of the values in the trt.EngineCapability enum and are case-insensitive. ",
                default=None,
            )

        if self._allow_tensor_formats:
            self.group.add_argument(
                "--direct-io",
                help="Disallow reformatting layers at network input/output tensors which have user-specified formats. ",
                action="store_true",
                default=None,
            )

        self.group.add_argument(
            "--tiling-optimization-level",
            help="The tiling optimization level. Setting a higher optimization "
            "level allows TensorRT to spend more building time for more tiling strategies. "
            "Values come from the names of values in the `trt.TilingOptimizationLevel` enum and are case-insensitive.",
            default=None,
        )

        if polygraphy_config.USE_TENSORRT_RTX and self._allow_compute_capabilities:
            compute_capabilities_group = self.group.add_mutually_exclusive_group()

            compute_capabilities_group.add_argument(
                "--use-gpu",
                help="Use the current GPU device as target for engine compilation. "
                "Equivalent to setting ComputeCapability.CURRENT.",
                action="store_true",
                default=None,
            )

            compute_capabilities_group.add_argument(
                "--compute-capabilities",
                help="Specify target compute capabilities for engine compilation. "
                "Values should be major.minor versions (e.g., '7.5 8.0 8.6').",
                nargs="+",
                default=None,
            )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            profile_dicts (List[OrderedDict[str, Tuple[Shape]]]):
                A list of profiles where each profile is a dictionary that maps
                input names to a tuple of (min, opt, max) shapes.
            tf32 (bool): Whether to enable TF32.
            fp16 (bool): Whether to enable FP16.
            bf16 (bool): Whether to enable BF16.
            fp8  (bool): Whether to enable FP8.
            int8 (bool): Whether to enable INT8.
            precision_constraints (str): The precision constraints to apply.
            restricted (bool): Whether to enable safety scope checking in the builder.
            calibration_cache (str): Path to the calibration cache.
            calibration_base_class (str): The name of the base class to use for the calibrator.
            sparse_weights (bool): Whether to enable sparse weights.
            load_timing_cache (str): Path from which to load a timing cache.
            load_tactics (str): Path from which to load a tactic replay file.
            save_tactics (str): Path at which to save a tactic replay file.
            tactic_sources (List[str]): Strings representing enum values of the tactic sources to enable.
            trt_config_script (str): Path to a custom TensorRT config script.
            trt_config_func_name (str): Name of the function in the custom config script that creates the config.
            trt_config_postprocess_script (str): Path to a TensorRT config postprocessing script.
            trt_config_postprocess_func_name (str): Name of the function in the config postprocessing script that applies the post-processing.
            use_dla (bool): Whether to enable DLA.
            allow_gpu_fallback (bool): Whether to allow GPU fallback when DLA is enabled.
            memory_pool_limits (Dict[str, int]): Mapping of strings representing memory pool enum values to memory limits in bytes.
            engine_capability (str): The desired engine capability.
            direct_io (bool): Whether to disallow reformatting layers at network input/output tensors which have user-specified formats.
            preview_features (List[str]): Names of preview features to enable.
            refittable (bool): Whether the engine should be refittable.
            strip_plan (bool): Whether the engine should be built with the refittable weights stripped.
            builder_optimization_level (int): The builder optimization level.
            hardware_compatibility_level (str): A string representing a hardware compatibility level enum value.
            profiling_verbosity (str): A string representing a profiling verbosity level enum value.
            max_aux_streams (int): The maximum number of auxiliary streams that TensorRT is allowed to use.
            version_compatible (bool): Whether or not to build a TensorRT forward-compatible.
            exclude_lean_runtime (bool): Whether to exclude the lean runtime from a version compatible plan.
            quantization_flags (List[str]): Names of quantization flags to enable.
            error_on_timing_cache_miss (bool): Whether to emit error when a tactic being timed is not present in the timing cache.
            disable_compilation_cache (bool): Whether to disable caching JIT-compiled code.
            weight_streaming (bool): Whether to enable weight streaming for the TensorRT Engine.
            runtime_platform (str): A string representing the target runtime platform enum value.
            tiling_optimization_level (str): The tiling optimization level.
            use_gpu (bool): Whether to use the current GPU device as target for engine compilation.
            compute_capabilities (List[Tuple[int, int]]): List of (major, minor) compute capability tuples to target for engine compilation.
        """

        trt_min_shapes = args_util.get(args, "trt_min_shapes", default=[])
        trt_max_shapes = args_util.get(args, "trt_max_shapes", default=[])
        trt_opt_shapes = args_util.get(args, "trt_opt_shapes", default=[])

        default_shapes = TensorMetadata()
        if self._allow_custom_input_shapes:
            if not hasattr(self.arg_groups[ModelArgs], "input_shapes"):
                G_LOGGER.internal_error(
                    "ModelArgs must be parsed before TrtConfigArgs!"
                )
            default_shapes = self.arg_groups[ModelArgs].input_shapes

        self.profile_dicts = parse_profile_shapes(
            default_shapes, trt_min_shapes, trt_opt_shapes, trt_max_shapes
        )

        self.tf32 = args_util.get(args, "tf32")
        self.fp16 = args_util.get(args, "fp16")
        self.bf16 = args_util.get(args, "bf16")
        self.int8 = args_util.get(args, "int8")
        self.fp8 = args_util.get(args, "fp8")
        self.precision_constraints = args_util.get(args, "precision_constraints")

        if self.precision_constraints == "none":
            self.precision_constraints = None

        self.restricted = args_util.get(args, "restricted")
        self.refittable = args_util.get(args, "refittable")
        self.strip_plan = args_util.get(args, "strip_plan")

        self.calibration_cache = args_util.get(args, "calibration_cache")
        calib_base = args_util.get(args, "calibration_base_class")
        self.calibration_base_class = None
        if calib_base is not None:
            self.calibration_base_class = inline(
                safe("trt.{:}", inline_identifier(calib_base))
            )

        self._quantile = args_util.get(args, "quantile")
        self._regression_cutoff = args_util.get(args, "regression_cutoff")

        self.sparse_weights = args_util.get(args, "sparse_weights")

        self.load_timing_cache = args_util.get(args, "load_timing_cache")

        self.load_tactics = args_util.get(args, "load_tactics")
        self.save_tactics = args_util.get(args, "save_tactics")

        tactic_sources = args_util.get(args, "tactic_sources")
        self.tactic_sources = None
        if tactic_sources is not None:
            self.tactic_sources = [
                make_trt_enum_val("TacticSource", source) for source in tactic_sources
            ]

        self.trt_config_script, self.trt_config_func_name = (
            args_util.parse_script_and_func_name(
                args_util.get(args, "trt_config_script"),
                default_func_name="load_config",
            )
        )
        (
            self.trt_config_postprocess_script,
            self.trt_config_postprocess_func_name,
        ) = args_util.parse_script_and_func_name(
            args_util.get(args, "trt_config_postprocess_script"),
            default_func_name="postprocess_config",
        )

        func_name = args_util.get(args, "trt_config_func_name")
        if func_name is not None:
            mod.warn_deprecated(
                "--trt-config-func-name",
                "the config script argument",
                "0.50.0",
                always_show_warning=True,
            )
            self.trt_config_func_name = func_name

        self.use_dla = args_util.get(args, "use_dla")
        self.allow_gpu_fallback = args_util.get(args, "allow_gpu_fallback")

        memory_pool_limits = args_util.parse_arglist_to_dict(
            args_util.get(args, "memory_pool_limit"),
            cast_to=args_util.parse_num_bytes,
            allow_empty_key=False,
        )
        self.memory_pool_limits = None
        if memory_pool_limits is not None:
            self.memory_pool_limits = {
                make_trt_enum_val("MemoryPoolType", pool_type): pool_size
                for pool_type, pool_size in memory_pool_limits.items()
            }

        preview_features = args_util.get(args, "preview_features")
        self.preview_features = None
        if preview_features is not None:
            self.preview_features = [
                make_trt_enum_val("PreviewFeature", feature)
                for feature in preview_features
            ]

        engine_capability = args_util.get(args, "engine_capability")
        self.engine_capability = None
        if engine_capability is not None:
            self.engine_capability = make_trt_enum_val(
                "EngineCapability", engine_capability
            )

        self.direct_io = args_util.get(args, "direct_io")
        self.builder_optimization_level = args_util.get(
            args, "builder_optimization_level"
        )

        self.hardware_compatibility_level = None
        hardware_compatibility_level = args_util.get(
            args, "hardware_compatibility_level"
        )
        if hardware_compatibility_level is not None:
            self.hardware_compatibility_level = make_trt_enum_val(
                "HardwareCompatibilityLevel", hardware_compatibility_level
            )

        self.runtime_platform = None
        runtime_platform = args_util.get(
            args, "runtime_platform"
        )
        if runtime_platform is not None:
            self.runtime_platform = make_trt_enum_val(
                "RuntimePlatform", runtime_platform
            )

        self.profiling_verbosity = None
        profiling_verbosity = args_util.get(args, "profiling_verbosity")
        if profiling_verbosity is not None:
            self.profiling_verbosity = make_trt_enum_val(
                "ProfilingVerbosity", profiling_verbosity
            )

        self.max_aux_streams = args_util.get(args, "max_aux_streams")
        self.version_compatible = args_util.get(args, "version_compatible")
        self.exclude_lean_runtime = args_util.get(args, "exclude_lean_runtime")

        quantization_flags = args_util.get(args, "quantization_flags")
        self.quantization_flags = None
        if quantization_flags is not None:
            self.quantization_flags = [
                make_trt_enum_val("QuantizationFlag", flag)
                for flag in quantization_flags
            ]

        if self.exclude_lean_runtime and not self.version_compatible:
            G_LOGGER.critical(
                f"`--exclude-lean-runtime` requires `--version-compatible` to be enabled."
            )

        self.error_on_timing_cache_miss = args_util.get(
            args, "error_on_timing_cache_miss"
        )

        self.disable_compilation_cache = args_util.get(
            args, "disable_compilation_cache"
        )

        self.weight_streaming = args_util.get(args, "weight_streaming")

        self.tiling_optimization_level = None
        tiling_optimization_level = args_util.get(
            args, "tiling_optimization_level"
        )
        if tiling_optimization_level is not None:
            self.tiling_optimization_level = make_trt_enum_val(
                "TilingOptimizationLevel", tiling_optimization_level
            )

        # Parse compute capabilities arguments if enabled and TensorRT-RTX is available
        self.use_gpu = False
        self.compute_capabilities = None

        if self._allow_compute_capabilities and polygraphy_config.USE_TENSORRT_RTX:
            self.use_gpu = args_util.get(args, "use_gpu", default=False)
            compute_capabilities_list = args_util.get(args, "compute_capabilities")

            if compute_capabilities_list:
                # Parse compute capabilities from list of strings
                try:
                    capabilities = []
                    for cap_str in compute_capabilities_list:
                        major, minor = map(int, cap_str.split('.'))
                        capabilities.append((major, minor))
                    self.compute_capabilities = capabilities
                except ValueError:
                    G_LOGGER.critical(f"Invalid compute capabilities format: {compute_capabilities_list}. "
                                      "Expected format: space-separated 'major.minor' versions (e.g., '7.5 8.0').")

    def add_to_script_impl(self, script):
        profiles = []
        for profile_dict in self.profile_dicts:
            profile_str = "Profile()"
            for name in profile_dict.keys():
                profile_str += safe(
                    ".add({:}, min={:}, opt={:}, max={:})", name, *profile_dict[name]
                ).unwrap()
            profiles.append(profile_str)
        if profiles:
            script.add_import(imports=["Profile"], frm="polygraphy.backend.trt")
            profiles = safe(
                "[\n{tab}{:}\n]",
                inline(safe(f",\n{constants.TAB}".join(profiles))),
                tab=inline(safe(constants.TAB)),
            )
            profile_name = script.add_loader(profiles, "profiles")
        else:
            profile_name = None

        calibrator = None
        if (
            any(
                arg is not None
                for arg in [self.calibration_cache, self.calibration_base_class]
            )
            and not self.int8
        ):
            G_LOGGER.warning(
                "Some int8 calibrator options were set, but int8 precision is not enabled. "
                "Calibration options will be ignored. Please set --int8 to enable calibration. "
            )

        if self.int8:
            script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
            script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
            data_loader_name = self.arg_groups[DataLoaderArgs].add_to_script(script)
            if self.calibration_base_class:
                script.add_import(imports=tensorrt_module_and_version_string(), imp_as="trt")

            if (
                self.arg_groups[DataLoaderArgs].is_using_random_data()
                and (
                    not self.calibration_cache
                    or not os.path.exists(self.calibration_cache)
                )
                and self._allow_random_data_calib_warning
            ):
                G_LOGGER.warning(
                    "Int8 Calibration is using randomly generated input data.\n"
                    "This could negatively impact accuracy if the inference-time input data is dissimilar "
                    "to the randomly generated calibration data.\n"
                    "You may want to consider providing real data via the --data-loader-script option."
                )

            calibrator = make_invocable(
                "Calibrator",
                data_loader=(
                    data_loader_name
                    if data_loader_name
                    else inline(safe("DataLoader()"))
                ),
                cache=self.calibration_cache,
                BaseClass=self.calibration_base_class,
                quantile=self._quantile,
                regression_cutoff=self._regression_cutoff,
            )

        algo_selector = None
        if self.load_tactics is not None:
            script.add_import(imports=["TacticReplayer"], frm="polygraphy.backend.trt")
            algo_selector = make_invocable("TacticReplayer", replay=self.load_tactics)
        elif self.save_tactics is not None:
            script.add_import(imports=["TacticRecorder"], frm="polygraphy.backend.trt")
            algo_selector = make_invocable("TacticRecorder", record=self.save_tactics)

        # Add a `tensorrt` import if any argument requires direct access to the module.
        if any(
            arg is not None
            for arg in [
                self.tactic_sources,
                self.memory_pool_limits,
                self.preview_features,
                self.engine_capability,
                self.profiling_verbosity,
                self.hardware_compatibility_level,
                self.runtime_platform,
                self.quantization_flags,
                self.tiling_optimization_level,
                self.use_gpu,
                self.compute_capabilities,
            ]
        ):
            script.add_import(imports=tensorrt_module_and_version_string(), imp_as="trt")

        if self.trt_config_script is not None:
            script.add_import(
                imports=["InvokeFromScript"], frm="polygraphy.backend.common"
            )
            config_loader_str = make_invocable(
                "InvokeFromScript",
                self.trt_config_script,
                name=self.trt_config_func_name,
            )
        else:
            # Use CreateConfigRTX if TensorRT-RTX is enabled, otherwise use CreateConfig
            if polygraphy_config.USE_TENSORRT_RTX:
                config_class = "CreateConfigRTX"
                config_alias = "CreateTrtConfigRTX"
                extra_args = {
                    "use_gpu": self.use_gpu,
                    "compute_capabilities": self.compute_capabilities,
                }
            else:
                config_class = "CreateConfig"
                config_alias = "CreateTrtConfig"
                extra_args = {
                    "tf32": self.tf32,
                    "fp16": self.fp16,
                    "bf16": self.bf16,
                    "int8": self.int8,
                    "fp8": self.fp8,
                    "calibrator": calibrator,
                    "use_dla": self.use_dla,
                    "allow_gpu_fallback": self.allow_gpu_fallback,
                }

            config_loader_str = make_invocable_if_nondefault(
                config_alias,
                precision_constraints=self.precision_constraints,
                restricted=self.restricted,
                profiles=profile_name,
                load_timing_cache=self.load_timing_cache,
                algorithm_selector=algo_selector,
                sparse_weights=self.sparse_weights,
                tactic_sources=self.tactic_sources,
                memory_pool_limits=self.memory_pool_limits,
                refittable=self.refittable,
                strip_plan=self.strip_plan,
                preview_features=self.preview_features,
                engine_capability=self.engine_capability,
                direct_io=self.direct_io,
                builder_optimization_level=self.builder_optimization_level,
                hardware_compatibility_level=self.hardware_compatibility_level,
                profiling_verbosity=self.profiling_verbosity,
                max_aux_streams=self.max_aux_streams,
                version_compatible=self.version_compatible,
                exclude_lean_runtime=self.exclude_lean_runtime,
                quantization_flags=self.quantization_flags,
                error_on_timing_cache_miss=self.error_on_timing_cache_miss,
                disable_compilation_cache=self.disable_compilation_cache,
                weight_streaming=self.weight_streaming,
                runtime_platform=self.runtime_platform,
                tiling_optimization_level=self.tiling_optimization_level,
                **extra_args
            )
            
            if config_loader_str is None and polygraphy_config.USE_TENSORRT_RTX:
                config_loader_str = make_invocable(config_alias)

            if config_loader_str is not None:
                if polygraphy_config.USE_TENSORRT_RTX:
                    script.add_import(
                        imports=config_class,
                        frm="polygraphy.backend.tensorrt_rtx",
                        imp_as=config_alias,
                    )
                else:
                    script.add_import(
                        imports=config_class,
                        frm="polygraphy.backend.trt",
                        imp_as=config_alias,
                    )

        if config_loader_str is not None:
            config_loader_name = script.add_loader(
                config_loader_str, "create_trt_config"
            )
        else:
            config_loader_name = None

        if self.trt_config_postprocess_script is not None:
            # Need to set up a default config if there isn't one since `PostprocessConfig` will require a config.
            if config_loader_name is None:
                script.add_import(
                    imports="CreateConfig",
                    frm="polygraphy.backend.trt",
                    imp_as="CreateTrtConfig",
                )
                config_loader_name = script.add_loader(
                    make_invocable("CreateTrtConfig"), "create_trt_config"
                )

            script.add_import(
                imports=["InvokeFromScript"], frm="polygraphy.backend.common"
            )
            script.add_import(
                imports=["PostprocessConfig"],
                frm="polygraphy.backend.trt",
                imp_as="PostprocessTrtConfig",
            )
            func = make_invocable(
                "InvokeFromScript",
                self.trt_config_postprocess_script,
                name=self.trt_config_postprocess_func_name,
            )
            config_loader_name = script.add_loader(
                make_invocable("PostprocessTrtConfig", config_loader_name, func=func),
                "postprocess_trt_config",
            )

        return config_loader_name

    def create_config(self, builder, network):
        """
        Creates a TensorRT BuilderConfig according to arguments provided on the command-line.

        Args:
            builder (trt.Builder):
                    The TensorRT builder to use to create the configuration.
            network (trt.INetworkDefinition):
                    The TensorRT network for which to create the config. The network is used to
                    automatically create a default optimization profile if none are provided.

        Returns:
            trt.IBuilderConfig: The TensorRT builder configuration.
        """
        # Use CreateConfigRTX if TensorRT-RTX is enabled, otherwise use CreateConfig
        if polygraphy_config.USE_TENSORRT_RTX:
            from polygraphy.backend.tensorrt_rtx import CreateConfigRTX
            default_loader = CreateConfigRTX()
        else:
            from polygraphy.backend.trt import CreateConfig
            default_loader = CreateConfig()

        loader = util.default(args_util.run_script(self.add_to_script), default_loader)
        return loader(builder, network)
