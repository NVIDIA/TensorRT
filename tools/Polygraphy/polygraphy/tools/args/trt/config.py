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
import copy
import os

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import assert_identifier, inline, make_invocable, make_invocable_if_nondefault, safe


def parse_profile_shapes(default_shapes, min_args, opt_args, max_args):
    """
    Parses TensorRT profile options from command-line arguments.

    Args:
        default_shapes (TensorMetadata): The inference input shapes.

    Returns:
     List[Tuple[OrderedDict[str, Shape]]]:
            A list of profiles with each profile comprised of three dictionaries
            (min, opt, max) mapping input names to shapes.
    """

    def get_shapes(lst, idx):
        nonlocal default_shapes
        default_shapes = copy.copy(default_shapes)
        if idx < len(lst):
            default_shapes.update(args_util.parse_meta(lst[idx], includes_dtype=False))

        # Don't care about dtype, and need to override dynamic dimensions
        shapes = {name: util.override_dynamic_shape(shape) for name, (_, shape) in default_shapes.items()}

        for name, shape in shapes.items():
            if tuple(default_shapes[name].shape) != tuple(shape):
                G_LOGGER.warning(
                    "Input tensor: {:} | For TensorRT profile, overriding dynamic shape: {:} to: {:}".format(
                        name, default_shapes[name].shape, shape
                    ),
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
                "Mismatch in input names between minimum shapes ({:}) and optimum shapes "
                "({:})".format(list(min_shapes.keys()), list(opt_shapes.keys()))
            )
        elif sorted(opt_shapes.keys()) != sorted(max_shapes.keys()):
            G_LOGGER.critical(
                "Mismatch in input names between optimum shapes ({:}) and maximum shapes "
                "({:})".format(list(opt_shapes.keys()), list(max_shapes.keys()))
            )

        profiles.append((min_shapes, opt_shapes, max_shapes))
    return profiles


@mod.export()
class TrtConfigArgs(BaseArgs):
    def __init__(self, strict_types_default=None, random_data_calib_warning=True):
        """
        Args:
            strict_types_default (bool): Whether strict types should be enabled by default.
            random_data_calib_warning (bool):
                    Whether to issue a warning when randomly generated data is being used
                    for calibration.
        """
        super().__init__()
        self.model_args = None
        self.data_loader_args = None
        self._strict_types_default = strict_types_default
        self._random_data_calib_warning = random_data_calib_warning

    def add_to_parser(self, parser):
        trt_config_args = parser.add_argument_group(
            "TensorRT Builder Configuration", "Options for TensorRT Builder Configuration"
        )
        trt_config_args.add_argument(
            "--trt-min-shapes",
            action="append",
            help="The minimum shapes the optimization profile(s) will support. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-min-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )
        trt_config_args.add_argument(
            "--trt-opt-shapes",
            action="append",
            help="The shapes for which the optimization profile(s) will be most performant. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-opt-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )
        trt_config_args.add_argument(
            "--trt-max-shapes",
            action="append",
            help="The maximum shapes the optimization profile(s) will support. "
            "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
            "Format: --trt-max-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]",
            nargs="+",
            default=[],
        )

        trt_config_args.add_argument(
            "--tf32", help="Enable tf32 precision in TensorRT", action="store_true", default=None
        )
        trt_config_args.add_argument(
            "--fp16", help="Enable fp16 precision in TensorRT", action="store_true", default=None
        )
        trt_config_args.add_argument(
            "--int8",
            help="Enable int8 precision in TensorRT. "
            "If calibration is required but no calibration cache is provided, this option will cause TensorRT to run "
            "int8 calibration using the Polygraphy data loader to provide calibration data. ",
            action="store_true",
            default=None,
        )
        if self._strict_types_default:
            trt_config_args.add_argument(
                "--no-strict-types",
                help="Disables strict types in TensorRT, allowing it to choose tactics outside the "
                "layer precision set.",
                action="store_false",
                default=True,
                dest="strict_types",
            )
        else:
            trt_config_args.add_argument(
                "--strict-types",
                help="Enable strict types in TensorRT, forcing it to choose tactics based on the "
                "layer precision set, even if another precision is faster.",
                action="store_true",
                default=None,
                dest="strict_types",
            )

        trt_config_args.add_argument(
            "--sparse-weights",
            help="Enable optimizations for sparse weights in TensorRT",
            action="store_true",
            default=None,
        )

        trt_config_args.add_argument(
            "--workspace",
            metavar="BYTES",
            help="Amount of memory, in bytes, to allocate for the TensorRT builder's workspace. "
            "Optionally, use a `K`, `M`, or `G` suffix to indicate KiB, MiB, or GiB respectively."
            "For example, `--workspace=16M` is equivalent to `--workspace=16777216`",
            default=None,
        )
        trt_config_args.add_argument(
            "--calibration-cache",
            help="Path to load/save a calibration cache. "
            "Used to store calibration scales to speed up the process of int8 calibration. "
            "If the provided path does not yet exist, int8 calibration scales will be calculated and written to it during engine building. "
            "If the provided path does exist, it will be read and int8 calibration will be skipped during engine building. ",
            default=None,
        )
        trt_config_args.add_argument(
            "--calib-base-cls",
            "--calibration-base-class",
            dest="calibration_base_class",
            help="The name of the calibration base class to use. For example, 'IInt8MinMaxCalibrator'. ",
            default=None,
        )
        trt_config_args.add_argument(
            "--quantile",
            type=float,
            help="The quantile to use for IInt8LegacyCalibrator. Has no effect for other calibrator types.",
            default=None,
        )
        trt_config_args.add_argument(
            "--regression-cutoff",
            type=float,
            help="The regression cutoff to use for IInt8LegacyCalibrator. Has no effect for other calibrator types.",
            default=None,
        )

        trt_config_args.add_argument(
            "--timing-cache",
            help="Path to load/save tactic timing cache. "
            "Used to cache tactic timing information to speed up the engine building process. "
            "Existing caches will be appended to with any new timing information gathered. ",
            default=None,
        )

        replay = trt_config_args.add_mutually_exclusive_group()
        replay.add_argument(
            "--tactic-replay",
            help="[DEPRECATED - use --load/save-tactics] Path to load/save a tactic replay file. "
            "Used to record and replay tactics selected by TensorRT to provide deterministic engine builds. "
            "If the provided path does not yet exist, tactics will be recorded and written to it. "
            "If the provided path does exist, it will be read and used to replay previously recorded tactics. ",
            default=None,
        )
        replay.add_argument(
            "--save-tactics",
            help="Path to save a tactic replay file. "
            "Tactics selected by TensorRT will be recorded and stored at this location. ",
            default=None,
        )
        replay.add_argument(
            "--load-tactics",
            help="Path to load a tactic replay file. "
            "The tactics specified in the file will be used to override TensorRT's default selections. ",
            default=None,
        )

        trt_config_args.add_argument(
            "--tactic-sources",
            help="Tactic sources to enable. This controls which libraries "
            "(e.g. cudnn, cublas, etc.) TensorRT is allowed to load tactics from. "
            "Values come from the names of the values in the trt.TacticSource enum, and are case-insensitive. "
            "If no arguments are provided, e.g. '--tactic-sources', then all tactic sources are disabled.",
            nargs="*",
            default=None,
        )

        trt_config_args.add_argument(
            "--trt-config-script",
            help="Path to a Python script that defines a function that creates a "
            "TensorRT IBuilderConfig. The function should take a builder and network as parameters and return a "
            "TensorRT builder configuration. When this option is specified, all other config arguments are ignored. ",
            default=None,
        )
        trt_config_args.add_argument(
            "--trt-config-func-name",
            help="When using a trt-config-script, this specifies the name of the function "
            "that creates the config. Defaults to `load_config`. ",
            default="load_config",
        )
        trt_config_args.add_argument(
            "--trt-safety-restricted",
            help="Enable safety scope checking in TensorRT",
            action="store_true",
            default=None,
            dest="restricted",
        )
        trt_config_args.add_argument(
            "--use-dla",
            help="[EXPERIMENTAL] Use DLA as the default device type",
            action="store_true",
            default=None,
        )
        trt_config_args.add_argument(
            "--allow-gpu-fallback",
            help="[EXPERIMENTAL] Allow layers unsupported on the DLA to fall back to GPU. Has no effect if --dla is not set.",
            action="store_true",
            default=None,
        )

    def register(self, maker):
        from polygraphy.tools.args.data_loader import DataLoaderArgs
        from polygraphy.tools.args.model import ModelArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, DataLoaderArgs):
            self.data_loader_args = maker

    def parse(self, args):
        trt_min_shapes = args_util.get(args, "trt_min_shapes", default=[])
        trt_max_shapes = args_util.get(args, "trt_max_shapes", default=[])
        trt_opt_shapes = args_util.get(args, "trt_opt_shapes", default=[])

        default_shapes = TensorMetadata()
        if self.model_args is not None:
            assert hasattr(self.model_args, "input_shapes"), "ModelArgs must be parsed before TrtConfigArgs!"
            default_shapes = self.model_args.input_shapes

        self.profile_dicts = parse_profile_shapes(default_shapes, trt_min_shapes, trt_opt_shapes, trt_max_shapes)

        self.workspace = args_util.parse_num_bytes(args_util.get(args, "workspace"))

        self.tf32 = args_util.get(args, "tf32")
        self.fp16 = args_util.get(args, "fp16")
        self.int8 = args_util.get(args, "int8")
        self.strict_types = args_util.get(args, "strict_types")
        self.restricted = args_util.get(args, "restricted")

        self.calibration_cache = args_util.get(args, "calibration_cache")
        calib_base = args_util.get(args, "calibration_base_class")
        self.calibration_base_class = None
        if calib_base is not None:
            calib_base = safe(assert_identifier(calib_base))
            self.calibration_base_class = inline(safe("trt.{:}", inline(calib_base)))

        self.quantile = args_util.get(args, "quantile")
        self.regression_cutoff = args_util.get(args, "regression_cutoff")

        self.sparse_weights = args_util.get(args, "sparse_weights")
        self.timing_cache = args_util.get(args, "timing_cache")

        tactic_replay = args_util.get(args, "tactic_replay")
        self.load_tactics = args_util.get(args, "load_tactics")
        self.save_tactics = args_util.get(args, "save_tactics")
        if tactic_replay is not None:
            mod.warn_deprecated("--tactic-replay", "--save-tactics or --load-tactics", remove_in="0.35.0")
            G_LOGGER.warning("--tactic-replay is deprecated. Use either --save-tactics or --load-tactics instead.")
            if os.path.exists(tactic_replay) and util.get_file_size(tactic_replay) > 0:
                self.load_tactics = tactic_replay
            else:
                self.save_tactics = tactic_replay

        tactic_sources = args_util.get(args, "tactic_sources")
        self.tactic_sources = None
        if tactic_sources is not None:
            self.tactic_sources = []
            for source in tactic_sources:
                source = safe(assert_identifier(source.upper()))
                source_str = safe("trt.TacticSource.{:}", inline(source))
                self.tactic_sources.append(inline(source_str))

        self.trt_config_script = args_util.get(args, "trt_config_script")
        self.trt_config_func_name = args_util.get(args, "trt_config_func_name")

        self.use_dla = args_util.get(args, "use_dla")
        self.allow_gpu_fallback = args_util.get(args, "allow_gpu_fallback")

    def add_trt_config_loader(self, script):
        profiles = []
        for (min_shape, opt_shape, max_shape) in self.profile_dicts:
            profile_str = "Profile()"
            for name in min_shape.keys():
                profile_str += safe(
                    ".add({:}, min={:}, opt={:}, max={:})", name, min_shape[name], opt_shape[name], max_shape[name]
                ).unwrap()
            profiles.append(profile_str)
        if profiles:
            script.add_import(imports=["Profile"], frm="polygraphy.backend.trt")
            profiles = safe("[\n\t{:}\n]", inline(safe(",\n\t".join(profiles))))
            profile_name = script.add_loader(profiles, "profiles")
        else:
            profile_name = None

        calibrator = None
        if any(arg is not None for arg in [self.calibration_cache, self.calibration_base_class]) and not self.int8:
            G_LOGGER.warning(
                "Some int8 calibrator options were set, but int8 precision is not enabled. "
                "Calibration options will be ignored. Please set --int8 to enable calibration. "
            )

        if self.int8 and self.data_loader_args is not None:  # We cannot do calibration if there is no data loader.
            script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
            script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
            data_loader_name = self.data_loader_args.add_data_loader(script)
            if self.calibration_base_class:
                script.add_import(imports=["tensorrt as trt"])

            if (
                self.data_loader_args.is_using_random_data()
                and (not self.calibration_cache or not os.path.exists(self.calibration_cache))
                and self._random_data_calib_warning
            ):
                G_LOGGER.warning(
                    "Int8 Calibration is using randomly generated input data.\n"
                    "This could negatively impact accuracy if the inference-time input data is dissimilar "
                    "to the randomly generated calibration data.\n"
                    "You may want to consider providing real data via the --data-loader-script option."
                )

            calibrator = make_invocable(
                "Calibrator",
                data_loader=data_loader_name if data_loader_name else inline(safe("DataLoader()")),
                cache=self.calibration_cache,
                BaseClass=self.calibration_base_class,
                quantile=self.quantile,
                regression_cutoff=self.regression_cutoff,
            )

        algo_selector = None
        if self.load_tactics is not None:
            script.add_import(imports=["TacticReplayer"], frm="polygraphy.backend.trt")
            algo_selector = make_invocable("TacticReplayer", replay=self.load_tactics)
        elif self.save_tactics is not None:
            script.add_import(imports=["TacticRecorder"], frm="polygraphy.backend.trt")
            algo_selector = make_invocable("TacticRecorder", record=self.save_tactics)

        if self.tactic_sources is not None:
            script.add_import(imports=["tensorrt as trt"])

        if self.trt_config_script is not None:
            script.add_import(imports=["InvokeFromScript"], frm="polygraphy.backend.common")
            config_loader_str = make_invocable(
                "InvokeFromScript", self.trt_config_script, name=self.trt_config_func_name
            )
        else:
            config_loader_str = make_invocable_if_nondefault(
                "CreateTrtConfig",
                max_workspace_size=self.workspace,
                tf32=self.tf32,
                fp16=self.fp16,
                int8=self.int8,
                strict_types=self.strict_types,
                restricted=self.restricted,
                profiles=profile_name,
                calibrator=calibrator,
                load_timing_cache=(
                    self.timing_cache if self.timing_cache and os.path.exists(self.timing_cache) else None
                ),
                algorithm_selector=algo_selector,
                sparse_weights=self.sparse_weights,
                tactic_sources=self.tactic_sources,
                use_dla=self.use_dla,
                allow_gpu_fallback=self.allow_gpu_fallback,
            )
            if config_loader_str is not None:
                script.add_import(imports=["CreateConfig as CreateTrtConfig"], frm="polygraphy.backend.trt")

        if config_loader_str is not None:
            config_loader_name = script.add_loader(config_loader_str, "create_trt_config")
        else:
            config_loader_name = None
        return config_loader_name

    def create_config(self, builder, network):
        from polygraphy.backend.trt import CreateConfig

        loader = util.default(args_util.run_script(self.add_trt_config_loader), CreateConfig())
        return loader(builder, network)
