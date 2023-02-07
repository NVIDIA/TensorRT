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
"""
This file defines the `TrtexecRunnerArgs` argument group, which manages
command-line options that control the `TrtexecRunner` runner.

The argument group implements the standard `BaseRunnerArgs` interface, which inherits from `BaseArgs`.
"""

import polygraphy
from polygraphy import mod
from polygraphy.tools.args import ModelArgs, TrtConfigArgs, TrtLoadPluginsArgs, TrtLoadNetworkArgs, TrtSaveEngineArgs, util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.script import make_invocable

@mod.export()
class TrtexecRunnerArgs(BaseRunnerArgs):
    """
    Trtexec Runner Inference: running inference with the trtexec backend.

    Depends on:
        ModelArgs
        TrtConfigArgs
        TrtLoadPluginsArgs
        TrtSaveEngineArgs
    """

    def get_name_opt_impl(self):
        return "Trtexec Runner", "trtexec"

    def add_parser_args_impl(self):
        """
        Add command-line arguments that trtexec supports
        """

        self.group.add_argument(
            "--trtexec-path",
            help="Path to find trtexec binary. By default, it expects to find it in PATH",
            default=None,
        )

        self.group.add_argument(
            "--use-cuda-graph",
            help="Use CUDA graph to capture engine execution and then launch inference (default = disabled). This flag may be ignored if the graph capture fails",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--avg-runs",
            help="Report performance measurements averaged over N consecutive iterations (default = 10)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--best",
            help="Enable all precisions to achieve the best performance (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--duration",
            help="Run performance measurements for at least N seconds wallclock time (default = 3)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--device",
            help="Select cuda device N (default = 0)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--streams",
            help="Instantiate N engines to use concurrently (default = 1)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--min-timing",
            help="Set the minimum number of iterations used in kernel selection (default = 1)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--avg-timing",
            help="Set the number of times averaged in each iteration for kernel selection (default = 8)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--expose-dma",
            help="Serialize DMA transfers to and from device (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--no-data-transfers",
            help="Disable DMA transfers to and from device (default = enabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--trtexec-warmup",
            help="Run for N milliseconds on trtexec to warmup before measuring performance (default = 200)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--trtexec-iterations",
            help="Run at least N inference iterations on trtexec (default = 10)",
            default=None,
            type=int,
        )

        self.group.add_argument(
            "--trtexec-export-times",
            help="Write the timing results in a json file",
            default=None,
        )

        self.group.add_argument(
            "--trtexec-export-output",
            help="Write the output tensors to a json file",
            default=None,
        )

        self.group.add_argument(
            "--trtexec-export-profile",
            help="Write the profile information per layer in a json file",
            default=None,
        )

        self.group.add_argument(
            "--trtexec-export-layer-info",
            help="Write the layer information of the engine in a json file",
            default=None,
        )

        # Optional

        self.group.add_argument(
            "--use-spin-wait",
            help="Actively synchronize on GPU events. This option may decrease synchronization time but increase CPU usage and power (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--threads",
            help="Enable multithreading to drive engines with independent threads or speed up refitting (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--use-managed-memory",
            help="Use managed memory instead of separate host and device allocations (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--dump-refit",
            help="Print the refittable layers and weights from a refittable engine",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--dump-output",
            help="Print the output tensor(s) of the last inference iteration (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--dump-profile",
            help="Print profile information per layer (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--dump-layer-info",
            help="Print layer information of the engine to console (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--separate-profile-run",
            help="Do not attach the profiler in the benchmark run; if profiling is enabled, a second profile run will be executed (default = disabled)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--trtexec-no-builder-cache",
            help="Disable timing cache in builder (default is to enable timing cache)",
            action="store_true",
            default=False,
        )

        self.group.add_argument(
            "--trtexec-profiling-verbosity",
            help="Specify profiling verbosity. mode ::= layer_names_only|detailed|none (default = layer_names_only)",
            default=False,
        )

        self.group.add_argument(
            "--layer-output-types",
            help="""Control per-layer output type constraints. Effective only when precisionConstraints is set to
                              "obey" or "prefer". (default = none)"
                              The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a
                              layerName to specify the default precision for all the unspecified layers. If a layer has more than
                              one output, then multiple types separated by "+" can be provided for this layer.
                              Per-layer output type spec ::= layerOutputTypes[","spec]
                                                    layerOutputTypes ::= layerName":"type
                                                    type ::= "fp32"|"fp16"|"int32"|"int8"["+"type]",
                """,
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:
        """

        # Required options
        self.trtexec_path = args_util.get(args, "trtexec_path")
        self.use_cuda_graph = args_util.get(args, "use_cuda_graph")
        self.avg_runs = args_util.get(args, "avg_runs")
        self.best = args_util.get(args, "best")
        self.duration = args_util.get(args, "duration")
        self.device = args_util.get(args, "device")
        self.streams = args_util.get(args, "streams")
        self.min_timing = args_util.get(args, "min_timing")
        self.avg_timing = args_util.get(args, "avg_timing")
        self.expose_dma = args_util.get(args, "expose_dma")
        self.no_data_transfers = args_util.get(args, "no_data_transfers")
        self.trtexec_warmup = args_util.get(args, "trtexec_warmup")
        self.trtexec_iterations = args_util.get(args, "trtexec_iterations")
        self.trtexec_export_times = args_util.get(args, "trtexec_export_times")
        self.trtexec_export_output = args_util.get(args, "trtexec_export_output")
        self.trtexec_export_profile = args_util.get(args, "trtexec_export_profile")
        self.trtexec_export_layer_info = args_util.get(args, "trtexec_export_layer_info")

        # Optional options
        self.use_spin_wait = args_util.get(args, "use_spin_wait")
        self.threads = args_util.get(args, "threads")
        self.use_managed_memory = args_util.get(args, "use_managed_memory")
        self.dump_refit = args_util.get(args, "dump_refit")
        self.dump_output = args_util.get(args, "dump_output")
        self.dump_profile = args_util.get(args, "dump_profile")
        self.dump_layer_info = args_util.get(args, "dump_layer_info")
        self.separate_profile_run = args_util.get(args, "separate_profile_run")
        self.trtexec_no_builder_cache = args_util.get(args, "trtexec_no_builder_cache")
        self.trtexec_profiling_verbosity = args_util.get(args, "trtexec_profiling_verbosity")
        self.layer_output_types = args_util.get(args, "layer_output_types")

    def add_to_script_impl(self, script):
        model_path = self.arg_groups[ModelArgs].path
        model_type = self.arg_groups[ModelArgs].model_type
        input_shapes = self.arg_groups[ModelArgs].input_shapes or None

        profile_dicts = self.arg_groups[TrtConfigArgs].profile_dicts
        tf32 = self.arg_groups[TrtConfigArgs].tf32
        fp16 = self.arg_groups[TrtConfigArgs].fp16
        int8 = self.arg_groups[TrtConfigArgs].int8
        allow_gpu_fallback = self.arg_groups[TrtConfigArgs].allow_gpu_fallback
        precision_constraints = self.arg_groups[TrtConfigArgs].precision_constraints
        workspace = self.arg_groups[TrtConfigArgs].workspace
        use_dla = self.arg_groups[TrtConfigArgs].use_dla
        if mod.version(polygraphy.__version__) >= mod.version('0.39.0'):
            refit = self.arg_groups[TrtConfigArgs].refittable
        else:
            refit = None

        plugins = self.arg_groups[TrtLoadPluginsArgs].plugins
        layer_precisions = self.arg_groups[TrtLoadNetworkArgs].layer_precisions
        if layer_precisions:
            layer_precisions = {layer:str(precision) for (layer, precision) in layer_precisions.items()}

        save_engine = self.arg_groups[TrtSaveEngineArgs].path

        # Add an import for the Trtexec runner.
        script.add_import(imports=["TrtexecRunner"], frm="polygraphy_trtexec.backend")
        # Add the Trtexec runner using the `Script.add_runner()` API.
        script.add_runner(make_invocable(
            "TrtexecRunner",
            model_path=model_path,
            model_type=model_type,

            trtexec_path = self.trtexec_path,
            use_cuda_graph=self.use_cuda_graph,
            avg_runs=self.avg_runs,
            best=self.best,
            duration=self.duration,
            device=self.device,
            streams=self.streams,
            min_timing=self.min_timing,
            avg_timing=self.avg_timing,
            expose_dma=self.expose_dma,
            no_data_transfers=self.no_data_transfers,
            trtexec_warmup=self.trtexec_warmup,
            trtexec_iterations=self.trtexec_iterations,
            trtexec_export_times=self.trtexec_export_times,
            trtexec_export_output=self.trtexec_export_output,
            trtexec_export_profile=self.trtexec_export_profile,
            trtexec_export_layer_info=self.trtexec_export_layer_info,

            # Optional
            use_spin_wait=self.use_spin_wait,
            threads=self.threads,
            use_managed_memory=self.use_managed_memory,
            dump_refit=self.dump_refit,
            dump_output=self.dump_output,
            dump_profile=self.dump_profile,
            dump_layer_info=self.dump_layer_info,
            refit=refit,
            separate_profile_run=self.separate_profile_run,
            trtexec_no_builder_cache=self.trtexec_no_builder_cache,
            trtexec_profiling_verbosity=self.trtexec_profiling_verbosity,
            layer_output_types=self.layer_output_types,

            input_shapes=input_shapes,
            profile_dicts=profile_dicts,
            tf32=tf32,
            fp16=fp16,
            int8=int8,
            allow_gpu_fallback=allow_gpu_fallback,
            precision_constraints=precision_constraints,
            workspace=workspace,
            use_dla=use_dla,
            layer_precisions=layer_precisions,
            plugins=plugins,
            save_engine=save_engine,
            ))
