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

from polygraphy import mod, util
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.args.backend.onnx.loader import OnnxLoadArgs
from polygraphy.tools.args.backend.trt.config import TrtConfigArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class TrtLoadPluginsArgs(BaseArgs):
    """
    TensorRT Plugin Loading: loading TensorRT plugins.
    """

    def add_parser_args_impl(self):
        self.group.add_argument("--plugins", help="Path(s) of plugin libraries to load", nargs="+", default=None)

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            plugins (List[str]): Path(s) to plugin libraries.
        """
        self.plugins = args_util.get(args, "plugins")

    # If plugins are present, wrap the provided loader/object with LoadPlugins
    def add_to_script_impl(self, script, loader_name: str):
        """
        Args:
            loader_name (str):
                    The name of the loader which should be consumed by the ``LoadPlugins`` loader.
        """
        if self.plugins:
            script.add_import(imports=["LoadPlugins"], frm="polygraphy.backend.trt")
            loader_str = make_invocable("LoadPlugins", plugins=self.plugins, obj=loader_name)
            loader_name = script.add_loader(loader_str, "load_plugins")
        return loader_name


@mod.export()
class TrtLoadNetworkArgs(BaseArgs):
    """
    TensorRT Network Loading: loading TensorRT networks.

    Depends on:

        - ModelArgs
        - TrtLoadPluginsArgs
        - OnnxLoadArgs: if allow_onnx_loading == True
    """

    def __init__(self, allow_custom_outputs: bool = None, allow_onnx_loading: bool = None):
        """
        Args:
            allow_custom_outputs (bool):
                    Whether to allow marking custom output tensors.
                    Defaults to True.
            allow_onnx_loading (bool):
                    Whether to allow parsing networks from an ONNX model.
                    Defaults to True.
        """
        super().__init__()
        self._allow_custom_outputs = util.default(allow_custom_outputs, True)
        self._allow_onnx_loading = util.default(allow_onnx_loading, True)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--explicit-precision",
            help="[DEPRECATED] Enable explicit precision mode",
            action="store_true",
            default=None,
        )
        if self._allow_custom_outputs:
            self.group.add_argument(
                "--trt-outputs",
                help="Name(s) of TensorRT output(s). "
                "Using '--trt-outputs mark all' indicates that all tensors should be used as outputs",
                nargs="+",
                default=None,
            )
            self.group.add_argument(
                "--trt-exclude-outputs",
                help="[EXPERIMENTAL] Name(s) of TensorRT output(s) to unmark as outputs.",
                nargs="+",
                default=None,
            )
        self.group.add_argument(
            "--trt-network-func-name",
            help="When using a trt-network-script instead of other model types, this specifies the name "
            "of the function that loads the network. Defaults to `load_network`.",
            default="load_network",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            outputs (List[str]): Names of output tensors.
            exclude_outputs (List[str]): Names of tensors which should be unmarked as outputs.
            trt_network_func_name (str): The name of the function in a custom network script that creates the network.
        """
        self.outputs = args_util.get_outputs(args, "trt_outputs")

        self.explicit_precision = args_util.get(args, "explicit_precision")
        if self.explicit_precision is not None:
            mod.warn_deprecated("--explicit-precision", use_instead=None, remove_in="0.42.0", always_show_warning=True)

        self.exclude_outputs = args_util.get(args, "trt_exclude_outputs")
        self.trt_network_func_name = args_util.get(args, "trt_network_func_name")

    def add_to_script_impl(self, script):
        model_file = self.arg_groups[ModelArgs].path
        model_type = self.arg_groups[ModelArgs].model_type
        outputs = args_util.get_outputs_for_script(script, self.outputs)

        if model_type == "trt-network-script":
            script.add_import(imports=["InvokeFromScript"], frm="polygraphy.backend.common")
            loader_str = make_invocable("InvokeFromScript", model_file, name=self.trt_network_func_name)
            loader_name = script.add_loader(loader_str, "load_network")
        elif self._allow_onnx_loading:
            if self.arg_groups[OnnxLoadArgs].must_use_onnx_loader(disable_custom_outputs=True):
                # When loading from ONNX, we need to disable custom outputs since TRT requires dtypes on outputs,
                # which our marking function doesn't guarantee.
                script.add_import(imports=["NetworkFromOnnxBytes"], frm="polygraphy.backend.trt")
                onnx_loader = self.arg_groups[OnnxLoadArgs].add_to_script(
                    script, disable_custom_outputs=True, serialize_model=True
                )
                loader_str = make_invocable(
                    "NetworkFromOnnxBytes",
                    self.arg_groups[TrtLoadPluginsArgs].add_to_script(script, onnx_loader),
                    explicit_precision=self.explicit_precision,
                )
                loader_name = script.add_loader(loader_str, "parse_network_from_onnx")
            else:
                script.add_import(imports=["NetworkFromOnnxPath"], frm="polygraphy.backend.trt")
                loader_str = make_invocable(
                    "NetworkFromOnnxPath",
                    self.arg_groups[TrtLoadPluginsArgs].add_to_script(script, model_file),
                    explicit_precision=self.explicit_precision,
                )
                loader_name = script.add_loader(loader_str, "parse_network_from_onnx")
        else:
            G_LOGGER.internal_error("Loading from ONNX is not enabled and a network script was not provided!")

        MODIFY_NETWORK = "ModifyNetworkOutputs"
        modify_network_str = make_invocable(
            MODIFY_NETWORK, loader_name, outputs=outputs, exclude_outputs=self.exclude_outputs
        )
        if str(modify_network_str) != str(make_invocable(MODIFY_NETWORK, loader_name)):
            script.add_import(imports=[MODIFY_NETWORK], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(modify_network_str, "modify_network")

        return loader_name

    def load_network(self):
        """
        Loads a TensorRT Network model according to arguments provided on the command-line.

        Returns:
            tensorrt.INetworkDefinition
        """
        loader = args_util.run_script(self.add_to_script)
        return loader()


@mod.export()
class TrtSaveEngineArgs(BaseArgs):
    """
    TensorRT Engine Saving: saving TensorRT engines.
    """

    def __init__(self, output_opt: str = None, output_short_opt: str = None):
        """
        Args:
            output_opt (str):
                    The name of the output path option.
                    Defaults to "output".
                    Use a value of ``False`` to disable the option.
            output_short_opt (str):
                    The short option to use for the output path.
                    Defaults to "-o".
                    Use a value of ``False`` to disable the short option.
        """
        super().__init__()
        self._output_opt = util.default(output_opt, "output")
        self._output_short_opt = util.default(output_short_opt, "-o")

    def add_parser_args_impl(self):
        if self._output_opt:
            params = ([self._output_short_opt] if self._output_short_opt else []) + [f"--{self._output_opt}"]
            self.group.add_argument(*params, help="Path to save the TensorRT Engine", dest="save_engine", default=None)

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            path (str): The path at which to save the TensorRT engine.
        """
        self.path = args_util.get(args, "save_engine")

    def add_to_script_impl(self, script, loader_name):
        """
        Args:
            loader_name (str):
                    The name of the loader which should be consumed by the ``SaveEngine`` loader.

        Returns:
            str: The name of the ``SaveEngine`` loader added to the script.
        """
        if self.path is None:
            return loader_name

        script.add_import(imports=["SaveEngine"], frm="polygraphy.backend.trt")
        return script.add_loader(make_invocable("SaveEngine", loader_name, path=self.path), "save_engine")

    def save_engine(self, engine, path=None):
        """
        Saves a TensorRT engine according to arguments provided on the command-line.

        Args:
            model (onnx.ModelProto): The TensorRT engine to save.

            path (str):
                    The path at which to save the engine.
                    If no path is provided, it is determined from command-line arguments.

        Returns:
            tensorrt.ICudaEngine: The engine that was saved.
        """
        with util.TempAttrChange(self, {"path": path}):
            loader = args_util.run_script(self.add_to_script, engine)
            return loader()


@mod.export()
class TrtLoadEngineArgs(BaseArgs):
    """
    TensorRT Engine: loading TensorRT engines.

    Depends on:

        - ModelArgs
        - TrtLoadPluginsArgs
        - TrtLoadNetworkArgs: if support for building engines is required
        - TrtConfigArgs: if support for building engines is required
        - TrtSaveEngineArgs: if allow_saving == True
    """

    def __init__(self, allow_saving: bool = None):
        """
        Args:
            allow_saving (bool):
                    Whether to allow loaded models to be saved.
                    Defaults to False.
        """
        super().__init__()
        self._allow_saving = util.default(allow_saving, False)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--save-timing-cache",
            help="Path to save tactic timing cache if building an engine. "
            "Existing caches will be appended to with any new timing information gathered. ",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            save_timing_cache (str): Path at which to save the tactic timing cache.
        """
        self.save_timing_cache = args_util.get(args, "save_timing_cache")

    def add_to_script_impl(self, script, network_name=None):
        """
        Args:
            network_name (str): The name of a variable in the script pointing to a network loader.
        """
        if self.arg_groups[ModelArgs].model_type == "engine":
            script.add_import(imports=["EngineFromBytes"], frm="polygraphy.backend.trt")
            script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")

            load_engine = script.add_loader(
                make_invocable("BytesFromPath", self.arg_groups[ModelArgs].path), "load_engine_bytes"
            )
            return script.add_loader(
                make_invocable(
                    "EngineFromBytes", self.arg_groups[TrtLoadPluginsArgs].add_to_script(script, load_engine)
                ),
                "deserialize_engine",
            )

        network_loader_name = network_name
        if network_loader_name is None:
            if TrtLoadNetworkArgs not in self.arg_groups:
                G_LOGGER.internal_error("TrtNetworkLoaderArgs is required for engine building!")
            network_loader_name = self.arg_groups[TrtLoadNetworkArgs].add_to_script(script)

        if TrtConfigArgs not in self.arg_groups:
            G_LOGGER.internal_error("TrtConfigArgs is required for engine building!")

        script.add_import(imports=["EngineFromNetwork"], frm="polygraphy.backend.trt")
        config_loader_name = self.arg_groups[TrtConfigArgs].add_to_script(script)
        loader_str = make_invocable(
            "EngineFromNetwork",
            self.arg_groups[TrtLoadPluginsArgs].add_to_script(script, network_loader_name),
            config=config_loader_name,
            # Needed to support legacy --timing-cache argument
            save_timing_cache=self.save_timing_cache or self.arg_groups[TrtConfigArgs].timing_cache,
        )
        loader_name = script.add_loader(loader_str, "build_engine")

        if self._allow_saving:
            loader_name = self.arg_groups[TrtSaveEngineArgs].add_to_script(script, loader_name)
        return loader_name

    def load_engine(self, network=None):
        """
        Loads a TensorRT engine according to arguments provided on the command-line.

        Args:
            network (Tuple[trt.Builder, trt.INetworkDefinition, Optional[parser]]):
                    A tuple containing a TensorRT builder, network and optionally parser.

        Returns:
            tensorrt.ICudaEngine: The engine.
        """
        loader = args_util.run_script(self.add_to_script, network)
        return loader()
