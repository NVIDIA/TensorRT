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

from polygraphy import mod, util
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class TrtPluginLoaderArgs(BaseArgs):
    def add_to_parser(self, parser):
        trt_args = parser.add_argument_group("TensorRT Plugin Loader", "Options for TensorRT Plugin Loader")
        trt_args.add_argument("--plugins", help="Path(s) of plugin libraries to load", nargs="+", default=None)

    def parse(self, args):
        self.plugins = args_util.get(args, "plugins")

    # If plugins are present, wrap the provided loader/object with LoadPlugins
    def wrap_if_plugins(self, script, loader_name):
        if self.plugins:
            script.add_import(imports=["LoadPlugins"], frm="polygraphy.backend.trt")
            loader_str = make_invocable("LoadPlugins", plugins=self.plugins, obj=loader_name)
            loader_name = script.add_loader(loader_str, "load_plugins")
        return loader_name


@mod.export()
class TrtNetworkLoaderArgs(BaseArgs):
    def __init__(self, outputs=True):
        super().__init__()
        self.onnx_loader_args = None

        self._outputs = outputs

    def add_to_parser(self, parser):
        trt_args = parser.add_argument_group("TensorRT Network Loader", "Options for TensorRT Network Loader")
        trt_args.add_argument(
            "--explicit-precision", help="Enable explicit precision mode", action="store_true", default=None
        )
        if self._outputs:
            trt_args.add_argument(
                "--trt-outputs",
                help="Name(s) of TensorRT output(s). "
                "Using '--trt-outputs mark all' indicates that all tensors should be used as outputs",
                nargs="+",
                default=None,
            )
            trt_args.add_argument(
                "--trt-exclude-outputs",
                help="[EXPERIMENTAL] Name(s) of TensorRT output(s) to unmark as outputs.",
                nargs="+",
                default=None,
            )
        trt_args.add_argument(
            "--trt-network-func-name",
            help="When using a trt-network-script instead of other model types, this specifies the name "
            "of the function that loads the network. Defaults to `load_network`.",
            default="load_network",
        )

    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs
        from polygraphy.tools.args.trt.config import TrtConfigArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, TrtConfigArgs):
            self.trt_config_args = maker
        if isinstance(maker, TrtPluginLoaderArgs):
            self.trt_plugin_args = maker

    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_plugin_args is not None, "TrtPluginLoaderArgs is required!"

    def parse(self, args):
        self.outputs = args_util.get_outputs(args, "trt_outputs")
        self.explicit_precision = args_util.get(args, "explicit_precision")
        self.exclude_outputs = args_util.get(args, "trt_exclude_outputs")
        self.trt_network_func_name = args_util.get(args, "trt_network_func_name")

    def add_trt_network_loader(self, script):
        model_file = self.model_args.model_file
        model_type = self.model_args.model_type
        outputs = args_util.get_outputs_for_script(script, self.outputs)

        if model_type == "trt-network-script":
            script.add_import(imports=["InvokeFromScript"], frm="polygraphy.backend.common")
            loader_str = make_invocable("InvokeFromScript", model_file, name=self.trt_network_func_name)
            loader_name = script.add_loader(loader_str, "load_network")
        # When loading from ONNX, we need to disable custom outputs since TRT requires dtypes on outputs, which our marking function doesn't guarantee.
        elif self.onnx_loader_args is not None and self.onnx_loader_args.should_use_onnx_loader(
            disable_custom_outputs=True
        ):
            script.add_import(imports=["NetworkFromOnnxBytes"], frm="polygraphy.backend.trt")
            onnx_loader = self.onnx_loader_args.add_serialized_onnx_loader(script, disable_custom_outputs=True)
            loader_str = make_invocable(
                "NetworkFromOnnxBytes",
                self.trt_plugin_args.wrap_if_plugins(script, onnx_loader),
                explicit_precision=self.explicit_precision,
            )
            loader_name = script.add_loader(loader_str, "parse_network_from_onnx")
        else:
            script.add_import(imports=["NetworkFromOnnxPath"], frm="polygraphy.backend.trt")
            loader_str = make_invocable(
                "NetworkFromOnnxPath",
                self.trt_plugin_args.wrap_if_plugins(script, model_file),
                explicit_precision=self.explicit_precision,
            )
            loader_name = script.add_loader(loader_str, "parse_network_from_onnx")

        MODIFY_NETWORK = "ModifyNetworkOutputs"
        modify_network_str = make_invocable(
            MODIFY_NETWORK, loader_name, outputs=outputs, exclude_outputs=self.exclude_outputs
        )
        if str(modify_network_str) != str(make_invocable(MODIFY_NETWORK, loader_name)):
            script.add_import(imports=[MODIFY_NETWORK], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(modify_network_str, "modify_network")

        return loader_name

    def get_network_loader(self):
        return args_util.run_script(self.add_trt_network_loader)

    def load_network(self):
        return self.get_network_loader()()


@mod.export()
class TrtEngineSaveArgs(BaseArgs):
    def __init__(self, output="output", short_opt="-o"):
        super().__init__()
        self._output = output
        self._short_opt = short_opt

    def add_to_parser(self, parser):
        if self._output:
            self.group = parser.add_argument_group(
                "TensorRT Engine Save Options", "Options for saving TensorRT engines"
            )
            flag = "--{:}".format(self._output)
            short = self._short_opt or flag
            self.group.add_argument(
                short, flag, help="Path to save the TensorRT Engine", dest="save_engine", default=None
            )

    def parse(self, args):
        self.path = args_util.get(args, "save_engine")

    def add_save_engine(self, script, loader_name):
        if self.path is None:
            return loader_name

        script.add_import(imports=["SaveEngine"], frm="polygraphy.backend.trt")
        return script.add_loader(make_invocable("SaveEngine", loader_name, path=self.path), "save_engine")

    def save_engine(self, engine, path=None):
        with util.TempAttrChange(self, "path", path):
            loader = args_util.run_script(self.add_save_engine, engine)
            return loader()


@mod.export()
class TrtEngineLoaderArgs(BaseArgs):
    def __init__(self, save=False):
        super().__init__()
        self.trt_engine_save_args = None
        self._save = save

    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.trt.config import TrtConfigArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, TrtConfigArgs):
            self.trt_config_args = maker
        if isinstance(maker, TrtNetworkLoaderArgs):
            self.trt_network_loader_args = maker
        if isinstance(maker, TrtPluginLoaderArgs):
            self.trt_plugin_args = maker
        if self._save and isinstance(maker, TrtEngineSaveArgs):
            self.trt_engine_save_args = maker

    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_plugin_args is not None, "TrtPluginLoaderArgs is required!"
        assert not self._save or self.trt_engine_save_args is not None, "TrtEngineSaveArgs is required to use save=True"

    def parse(self, args):
        self.plugins = args_util.get(args, "plugins")

    def add_trt_serialized_engine_loader(self, script):
        assert self.model_args is not None, "ModelArgs is required for engine deserialization!"

        script.add_import(imports=["EngineFromBytes"], frm="polygraphy.backend.trt")
        script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")

        load_engine = script.add_loader(
            make_invocable("BytesFromPath", self.model_args.model_file), "load_engine_bytes"
        )
        return script.add_loader(
            make_invocable("EngineFromBytes", self.trt_plugin_args.wrap_if_plugins(script, load_engine)),
            "deserialize_engine",
        )

    def add_trt_build_engine_loader(self, script, network_name=None):
        if network_name:
            network_loader_name = network_name
        else:
            assert self.trt_network_loader_args is not None, "TrtNetworkLoaderArgs is required for engine building!"
            network_loader_name = self.trt_network_loader_args.add_trt_network_loader(script)

        assert self.trt_config_args is not None, "TrtConfigArgs is required for engine building!"

        script.add_import(imports=["EngineFromNetwork"], frm="polygraphy.backend.trt")
        config_loader_name = self.trt_config_args.add_trt_config_loader(script)
        loader_str = make_invocable(
            "EngineFromNetwork",
            self.trt_plugin_args.wrap_if_plugins(script, network_loader_name),
            config=config_loader_name,
            save_timing_cache=self.trt_config_args.timing_cache,
        )
        loader_name = script.add_loader(loader_str, "build_engine")

        if self.trt_engine_save_args is not None:
            loader_name = self.trt_engine_save_args.add_save_engine(script, loader_name)
        return loader_name

    def build_engine(self, network=None):
        loader = args_util.run_script(self.add_trt_build_engine_loader, network)
        return loader()

    def load_serialized_engine(self):
        loader = args_util.run_script(self.add_trt_serialized_engine_loader)
        return loader()
