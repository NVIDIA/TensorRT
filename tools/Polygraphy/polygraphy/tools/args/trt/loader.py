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
from polygraphy.common import constants
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Inline, Script
from polygraphy.util import misc


# FIXME: This should be split into separate network and config argument groups.
class TrtLoaderArgs(BaseArgs):
    def __init__(self, config=True, outputs=True, network_api=False):
        self._config = config
        self._outputs = outputs
        self._network_api = network_api


    def add_to_parser(self, parser):
        trt_args = parser.add_argument_group("TensorRT", "Options for TensorRT")
        if self._config:
            trt_args.add_argument("--trt-min-shapes", action='append', help="The minimum shapes the optimization profile(s) will support. "
                                "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                                "Format: --trt-min-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])
            trt_args.add_argument("--trt-opt-shapes", action='append', help="The shapes for which the optimization profile(s) will be most performant. "
                                "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                                "Format: --trt-opt-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])
            trt_args.add_argument("--trt-max-shapes", action='append', help="The maximum shapes the optimization profile(s) will support. "
                                "Specify this option once for each profile. If not provided, inference-time input shapes are used. "
                                "Format: --trt-max-shapes <input0>,D0xD1x..xDN .. <inputN>,D0xD1x..xDN", nargs="+", default=[])

            trt_args.add_argument("--tf32", help="Enable tf32 precision in TensorRT", action="store_true", default=None)
            trt_args.add_argument("--fp16", help="Enable fp16 precision in TensorRT", action="store_true", default=None)
            trt_args.add_argument("--int8", help="Enable int8 precision in TensorRT", action="store_true", default=None)
            trt_args.add_argument("--strict-types", help="Enable strict types in TensorRT, forcing it to choose tactics based on the "
                                                        "layer precision set, even if another precision is faster.", action="store_true", default=None)
            # Workspace uses float to enable scientific notation (e.g. 1e9)
            trt_args.add_argument("--workspace", metavar="BYTES", help="Memory in bytes to allocate for the TensorRT builder's workspace", type=float, default=None)
            trt_args.add_argument("--calibration-cache", help="Path to the calibration cache", default=None)
            trt_args.add_argument("--plugins", help="Path(s) of additional plugin libraries to load", nargs="+", default=None)
        trt_args.add_argument("--explicit-precision", help="Enable explicit precision mode", action="store_true", default=None)
        trt_args.add_argument("--ext", help="Enable parsing ONNX models with externally stored weights", action="store_true", default=None)
        if self._outputs:
            trt_args.add_argument("--trt-outputs", help="Name(s) of TensorRT output(s). "
                                "Using '--trt-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
            trt_args.add_argument("--trt-exclude-outputs", help="[EXPERIMENTAL] Name(s) of TensorRT output(s) to unmark as outputs.",
                                nargs="+", default=None)
        if self._network_api:
            trt_args.add_argument("--network-api", help="[EXPERIMENTAL] Generated script will include placeholder code for defining a TensorRT Network using "
                                "the network API. Only valid if --gen/--gen-script is also enabled.", action="store_true", default=None)


    def register(self, maker):
        from polygraphy.tools.args.data_loader import DataLoaderArgs
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, DataLoaderArgs):
            self.data_loader_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        if self._config:
            assert self.data_loader_args is not None, "DataLoaderArgs is required if config is enabled!"


    def parse(self, args):
        self.plugins = tools_util.get(args, "plugins")
        self.outputs = tools_util.get_outputs(args, "trt_outputs")
        self.network_api = tools_util.get(args, "network_api")
        self.ext = tools_util.get(args, "ext")
        self.explicit_precision = tools_util.get(args, "explicit_precision")
        self.exclude_outputs = tools_util.get(args, "trt_exclude_outputs")

        self.trt_min_shapes = misc.default_value(tools_util.get(args, "trt_min_shapes"), [])
        self.trt_max_shapes = misc.default_value(tools_util.get(args, "trt_max_shapes"), [])
        self.trt_opt_shapes = misc.default_value(tools_util.get(args, "trt_opt_shapes"), [])

        workspace = tools_util.get(args, "workspace")
        self.workspace = int(workspace) if workspace is not None else workspace

        self.tf32 = tools_util.get(args, "tf32")
        self.fp16 = tools_util.get(args, "fp16")
        self.int8 = tools_util.get(args, "int8")

        self.calibration_cache = tools_util.get(args, "calibration_cache")
        self.strict_types = tools_util.get(args, "strict_types")


    # If plugins are present, wrap the provided loader/object with LoadPlugins
    def _wrap_if_plugins(self, script, obj_name):
        if self.plugins:
            script.add_import(imports=["LoadPlugins"], frm="polygraphy.backend.trt")
            loader_str = Script.invoke("LoadPlugins", obj_name, plugins=self.plugins)
            obj_name = script.add_loader(loader_str, "load_plugins")
        return obj_name


    def add_trt_network_loader(self, script):
        model_file = self.model_args.model_file
        outputs = tools_util.get_outputs_for_script(script, self.outputs)

        if self.network_api:
            CREATE_NETWORK_FUNC = Inline("create_network")

            script.add_import(imports=["CreateNetwork"], frm="polygraphy.backend.trt")
            script.add_import(imports=["func"], frm="polygraphy.common")

            script.append_prefix("# Manual TensorRT network creation")
            script.append_prefix("@func.extend(CreateNetwork())")
            script.append_prefix("def {:}(builder, network):".format(CREATE_NETWORK_FUNC))
            script.append_prefix("{tab}import tensorrt as trt\n".format(tab=constants.TAB))
            script.append_prefix("{tab}# Define your network here. Make sure to mark outputs!".format(tab=constants.TAB))
            net_inputs = self.model_args.input_shapes
            if net_inputs:
                for name, (dtype, shape) in net_inputs.items():
                    script.append_prefix("{tab}{name} = network.add_input(name='{name}', shape={shape}, dtype=trt.float32) # TODO: Set dtype".format(
                                            name=name, shape=shape, tab=constants.TAB))
            script.append_prefix("{tab}# TODO: network.mark_output(...)\n".format(tab=constants.TAB))
            return CREATE_NETWORK_FUNC

        should_use_onnx_loader = not self.ext and self.onnx_loader_args is not None

        if should_use_onnx_loader:
            script.add_import(imports=["NetworkFromOnnxBytes"], frm="polygraphy.backend.trt")
            onnx_loader = self.onnx_loader_args.add_serialized_onnx_loader(script, disable_outputs=True)
            loader_str = Script.invoke("NetworkFromOnnxBytes", self._wrap_if_plugins(script, onnx_loader), explicit_precision=self.explicit_precision)
            loader_name = script.add_loader(loader_str, "parse_network_from_onnx")
        else:
            script.add_import(imports=["NetworkFromOnnxPath"], frm="polygraphy.backend.trt")
            loader_str = Script.invoke("NetworkFromOnnxPath", self._wrap_if_plugins(script, model_file), explicit_precision=self.explicit_precision)
            loader_name = script.add_loader(loader_str, "parse_network_from_onnx")

        MODIFY_NETWORK = "ModifyNetwork"
        modify_network_str = Script.invoke(MODIFY_NETWORK, loader_name, outputs=outputs, exclude_outputs=self.exclude_outputs)
        if modify_network_str != Script.invoke(MODIFY_NETWORK, loader_name):
            script.add_import(imports=[MODIFY_NETWORK], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(modify_network_str, "modify_network")

        return loader_name


    def add_trt_config_loader(self, script, data_loader_name):
        profiles = []
        profile_args = tools_util.parse_profile_shapes(self.model_args.input_shapes, self.trt_min_shapes, self.trt_opt_shapes, self.trt_max_shapes)
        for (min_shape, opt_shape, max_shape) in profile_args:
            profile_str = "Profile()"
            for name in min_shape.keys():
                profile_str += Script.format_str(".add({:}, min={:}, opt={:}, max={:})", name, min_shape[name], opt_shape[name], max_shape[name])
            profiles.append(Inline(profile_str))
        if profiles:
            script.add_import(imports=["Profile"], frm="polygraphy.backend.trt")
            sep = Inline("\n{:}".format(constants.TAB))
            profiles = Script.format_str("[{:}{:}\n]", sep, Inline((",{:}".format(sep)).join(profiles)))
            profile_name = script.add_loader(profiles, "profiles")
        else:
            profile_name = None

        calibrator = None
        if self.int8:
            script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
            script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
            calibrator = Script.invoke("Calibrator", data_loader=Inline(data_loader_name) if data_loader_name else Inline("DataLoader()"),
                                    cache=self.calibration_cache)

        config_loader_str = Script.invoke_if_nondefault("CreateTrtConfig", max_workspace_size=self.workspace, tf32=self.tf32,
                                                        fp16=self.fp16, int8=self.int8, strict_types=self.strict_types,
                                                        profiles=profile_name, calibrator=Inline(calibrator) if calibrator else None)
        if config_loader_str is not None:
            script.add_import(imports=["CreateConfig as CreateTrtConfig"], frm="polygraphy.backend.trt")
            config_loader_name = script.add_loader(config_loader_str, "create_trt_config")
        else:
            config_loader_name = None
        return config_loader_name


    def add_trt_serialized_engine_loader(self, script):
        script.add_import(imports=["EngineFromBytes"], frm="polygraphy.backend.trt")
        script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")

        load_engine = script.add_loader(Script.invoke("BytesFromPath", self.model_args.model_file), "load_engine")
        return script.add_loader(Script.invoke("EngineFromBytes", self._wrap_if_plugins(script, load_engine)), "deserialize_engine")


    def get_trt_network_loader(self):
        script = Script()
        loader_name = self.add_trt_network_loader(script)
        exec(str(script), globals(), locals())
        return locals()[loader_name]


    def get_trt_config_loader(self, data_loader):
        script = Script()
        loader_name = self.add_trt_config_loader(script, data_loader_name="data_loader")
        exec(str(script), globals(), locals())
        return locals()[loader_name]


    def get_trt_serialized_engine_loader(self):
        script = Script()
        loader_name = self.add_trt_serialized_engine_loader(script)
        exec(str(script), globals(), locals())
        return locals()[loader_name]
