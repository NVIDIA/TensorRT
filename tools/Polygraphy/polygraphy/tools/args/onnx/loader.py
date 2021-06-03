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
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Script


class OnnxLoaderArgs(BaseArgs):
    def __init__(self, write=True, outputs=True, shape_inference_default=None):
        self.tf2onnx_loader_args = None
        self._write = write
        self._outputs = outputs
        self._shape_inference_default = shape_inference_default


    def add_to_parser(self, parser):
        onnx_args = parser.add_argument_group("ONNX Options", "Options for ONNX")
        if self._write:
            onnx_args.add_argument("--save-onnx", help="Path to save the ONNX model", default=None)

        if self._shape_inference_default:
            onnx_args.add_argument("--no-shape-inference", help="Disable ONNX shape inference when loading the model", action="store_true", default=None)
        else:
            onnx_args.add_argument("--shape-inference", help="Enable ONNX shape inference when loading the model", action="store_true", default=None)

        if self._outputs:
            onnx_args.add_argument("--onnx-outputs", help="Name(s) of ONNX output(s). "
                                "Using '--onnx-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
            onnx_args.add_argument("--onnx-exclude-outputs", help="[EXPERIMENTAL] Name(s) of ONNX output(s) to unmark as outputs.", nargs="+", default=None)


    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.tf2onnx.loader import Tf2OnnxLoaderArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker

        if isinstance(maker, Tf2OnnxLoaderArgs):
            self.tf2onnx_loader_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"


    def parse(self, args):
        self.save_onnx = tools_util.get(args, "save_onnx")
        if hasattr(args, "no_shape_inference"):
            self.do_shape_inference = None if tools_util.get(args, "no_shape_inference") else True
        else:
            self.do_shape_inference = tools_util.get(args, "shape_inference")
        self.outputs = tools_util.get_outputs(args, "onnx_outputs")
        self.exclude_outputs = tools_util.get(args, "onnx_exclude_outputs")


    def _get_modify_onnx_str(self, script, loader_name, disable_outputs=None):
        if disable_outputs:
            outputs = None
            exclude_outputs = None
        else:
            outputs = tools_util.get_outputs_for_script(script, self.outputs)
            exclude_outputs = self.exclude_outputs

        MODIFY_ONNX = "ModifyOnnx"
        modify_onnx_str = Script.invoke(MODIFY_ONNX, loader_name, do_shape_inference=self.do_shape_inference,
                                        outputs=outputs, exclude_outputs=exclude_outputs)
        if modify_onnx_str != Script.invoke(MODIFY_ONNX, loader_name):
            script.add_import(imports=[MODIFY_ONNX], frm="polygraphy.backend.onnx")
            return modify_onnx_str
        return None


    def add_onnx_loader(self, script, disable_outputs=None, suffix=None):
        if self.model_args.model_type == "onnx":
            script.add_import(imports=["OnnxFromPath"], frm="polygraphy.backend.onnx")
            loader_str = Script.invoke("OnnxFromPath", self.model_args.model_file)
            loader_name = script.add_loader(loader_str, "load_onnx", suffix=suffix)
        else:
            if self.tf2onnx_loader_args is None:
                G_LOGGER.critical("Could not load: {:}. Is it an ONNX model?".format(self.model_args.model_file))
            loader_name = self.tf2onnx_loader_args.add_to_script(script)

        modify_onnx_str = self._get_modify_onnx_str(script, loader_name, disable_outputs=disable_outputs)
        if modify_onnx_str is not None:
            loader_name = script.add_loader(modify_onnx_str, "modify_onnx")

        SAVE_ONNX = "SaveOnnx"
        save_onnx_str = Script.invoke(SAVE_ONNX, loader_name, path=self.save_onnx)
        if save_onnx_str != Script.invoke(SAVE_ONNX, loader_name):
            script.add_import(imports=[SAVE_ONNX], frm="polygraphy.backend.onnx")
            loader_name = script.add_loader(save_onnx_str, "save_onnx")

        return loader_name


    def add_serialized_onnx_loader(self, script, disable_outputs=None):
        model_file = self.model_args.model_file

        needs_modify = self._get_modify_onnx_str(script, "check_needs_modify", disable_outputs) is not None
        should_import_raw = self.model_args.model_type == "onnx" and not needs_modify

        if should_import_raw:
            script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")
            onnx_loader = script.add_loader(Script.invoke("BytesFromPath", model_file), "load_serialized_onnx")
        else:
            script.add_import(imports=["BytesFromOnnx"], frm="polygraphy.backend.onnx")
            onnx_loader = self.add_onnx_loader(script, disable_outputs=disable_outputs)
            onnx_loader = script.add_loader(Script.invoke("BytesFromOnnx", onnx_loader), "serialize_onnx")
        return onnx_loader


    def get_onnx_loader(self):
        script = Script()
        loader_name = self.add_onnx_loader(script)
        exec(str(script), globals(), locals())
        return locals()[loader_name]
