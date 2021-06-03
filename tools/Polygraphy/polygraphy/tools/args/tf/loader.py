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
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Script


class TfLoaderArgs(BaseArgs):
    def __init__(self, tftrt=True, artifacts=True, outputs=True):
        self._enable_tftrt = tftrt
        self._enable_artifacts = artifacts
        self._enable_outputs = outputs


    def add_to_parser(self, parser):
        tf_args = parser.add_argument_group("TensorFlow", "Options for TensorFlow")
        tf_args.add_argument("--ckpt", help="[EXPERIMENTAL] Name of the checkpoint to load. Required if the `checkpoint` file is missing. Should not include file extension "
                            "(e.g. to load `model.meta` use `--ckpt=model`)", default=None)
        if self._enable_outputs:
            tf_args.add_argument("--tf-outputs", help="Name(s) of TensorFlow output(s). "
                                 "Using '--tf-outputs mark all' indicates that all tensors should be used as outputs", nargs="+", default=None)
        if self._enable_artifacts:
            tf_args.add_argument("--save-pb", help="Path to save the TensorFlow frozen graphdef", default=None)
            tf_args.add_argument("--save-tensorboard", help="[EXPERIMENTAL] Path to save a TensorBoard visualization", default=None)
        tf_args.add_argument("--freeze-graph", help="[EXPERIMENTAL] Attempt to freeze the graph", action="store_true", default=None)
        if self._enable_tftrt:
            tftrt_args = parser.add_argument_group("TensorFlow-TensorRT", "[UNTESTED] Options for TensorFlow-TensorRT Integration")
            tftrt_args.add_argument("--tftrt", help="[UNTESTED] Enable TF-TRT integration", action="store_true", default=None)
            tftrt_args.add_argument("--minimum-segment-size", help="Minimum length of a segment to convert to TensorRT", type=int, default=None)
            tftrt_args.add_argument("--dynamic-op", help="Enable dynamic mode (defers engine build until runtime)", action="store_true", default=None)


    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.trt.loader import TrtLoaderArgs
        from polygraphy.tools.args.trt.runner import TrtRunnerArgs
        from polygraphy.tools.args.trt_legacy import TrtLegacyArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker

        if isinstance(maker, TrtLoaderArgs):
            self.trt_loader_args = maker

        if isinstance(maker, TrtLegacyArgs):
            self.trt_legacy_args = maker

        if isinstance(maker, TrtRunnerArgs):
            self.trt_runner_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        if self._enable_tftrt:
            assert self.trt_loader_args is not None, "TrtLoaderArgs is required when tftrt is enabled!"


    def parse(self, args):
        self.ckpt = tools_util.get(args, "ckpt")
        self.outputs = tools_util.get_outputs(args, "tf_outputs")
        self.save_pb = tools_util.get(args, "save_pb")
        self.save_tensorboard = tools_util.get(args, "save_tensorboard")
        self.freeze_graph = tools_util.get(args, "freeze_graph")
        self.tftrt = tools_util.get(args, "tftrt")
        self.minimum_segment_size = tools_util.get(args, "minimum_segment_size")
        self.dynamic_op = tools_util.get(args, "dynamic_op")


    def add_to_script(self, script, disable_outputs=None, suffix=None):
        if disable_outputs:
            outputs = None
        else:
            outputs = tools_util.get_outputs_for_script(script, self.outputs)

        model_file = self.model_args.model_file
        model_type = self.model_args.model_type

        if model_type == "ckpt":
            G_LOGGER.verbose("Loading a TensorFlow checkpoint. Please ensure you are not using the --use-subprocess flag".format(model_file), mode=LogMode.ONCE)
            script.add_import(imports=["GraphFromCkpt"], frm="polygraphy.backend.tf")
            loader_id = "load_ckpt"
            loader_str = Script.invoke("GraphFromCkpt", model_file, self.ckpt)
        elif model_type == "keras":
            script.add_import(imports=["GraphFromKeras"], frm="polygraphy.backend.tf")
            loader_id = "load_keras"
            loader_str = Script.invoke("GraphFromKeras", model_file)
        else:
            script.add_import(imports=["GraphFromFrozen"], frm="polygraphy.backend.tf")
            G_LOGGER.verbose("Attempting to load as a frozen graph. If this is not correct, please specify --model-type", mode=LogMode.ONCE)
            loader_id = "load_frozen"
            loader_str = Script.invoke("GraphFromFrozen", model_file)

        loader_name = script.add_loader(loader_str, loader_id, suffix=suffix)

        if self.freeze_graph:
            script.add_import(imports=["OptimizeGraph"], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(Script.invoke("OptimizeGraph", loader_name), "optimize_graph", suffix=suffix)
        if self.tftrt:
            script.add_import(imports=["UseTfTrt"], frm="polygraphy.backend.tf")
            loader_str = Script.invoke("UseTfTrt", loader_name, max_workspace_size=self.trt_loader_args.workspace, fp16=self.trt_loader_args.fp16, int8=self.trt_loader_args.int8,
                                       max_batch_size=self.trt_legacy_args.batch_size, is_dynamic_op=self.dynamic_op, minimum_segment_size=self.minimum_segment_size)
            loader_name = script.add_loader(loader_str, "use_tftrt", suffix=suffix)

        MODIFY_TF = "ModifyGraph"
        modify_tf_str = Script.invoke(MODIFY_TF, loader_name, outputs=outputs)
        if modify_tf_str != Script.invoke(MODIFY_TF, loader_name):
            script.add_import(imports=[MODIFY_TF], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(modify_tf_str, "modify_tf")

        engine_dir = None
        if self.tftrt:
            engine_dir = self.trt_runner_args.save_engine

        WRITE_TF = "SaveGraph"
        write_tf_str = Script.invoke(WRITE_TF, loader_name, path=self.save_pb, tensorboard_dir=self.save_tensorboard, engine_dir=engine_dir)
        if write_tf_str != Script.invoke(WRITE_TF, loader_name):
            script.add_import(imports=[WRITE_TF], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(write_tf_str, "save_tf")

        return loader_name


    def get_tf_loader(self):
        script = Script()
        loader_name = self.add_to_script(script)
        exec(str(script), globals(), locals())
        return locals()[loader_name]
