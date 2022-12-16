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
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class TfTrtArgs(BaseArgs):
    """
    [UNTESTED] TensorFlow-TensorRT Integration: TensorFlow-TensorRT.

    Depends on:

        - TrtConfigArgs
        - TrtLegacyRunnerArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--tftrt",
            "--use-tftrt",
            help="Enable TF-TRT integration",
            action="store_true",
            default=None,
            dest="use_tftrt",
        )
        self.group.add_argument(
            "--minimum-segment-size",
            help="Minimum length of a segment to convert to TensorRT",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--dynamic-op",
            help="Enable dynamic mode (defers engine build until runtime)",
            action="store_true",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            use_tftrt (bool): Whether to use TF-TRT.
            minimum_segment_size (int): The minimum size of segments offloaded to TRT.
            dynamic_op (bool): Whether to enable dynamic mode, which defers engine building until runtime.
        """
        self.use_tftrt = args_util.get(args, "use_tftrt")
        self.minimum_segment_size = args_util.get(args, "minimum_segment_size")
        self.dynamic_op = args_util.get(args, "dynamic_op")

    def add_to_script_impl(self, script, loader_name=None, suffix=None):
        """
        Args:
            loader_name (str): The name of the loader which should be consumed by the ``UseTfTrt`` loader.
        """
        if self.use_tftrt:
            from polygraphy.tools.args.backend.trt import TrtConfigArgs
            from polygraphy.tools.args.backend.trt_legacy import TrtLegacyRunnerArgs

            script.add_import(imports=["UseTfTrt"], frm="polygraphy.backend.tf")
            loader_str = make_invocable(
                "UseTfTrt",
                loader_name,
                max_workspace_size=self.arg_groups[TrtConfigArgs]._workspace,
                fp16=self.arg_groups[TrtConfigArgs].fp16,
                int8=self.arg_groups[TrtConfigArgs].int8,
                max_batch_size=self.arg_groups[TrtLegacyRunnerArgs].batch_size,
                is_dynamic_op=self.dynamic_op,
                minimum_segment_size=self.minimum_segment_size,
            )
            loader_name = script.add_loader(loader_str, "use_tftrt", suffix=suffix)
        return loader_name


@mod.export()
class TfLoadArgs(BaseArgs):
    """
    TensorFlow Model Loading: loading TensorFlow models.

    Depends on:

        - ModelArgs
        - TfTrtArgs: if allow_tftrt == True
        - TrtSaveEngineArgs: if allow_tftrt == True
    """

    def __init__(self, allow_artifacts: bool = None, allow_custom_outputs: bool = None, allow_tftrt: bool = None):
        """
        Args:
            allow_artifacts (bool):
                    Whether to allow saving artifacts to the disk, like frozen models or TensorBoard visualizations.
                    Defaults to True.
            allow_custom_outputs (bool):
                    Whether to allow marking custom output tensors.
                    Defaults to True.
            allow_tftrt (bool):
                    Whether to allow applying TF-TRT.
                    Defaults to False.

        """
        super().__init__()
        self._allow_artifacts = util.default(allow_artifacts, True)
        self._allow_custom_outputs = util.default(allow_custom_outputs, True)
        self._allow_tftrt = util.default(allow_tftrt, False)

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--ckpt",
            help="[EXPERIMENTAL] Name of the checkpoint to load. Required if the `checkpoint` file is missing. Should not include file extension "
            "(e.g. to load `model.meta` use `--ckpt=model`)",
            default=None,
        )
        if self._allow_custom_outputs:
            self.group.add_argument(
                "--tf-outputs",
                help="Name(s) of TensorFlow output(s). "
                "Using '--tf-outputs mark all' indicates that all tensors should be used as outputs",
                nargs="+",
                default=None,
            )

        if self._allow_artifacts:
            self.group.add_argument(
                "--save-pb",
                help="Path to save the TensorFlow frozen graphdef",
                default=None,
                dest="save_frozen_graph_path",
            )
            self.group.add_argument(
                "--save-tensorboard",
                help="[EXPERIMENTAL] Path to save a TensorBoard visualization",
                default=None,
                dest="save_tensorboard_path",
            )

        self.group.add_argument(
            "--freeze-graph", help="[EXPERIMENTAL] Attempt to freeze the graph", action="store_true", default=None
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            ckpt (str): Name of the checkpoint.
            outputs (List[str]): Names of output tensors.
            save_frozen_graph_path (str): The path at which the frozen graph will be saved.
            save_tensorboard_path (str): The path at which the TensorBoard visualization will be saved.
            freeze_graph (bool): Whether to attempt to freeze the graph.
        """
        self.ckpt = args_util.get(args, "ckpt")
        self.outputs = args_util.get_outputs(args, "tf_outputs")
        self.save_frozen_graph_path = args_util.get(args, "save_frozen_graph_path")
        self.save_tensorboard_path = args_util.get(args, "save_tensorboard_path")
        self.freeze_graph = args_util.get(args, "freeze_graph")

    def add_to_script_impl(self, script, disable_custom_outputs=None):
        """
        Args:
            disable_custom_outputs (bool):
                    Whether to disallow modifying outputs according to the `outputs` attribute.
                    Defaults to False.
        """

        model_file = self.arg_groups[ModelArgs].path
        model_type = self.arg_groups[ModelArgs].model_type

        if model_type == "ckpt":
            G_LOGGER.verbose(
                f"Loading a TensorFlow checkpoint from {model_file}. Please ensure you are not using the --use-subprocess flag",
                mode=LogMode.ONCE,
            )
            script.add_import(imports=["GraphFromCkpt"], frm="polygraphy.backend.tf")
            loader_id = "load_ckpt"
            loader_str = make_invocable("GraphFromCkpt", model_file, self.ckpt)
        elif model_type == "keras":
            script.add_import(imports=["GraphFromKeras"], frm="polygraphy.backend.tf")
            loader_id = "load_keras"
            loader_str = make_invocable("GraphFromKeras", model_file)
        elif model_type == "frozen":
            script.add_import(imports=["GraphFromFrozen"], frm="polygraphy.backend.tf")
            G_LOGGER.verbose(
                "Attempting to load as a frozen graph. If this is not correct, please specify --model-type",
                mode=LogMode.ONCE,
            )
            loader_id = "load_frozen"
            loader_str = make_invocable("GraphFromFrozen", model_file)
        else:
            G_LOGGER.critical(f"Model type: {model_type} cannot be imported with TensorFlow.")

        loader_name = script.add_loader(loader_str, loader_id)

        if self.freeze_graph:
            script.add_import(imports=["OptimizeGraph"], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(make_invocable("OptimizeGraph", loader_name), "optimize_graph")

        engine_dir = None
        if self._allow_tftrt:
            from polygraphy.tools.args.backend.trt import TrtSaveEngineArgs

            loader_name = self.arg_groups[TfTrtArgs].add_to_script(script, loader_name)
            engine_dir = self.arg_groups[TrtSaveEngineArgs].path

        MODIFY_TF = "ModifyGraphOutputs"
        outputs = None if disable_custom_outputs else args_util.get_outputs_for_script(script, self.outputs)
        modify_tf_str = make_invocable(MODIFY_TF, loader_name, outputs=outputs)
        if modify_tf_str != make_invocable(MODIFY_TF, loader_name):
            script.add_import(imports=[MODIFY_TF], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(modify_tf_str, "modify_tf")

        WRITE_TF = "SaveGraph"
        write_tf_str = make_invocable(
            WRITE_TF,
            loader_name,
            path=self.save_frozen_graph_path,
            tensorboard_dir=self.save_tensorboard_path,
            engine_dir=engine_dir,
        )
        if write_tf_str != make_invocable(WRITE_TF, loader_name):
            script.add_import(imports=[WRITE_TF], frm="polygraphy.backend.tf")
            loader_name = script.add_loader(write_tf_str, "save_tf")

        return loader_name

    def load_graph(self):
        """
        Loads a TensorFlow graph according to arguments provided on the command-line.

        Returns:
            tf.Graph
        """
        loader = args_util.run_script(self.add_to_script)
        return loader()
