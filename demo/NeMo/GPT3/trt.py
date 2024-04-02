#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys

import omegaconf
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from nemo_export import NeMoConverter, create_dir_if_not_exist
from GPT3.GPT3ModelConfig import GPT3ModelTRTConfig
from GPT3.trt_utils import load_trt_model
from interface import NeMoCommand, BaseModel
import onnx

sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.interface import FRAMEWORK_TENSORRT
from NNDF.logger import G_LOGGER
from NNDF.models import _log_fake_perf_metrics
from NNDF.networks import (
    NetworkModel,
    NetworkModels,
)

class GPT3NeMoTRT(NeMoCommand):
    def __init__(
        self,
        nemo_cfg,
        config_class=GPT3ModelTRTConfig,
        description="Runs TensorRT results for GPT3 model.",
        **kwargs
    ):
        super().__init__(nemo_cfg, config_class, description, model_classes=None, **kwargs)
        self.framework_name = FRAMEWORK_TENSORRT


    def setup_tokenizer_and_model(self):
        self.nemo_cfg.runtime = 'trt'
        self.model = BaseModel()
        self.model.cfg = self.nemo_cfg.model
        self.model.tokenizer = get_tokenizer(tokenizer_name='megatron-gpt-345m', vocab_file=None, merges_file=None)

        # Path to write new onnx models if need arises. Prevents overwrite of
        # user-provided onnx files in case opset_version needs to be upgraded
        # to 19 or onnx files with kv-cache needs to be written.
        onnx_workpath = os.path.join(
            self.workspace.dpath,
            "onnx",
        )
        if self.nemo_cfg.onnx_model_file:
            # Input by user, can be a read-only location.
            onnx_name = self.nemo_cfg.onnx_model_file
        else:
            onnx_name = os.path.join(
                onnx_workpath,
                f"model-{self.nemo_cfg.trainer.precision}.onnx",
            )
            self.nemo_cfg.onnx_model_file = onnx_name
            self.nemo_cfg.trt_export_options.timing_cache = self.timing_cache

            converter = NeMoConverter(self.nemo_cfg, MegatronGPTModel)
            if not os.path.isfile(onnx_name):
                # Convert NeMo model to ONNX model
                onnx_name = converter.nemo_to_onnx()

        def get_opset_version(name : str) -> int:
            """Returns opset.

            `model` here is local in scope and python's gc will collect
            it without manual memory management via `del`.
            """
            model = onnx.load(name, load_external_data=False)
            return model.opset_import[0].version

        opset_version = get_opset_version(onnx_name)
        if opset_version < 19:
            opset19_onnx_name = NeMoConverter.get_opset19_onnx_fpath(
                onnx_name, onnx_workpath
            )
            if not os.path.isfile(opset19_onnx_name):
                opset19_onnx_name = NeMoConverter.onnx_to_opset19(
                    onnx_name, onnx_workpath
                )

            if opset19_onnx_name != None:
                onnx_name = opset19_onnx_name

        # Add KV cache to ONNX model
        kv_output_policy = "kv_new"

        converter = NeMoConverter(self.nemo_cfg)

        def has_kv_cache_support(
            model_name: str, match_names=("key", "value", "kv")
        ) -> bool:
            """To detect onnx models with kv_cache exported, input node names
            contain match_names.
            """
            model = onnx.load(model_name, load_external_data=False)

            # Get network inputs.
            input_all = [node.name for node in model.graph.input]
            input_initializer =  [node.name for node in model.graph.initializer]
            net_input_names = list(set(input_all)  - set(input_initializer))

            kv_nodes = filter(
                lambda name: any(map(lambda match: match in name, match_names)),
                net_input_names,
            )
            return any(kv_nodes) and len(net_input_names) > 2

        if (not self.nemo_cfg.use_cache) and (has_kv_cache_support(onnx_name)):
            raise RuntimeError(
                "ONNX model has been exported with kv-cache enabled, but "
                "runtime configuration has kv-cache disabled. Consider "
                "enabling kv-cache support via the `use-cache` option."
            )

        if self.nemo_cfg.use_cache and (not has_kv_cache_support(onnx_name)):
            G_LOGGER.info(f"Converting {onnx_name} with KV-cache support")
            new_dir = onnx_workpath + f"_{kv_output_policy}"
            if self.nemo_cfg.onnx_export_options.use_fp8_storage:
                new_dir += f"_fp8_storage"
            onnx_output_fpath = os.path.join(new_dir, onnx_name.split("/")[-1])

            if not os.path.isfile(onnx_output_fpath):
                create_dir_if_not_exist(onnx_output_fpath)
                converter.create_onnx(onnx_name, onnx_output_fpath, kv_output_policy)
            onnx_name = onnx_output_fpath

        if self.nemo_cfg.onnx_export_options.prune:
            onnx_name = converter.prune_onnx(onnx_name)

        # Convert ONNX model to TRT engine
        self.nemo_cfg.trt_export_options.use_strongly_typed = self.use_strongly_typed
        self.nemo_cfg.trt_export_options.timing_cache = self.timing_cache
        self.nemo_cfg.trt_export_options.opt_seq_len = self.opt_seq_len

        suffixes = []
        suffixes.append("bs" + str(self.nemo_cfg.batch_size))
        if self.nemo_cfg.trt_export_options.opt_seq_len != None:
            suffixes.append("opt" + str(self.nemo_cfg.trt_export_options.opt_seq_len))
        if self.nemo_cfg.use_cache:
            suffixes.append("kv")
        if self.nemo_cfg.onnx_export_options.use_fp8_storage:
            suffixes.append("fp8_storage")
        if self.nemo_cfg.trt_export_options.sparse:
            suffixes.append("sp")
        if not self.nemo_cfg.trt_export_options.use_strongly_typed:
            suffixes.append("no_strongly_typed")
        suffix = "-".join(suffixes)
        trt_fpath = os.path.join(self.workspace.dpath, f"trt-{suffix}.plan")

        if os.path.isfile(trt_fpath):
            G_LOGGER.debug(f"TRT Engine plan exists at location {trt_fpath}.")
            _log_fake_perf_metrics()
        else:
            converter.onnx_to_trt(onnx_name, trt_fpath)

        self.nemo_cfg.trt_engine_file = trt_fpath
        self.model.trt = load_trt_model(self.nemo_cfg)
        self.tokenizer = self.model.tokenizer
        onnx_models = [
            NetworkModel(
                name=GPT3ModelTRTConfig.NETWORK_FULL_NAME, fpath=self.nemo_cfg.onnx_model_file,
            )
        ]
        return NetworkModels(torch=None, onnx=onnx_models, trt=None)

    def add_args(self):
        super().add_args()
        engine_group = self._parser.add_argument_group("trt engine")
        engine_group.add_argument(
            "--opt-seq-len",
            default=None,
            help="Set optimized input sequence length to be used in engine building",
            type=int,
        )
        engine_group.add_argument(
            "--no-timing-cache",
            default=False,
            help="Set to not use timing cache for speeding up engine building",
            action="store_true",
        )
        engine_group.add_argument(
            "--no-strongly-typed",
            default=False,
            help="Disable strongly typed mode in engine building",
            action="store_true",
        )

    def process_framework_specific_arguments(
        self,
        opt_seq_len: int = None,
        no_timing_cache: bool = False,
        no_strongly_typed: bool = False,
        **kwargs
    ):
        self.opt_seq_len = opt_seq_len
        self.use_timing_cache = not no_timing_cache
        self.use_strongly_typed = not no_strongly_typed
        self.timing_cache = self.workspace.get_timing_cache() if self.use_timing_cache else None

# Entry point
def getGPT3NeMoTRT():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config.yaml")
    nemo_cfg = omegaconf.OmegaConf.load(config_path)
    return GPT3NeMoTRT(nemo_cfg)

# Entry point
RUN_CMD = getGPT3NeMoTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
