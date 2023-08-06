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

import onnxruntime as ort
import onnx
import omegaconf
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from interface import NeMoCommand, BaseModel
from nemo_export import NeMoConverter
from GPT3.GPT3ModelConfig import GPT3ModelTRTConfig

sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.interface import FRAMEWORK_ONNXRT
from NNDF.logger import G_LOGGER
from NNDF.networks import (
    NetworkModel,
    NetworkModels,
)

class GPT3NeMoOnnxRT(NeMoCommand):
    def __init__(
        self,
        nemo_cfg,
        config_class=GPT3ModelTRTConfig,
        description="Runs ONNX Runtime results for GPT3 model.",
        **kwargs
    ):
        super().__init__(nemo_cfg, config_class, description, model_classes=None, **kwargs)
        self.framework_name = FRAMEWORK_ONNXRT


    def load_onnx_model(self):
        G_LOGGER.info(f'Loading ONNX model from {self.nemo_cfg.onnx_model_file}')

        def get_opset_version(name : str) -> int:
            """Returns opset.

            `model` here is local in scope and python's gc will collect
            it without manual memory management via `del`.
            """
            model = onnx.load(name, load_external_data=False)
            return model.opset_import[0].version

        assert get_opset_version(self.nemo_cfg.onnx_model_file) == 17
        return ort.InferenceSession(self.nemo_cfg.onnx_model_file)


    def setup_tokenizer_and_model(self):
        self.nemo_cfg.runtime = 'onnx'
        self.model = BaseModel()
        self.model.cfg = self.nemo_cfg.model
        self.model.tokenizer = get_tokenizer(tokenizer_name='megatron-gpt-345m', vocab_file=None, merges_file=None)

        if not self.nemo_cfg.onnx_model_file:
            self.nemo_cfg.onnx_model_file = os.path.join(
                self.workspace.dpath,
                f"onnx/model-{self.nemo_cfg.trainer.precision}.onnx",
            )

        converter = NeMoConverter(self.nemo_cfg, MegatronGPTModel)
        if not os.path.isfile(self.nemo_cfg.onnx_model_file):
            # Convert NeMo model to ONNX model
            onnx_name = converter.nemo_to_onnx()
            self.nemo_cfg.onnx_model_file = onnx_name

        # The ONNX model is in opset17 by default.
        self.model.onnxrt = self.load_onnx_model()
        self.tokenizer = self.model.tokenizer
        onnx_models = [
            NetworkModel(
                name=GPT3ModelTRTConfig.NETWORK_FULL_NAME, fpath=self.nemo_cfg.onnx_model_file,
            )
        ]
        return NetworkModels(torch=None, onnx=onnx_models, trt=None)

# Entry point
def getGPT3NeMoOnnxRT():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config.yaml")
    nemo_cfg = omegaconf.OmegaConf.load(config_path)
    return GPT3NeMoOnnxRT(nemo_cfg)

# Entry point
RUN_CMD = getGPT3NeMoOnnxRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
