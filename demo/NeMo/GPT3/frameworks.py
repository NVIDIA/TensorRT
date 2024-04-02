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

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from GPT3.nemo_utils import load_nemo_model
from GPT3.GPT3ModelConfig import GPT3ModelTRTConfig
from interface import NeMoCommand

sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.interface import FRAMEWORK_NATIVE
from NNDF.networks import (
    NetworkModel,
    NetworkModels,
)

class GPT3NeMoTorch(NeMoCommand):
    def __init__(
        self,
        nemo_cfg,
        config_class=GPT3ModelTRTConfig,
        description="Runs framework results for GPT3 model with NeMo.",
        **kwargs
    ):
        super().__init__(nemo_cfg, config_class, description, model_classes=None, **kwargs)
        self.framework_name = FRAMEWORK_NATIVE

    def setup_tokenizer_and_model(self):
        self.nemo_cfg.runtime = 'nemo'
        self.model = load_nemo_model(self.nemo_cfg)
        self.tokenizer = self.model.tokenizer

        torch_models = [
            NetworkModel(
                name=GPT3ModelTRTConfig.NETWORK_FULL_NAME, fpath=self.workspace.torch_path
            )
        ]
        return NetworkModels(torch=torch_models, onnx=None, trt=None)

    def process_framework_specific_arguments(self, onnx_model: str = None, **kwargs):
        if onnx_model:
            raise RuntimeError(
                "native framework does not support loading an ONNX file via `onnx-model` yet. Please specify the NeMo model using `nemo-model` instead."
            )


# Entry point
def getGPT3NeMoTorch():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config.yaml")
    nemo_cfg = omegaconf.OmegaConf.load(config_path)
    return GPT3NeMoTorch(nemo_cfg)

# Entry point
RUN_CMD = getGPT3NeMoTorch()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
