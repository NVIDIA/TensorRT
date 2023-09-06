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

# torch
import torch

from transformers import AutoProcessor

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from NNDF.interface import FrameworkCommand
from NNDF.networks import (
    NetworkModels,
    NetworkModel,
)
from NNDF.logger import G_LOGGER
from Vision2Seq.Vision2SeqModelConfig import Vision2SeqModelTRTConfig
from Vision2Seq.export import Vision2SeqModelClass

class Vision2SeqHF(FrameworkCommand):
    def __init__(
        self,
        config_class=Vision2SeqModelTRTConfig,
        description="Runs framework results for Vision2Seq model.",
        model_classes=Vision2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            config_class, description=description, model_classes=model_classes, **kwargs
        )

    def setup_tokenizer_and_model(self):
        self.torch_model = self.load_torch_model()

        # self.tokenizer = self.download_tokenizer()
        processor = AutoProcessor.from_pretrained(self.config.metadata.variant)
        self.tokenizer = processor.tokenizer

        if self.cpu:
            self.torch_model = self.torch_model.cpu()
        else:
            self.torch_model = self.torch_model.cuda()

        if self.config.precision == torch.float16:
            self.torch_model = self.torch_model.half()

        self.encoder = self.config.encoder_classes["torch"].TorchModule(self.torch_model)
        self.decoder = self.config.decoder_classes["torch"].TorchModule(self.torch_model)

        torch_models = [
            NetworkModel(
                name=Vision2SeqModelTRTConfig.NETWORK_FULL_NAME, fpath=self.workspace.torch_path
            )
        ]
        return NetworkModels(torch=torch_models, onnx=None, trt=None)

    def process_framework_specific_arguments(self, cpu=False, **kwargs):
        self.use_cuda = not cpu
        self.cpu = cpu
        return kwargs

# Entry point
RUN_CMD = Vision2SeqHF()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
