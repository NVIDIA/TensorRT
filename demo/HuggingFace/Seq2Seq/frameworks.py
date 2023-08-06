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
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig
from Seq2Seq.measurements import calculate_perplexity_helper_encoder_decoder, calculate_perplexity_helper_decoder
from Seq2Seq.export import Seq2SeqModelClass

class Seq2SeqHF(FrameworkCommand):
    def __init__(
        self,
        config_class=Seq2SeqModelTRTConfig,
        description="Runs framework results for Seq2Seq model.",
        model_classes=Seq2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            config_class, description=description, model_classes=model_classes, **kwargs
        )

    def setup_tokenizer_and_model(self):
        self.torch_model = self.load_torch_model()

        self.tokenizer = self.download_tokenizer()

        if self.cpu:
            self.torch_model = self.torch_model.cpu()
        else:
            self.torch_model = self.torch_model.cuda()

        if self.config.precision == torch.float16:
            self.torch_model = self.torch_model.half()

        self.encoder = self.config.encoder_classes["torch"].TorchModule(self.torch_model) if self.config.is_encoder_decoder else None
        self.decoder = self.config.decoder_classes["torch"].TorchModule(self.torch_model)

        torch_models = [
            NetworkModel(
                name=Seq2SeqModelTRTConfig.NETWORK_FULL_NAME, fpath=self.workspace.torch_path
            )
        ]
        return NetworkModels(torch=torch_models, onnx=None, trt=None)

    def process_framework_specific_arguments(self, cpu=False, **kwargs):
        self.use_cuda = not cpu
        self.cpu = cpu
        return kwargs

    def calculate_perplexity(self, input_str: str, reference_str: str, use_cuda: bool = True):
        if self.config.num_beams > 1:
            G_LOGGER.warning("Perplexity calculation is disabled for num_beams>1 in frameworks. Default=None")
            return None

        if self.config.is_encoder_decoder:
            perplexity = calculate_perplexity_helper_encoder_decoder(
                encoder=self.encoder,
                decoder=self.decoder,
                tokenizer=self.tokenizer,
                input_str=input_str,
                reference_str=reference_str,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                use_cuda=use_cuda,
                use_mask=self.config.use_mask,
            )
        else:
            perplexity = calculate_perplexity_helper_decoder(
                decoder=self.decoder,
                tokenizer=self.tokenizer,
                input_str=reference_str,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                use_cuda=use_cuda,
                use_mask=self.config.use_mask,
            )

        G_LOGGER.info("Perplexity={}".format(perplexity))
        return perplexity

# Entry point
RUN_CMD = Seq2SeqHF()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
