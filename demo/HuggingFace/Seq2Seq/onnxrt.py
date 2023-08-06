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

"""
Executes ONNX Runtime framework code. See README.md for more information.
"""

import os
import sys

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# huggingface
from transformers import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

# torch
import torch

# TRT-HuggingFace
from NNDF.interface import OnnxRTCommand
from NNDF.networks import (
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
)
from NNDF.networks import Precision, NNConfig
from NNDF.tensorrt_utils import PolygraphyOnnxRunner
from NNDF.general_utils import confirm_folder_delete
from NNDF.logger import G_LOGGER
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig
from Seq2Seq.measurements import calculate_perplexity_helper_decoder, calculate_perplexity_helper_encoder_decoder
from Seq2Seq.export import Seq2SeqModelClass

class OnnxEncoder(PolygraphyOnnxRunner):
    """OnnxRT implemented network interface that is mainly to check correctness."""
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Unoptimized unconditional transfer to numpy for interfacing with polygraphy
        input_ids = input_ids.cpu().numpy().astype("int64")
        input_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            attention_mask = attention_mask.cpu().numpy().astype("int64")
            input_dict["attention_mask"] = attention_mask

        return BaseModelOutput(last_hidden_state = torch.from_numpy(self.runner.infer(input_dict)["encoder_hidden_states"]))

class OnnxDecoder(PolygraphyOnnxRunner, GenerationMixin):

    def __init__(self, engine_fpath: str, network_metadata: NetworkMetadata, config: NNConfig):
        super().__init__(engine_fpath, network_metadata)
        self.main_input_name = "input_ids"
        self.config = config
        self.generation_config = config.generation_config
        self.device = torch.device("cpu")

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_dict = {
            "input_ids": input_ids
        }
        if self.config.use_mask:
            if self.config.is_encoder_decoder:
                input_dict["attention_mask"] = torch.ones_like(input_ids)
            else:
                input_dict["attention_mask"] = kwargs["attention_mask"]

        if kwargs.get("encoder_outputs") is not None:
            input_dict["encoder_outputs"] = kwargs["encoder_outputs"]

        return input_dict

    def forward(self, input_ids, encoder_outputs = None, attention_mask = None, **kwargs):
        # Unoptimized unconditional transfer to numpy for interfacing with polygraphy
        input_ids = input_ids.cpu().numpy().astype("int64")
        data_type = "float32"
        input_dict = {"input_ids": input_ids}
        if self.config.use_mask:
            attention_mask = attention_mask.cpu().numpy().astype("int64")
            input_dict["attention_mask"] = attention_mask

        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs.last_hidden_state.cpu().numpy().astype(data_type)
            input_dict["encoder_hidden_states"] = encoder_hidden_states

        logits = self.runner.infer(input_dict)["logits"]

        return Seq2SeqLMOutput(logits=torch.from_numpy(logits))


class Seq2SeqOnnxRT(OnnxRTCommand):
    def __init__(
        self,
        config_class=Seq2SeqModelTRTConfig,
        description="Runs OnnxRT results for Seq2Seq model.",
        model_classes=Seq2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            config_class, description=description, model_classes=model_classes, **kwargs
        )

    def __call__(self):
        return super().__call__()

    def setup_tokenizer_and_model(self):

        if self.config.use_cache:
            G_LOGGER.warning("OnnxRT does not support use_cache. Will default to False.")
            self.config.use_cache = False
            self.metadata = self.metadata._replace(use_cache=False)
        if self.config.precision == torch.float16:
            G_LOGGER.warning("OnnxRT does not support fp16 ONNX yet. Ignored.")
            self.config.precision = torch.float32
            self.metadata = self.metadata._replace(precision=Precision(fp16=False))

        self.use_generator = False

        self.tokenizer = self.download_tokenizer()

        self.load_onnx_model()

        self.decoder = OnnxDecoder(
            engine_fpath = self.onnx_decoder.fpath,
            network_metadata = self.metadata,
            config = self.config,
        )

        if self.config.is_encoder_decoder:
            self.encoder = OnnxEncoder(
                self.onnx_encoder.fpath, self.metadata
            )

        onnx_models = [
            NetworkModel(
                name=self.config.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.onnx_decoder.fpath,
            )
        ]

        if self.config.is_encoder_decoder:
            onnx_models.append(
                NetworkModel(
                    name=self.config.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.onnx_encoder.fpath,
                )
            )

        return NetworkModels(torch=None, onnx=onnx_models, trt=None)

    def calculate_perplexity(self, input_str: str, reference_str: str, use_cuda: bool = True):
        if self.config.num_beams > 1:
            G_LOGGER.warning("Perplexity calculation is disabled for num_beams>1 in OnnxRT. Default=None")
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

    def cleanup(self,
    ) -> None:
        # Deactivates context
        if self.encoder:
            self.encoder.release()
        if self.decoder:
            self.decoder.release()

        if not self.keep_onnx_model:
            if self.onnx_decoder:
                self.onnx_decoder.cleanup()
            if self.config.is_encoder_decoder and self.onnx_encoder:
                self.onnx_encoder.cleanup()
            if self.use_generator and self.onnx_cross_attn_cache_generator:
                self.onnx_cross_attn_cache_generator.cleanup()

        if not self.keep_torch_model and self.workspace.torch_path is not None:

            confirm_folder_delete(
                self.workspace.torch_path,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not self.keep_onnx_model:
            self.workspace.cleanup()

# Entry point
RUN_CMD = Seq2SeqOnnxRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))