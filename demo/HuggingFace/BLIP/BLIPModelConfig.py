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

# Base Class
from Vision2Seq.Vision2SeqModelConfig import Vision2SeqModelTRTConfig

class BLIPModelTRTConfig(Vision2SeqModelTRTConfig):

    TARGET_MODELS = [
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="BLIP",
            **kwargs
        )

        self.use_fp32_encoder = True
        self.is_encoder_decoder = True # Vison2Seq will have at least one vision encoder and text decoder, thus is an encoder-decoder model in this sense

    def get_metadata_string(self, metadata) -> str:
        # Remove redundant name
        metadata = metadata._replace(variant=metadata.variant.replace("Salesforce/",""))
        return super().get_metadata_string(metadata)

    def from_hf_config(self, hf_config):
        super().from_hf_config(hf_config)
        # Get num_positions for the shape of vision_model output
        # ref: https://github.com/huggingface/transformers/blob/f1732e1374a082bf8e43bd0e4aa8a2da21a32a21/src/transformers/models/blip/modeling_blip.py#L232
        self.num_positions = (self.image_size // self.vision_config.patch_size) ** 2 + 1
        # Pass num_positions into text config
        self.text_config.num_positions = self.num_positions

        self.eos_token_id = self.text_config.sep_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.max_output_length = self.text_config.max_length
        self.min_output_length = self.text_config.min_length