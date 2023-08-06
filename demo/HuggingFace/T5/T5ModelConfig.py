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
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig

class T5ModelTRTConfig(Seq2SeqModelTRTConfig):

    TARGET_MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="T5",
            **kwargs
        )

        self.use_fp32_encoder = True

    def get_metadata_string(self, metadata) -> str:
        # Truncate flan-t5 variant
        if "google/" in metadata.variant:
            metadata = metadata._replace(variant=metadata.variant.lstrip("google/"))
        else:
            # Remove redundant t5 name
            metadata = metadata._replace(variant=metadata.variant.lstrip("t5-"))

        return super().get_metadata_string(metadata)

    def get_network_segments(self):
        """
        Returns exportable segments for T5. T5 is encoder/decoder
        """

        self.network_segments = [
            self.NETWORK_ENCODER_SEGMENT_NAME,
            self.NETWORK_DECODER_SEGMENT_NAME,
            self.NETWORK_FULL_NAME,
        ]

        return self.network_segments
