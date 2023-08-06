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

class BARTModelTRTConfig(Seq2SeqModelTRTConfig):

    TARGET_MODELS = [
        "facebook/bart-base",
        "facebook/bart-large",
        "facebook/bart-large-cnn",
        "facebook/mbart-large-50"
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="BART",
            **kwargs
        )

    def get_metadata_string(self, metadata) -> str:
        # Remove redundant bart name prefix
        if "mbart" in metadata.variant:
            metadata = metadata._replace(variant=metadata.variant.replace("facebook/mbart-","mbart-"))
        else:
            metadata = metadata._replace(variant=metadata.variant.replace("facebook/bart-",""))
        return super().get_metadata_string(metadata)

    def get_network_segments(self):
        """
        Returns exportable segments for BART.
        """

        self.network_segments = [
            self.NETWORK_ENCODER_SEGMENT_NAME,
            self.NETWORK_DECODER_SEGMENT_NAME,
            self.NETWORK_FULL_NAME,
        ]

        return self.network_segments
