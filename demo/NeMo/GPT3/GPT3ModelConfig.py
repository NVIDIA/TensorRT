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
import sys
sys.path.append('../../HuggingFace') # Include HuggingFace directory
from NNDF.networks import NNConfig, NetworkMetadata

class GPT3ModelTRTConfig(NNConfig):

    NETWORK_FULL_NAME = "full"
    TARGET_MODELS = [
        "gpt-126m",
        "gpt-1.3b",
        "gpt-5b",
    ]

    def __init__(
        self,
        metadata,
        **kwargs
    ):
        super().__init__(
            network_name="GPT3",
            **kwargs
        )
        self.nemo_config = None
        self.use_mask = False
        self.metadata = metadata
        self.variant = metadata.variant

    def from_nemo_config(self, nemo_config):
        self.nemo_config = nemo_config

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        """
        Serializes a Metadata object into string.
        String will be checked if friendly to filenames across Windows and Linux operating systems.
        This function is a modified version from HuggingFace/NNDF/networks.py.

        returns:
            string: <network>-<variant-name>[-<precision>]*-<others>
        """

        enabled_precisions = self.nemo_config.trt_export_options
        precision_str = "-".join(
            [
                k for k, v in {
                    "fp8": enabled_precisions.use_fp8,
                    "fp16": enabled_precisions.use_fp16,
                    "bf16": enabled_precisions.use_bf16,
                }.items() if v
            ]
        )

        result = [self.network_name, metadata.variant]
        if precision_str:
            result.append(precision_str)

        # Append max sequence length
        result.append("ms" + str(self.nemo_config.model.max_seq_len))

        if metadata.use_cache:
            result.append("kv_cache")

        final_str = "-".join(result)
        assert self._is_valid_filename(
            final_str
        ), "Metadata for current network {} is not filename friendly: {}.".format(
            self.network_name, final_str
        )

        return final_str
