#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Define a data structure for storing various paths used in DemoDiffusion.
"""

import dataclasses
import os
from typing import Dict


@dataclasses.dataclass
class DDPath:
    """Data class that stores various paths used in DemoDiffusion."""

    model_name_to_optimized_onnx_path: Dict[str, str] = dataclasses.field(default_factory=dict)
    model_name_to_engine_path: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Artifact paths.
    model_name_to_unoptimized_onnx_path: Dict[str, str] = dataclasses.field(default_factory=dict)
    model_name_to_weights_map_path: Dict[str, str] = dataclasses.field(default_factory=dict)
    model_name_to_refit_weights_path: Dict[str, str] = dataclasses.field(default_factory=dict)
    model_name_to_quantized_model_state_dict_path: Dict[str, str] = dataclasses.field(default_factory=dict)

    def create_directory(self) -> None:
        """Create directories for all paths, if they do not exist."""
        all_paths = [value for name_to_path in dataclasses.astuple(self) for value in name_to_path.values()]

        for path in all_paths:
            directory = os.path.dirname(path)

            # If `path` does not have a directory component, `directory` will be an empty string.
            # Only proceed if `directory` is non-empty.
            if directory:
                os.makedirs(directory, exist_ok=True)
