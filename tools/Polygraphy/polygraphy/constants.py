#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# For legacy purposes
from polygraphy.config import AUTOINSTALL_DEPS, INTERNAL_CORRECTNESS_CHECKS

DEFAULT_SHAPE_VALUE = 1
DEFAULT_SEED = 1

TAB = " " * 4  # The one true tab

MARK_ALL = "mark-all"
"""
Special value for ModifyOutputs loaders indicating that all values should be marked as outputs
"""

LEGACY_TYPE_MARKER = "polygraphy_serialized_json_type"
TYPE_MARKER = "polygraphy_class"
