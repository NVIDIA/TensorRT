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

"""
This file defines the entry point that will be exported by our extension module.
`polygraphy run` will use this to add our custom argument groups.
"""

from polygraphy_reshape_destroyer.args import ReplaceReshapeArgs, IdentityOnlyRunnerArgs

# The entry point is expected to take no arguments and return a list of argument group instances.
#
# NOTE: Argument groups will be parsed in the order in which they are provided,
#       and after all of Polygraphy's built-in argument groups.
def export_argument_groups():
    return [
        ReplaceReshapeArgs(),
        IdentityOnlyRunnerArgs(),
    ]
