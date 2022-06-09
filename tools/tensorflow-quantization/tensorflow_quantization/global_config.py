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
This module holds the quantization config class object that is accessed globally by library modules.
DIRECT USE OF THIS MODULE BY USER IS PROHIBITED.
"""

# List that holds quantization config class object, Length is always one!
# Object is added automatically on class creation
G_CONFIG_OBJECT = []


def add_config_object(config_object: "BaseConfig") -> None:
    """
    Add instance of quantize config class to the global list.
    Args:
        config_object : Instance of one of four quantize config class
    """
    assert (
            len(G_CONFIG_OBJECT) == 0
    ), "Looks like previous quatize object is alive. Did you call clear() on the object?"
    G_CONFIG_OBJECT.append(config_object)


def remove_config_object() -> None:
    """
    Remove instance of quantize config class from the global list.
    """
    if G_CONFIG_OBJECT:
        G_CONFIG_OBJECT.clear()


def get_config_object() -> "BaseConfig":
    """
    Return quantize config class object
    """
    assert (
            len(G_CONFIG_OBJECT) == 1
    ), "Have you created quantize config object before calling `quantize_model`?"
    if G_CONFIG_OBJECT:
        return G_CONFIG_OBJECT[0]


def is_config_object_created() -> bool:
    """
    Sanity check function for whether quantize config class object is created.
    """
    return len(G_CONFIG_OBJECT) == 1
