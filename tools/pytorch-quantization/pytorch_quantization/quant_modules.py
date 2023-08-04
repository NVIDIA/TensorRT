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
"""Dynamically replace the modules with quantized versions."""

from collections import namedtuple
import torch
from pytorch_quantization import nn as quant_nn

# Definition of the named tuple that is used to store mapping of the quantized modules
_quant_entry = namedtuple('quant_entry', 'orig_mod mod_name replace_mod')

# Global member of the file that contains the mapping of quantized modules
_DEFAULT_QUANT_MAP = [_quant_entry(torch.nn, "Conv1d", quant_nn.QuantConv1d),
                      _quant_entry(torch.nn, "Conv2d", quant_nn.QuantConv2d),
                      _quant_entry(torch.nn, "Conv3d", quant_nn.QuantConv3d),
                      _quant_entry(torch.nn, "ConvTranspose1d", quant_nn.QuantConvTranspose1d),
                      _quant_entry(torch.nn, "ConvTranspose2d", quant_nn.QuantConvTranspose2d),
                      _quant_entry(torch.nn, "ConvTranspose3d", quant_nn.QuantConvTranspose3d),
                      _quant_entry(torch.nn, "Linear", quant_nn.QuantLinear),
                      _quant_entry(torch.nn, "LSTM", quant_nn.QuantLSTM),
                      _quant_entry(torch.nn, "LSTMCell", quant_nn.QuantLSTMCell),
                      _quant_entry(torch.nn, "AvgPool1d", quant_nn.QuantAvgPool1d),
                      _quant_entry(torch.nn, "AvgPool2d", quant_nn.QuantAvgPool2d),
                      _quant_entry(torch.nn, "AvgPool3d", quant_nn.QuantAvgPool3d),
                      _quant_entry(torch.nn, "AdaptiveAvgPool1d", quant_nn.QuantAdaptiveAvgPool1d),
                      _quant_entry(torch.nn, "AdaptiveAvgPool2d", quant_nn.QuantAdaptiveAvgPool2d),
                      _quant_entry(torch.nn, "AdaptiveAvgPool3d", quant_nn.QuantAdaptiveAvgPool3d),]

class QuantModuleReplacementHelper():
    """To help replace torch.nn modules with quantized versions.

    This module is used to replace (by monkey patching) the torch.nn modules with their
    quantized versions as provided by either tool's internal implementation or any other
    user provided custom module.

    Attributes:
        orginal_func_map: A dict. Maintains the original torch.nn module mapping.
        quant_support_list: A list. Contains the names of modules for which a quantized
            version is provided by the tool.
        quant_map: A dict. Contains the map of the module name and its quantized versions.
        quant_switch_opt: A dict. A map to indicate which modules to be left unreplaced with
            their quantized versions. This dict is updated by a list provided from the user
            which indicates the modules to leave out in monkey patching.

    """
    def __init__(self):

        # Will hold the original modules to be replaced back
        self.orginal_func_map = set()

        # Maintains the list of supported quantized modules by the tool as default
        self.default_quant_map = _DEFAULT_QUANT_MAP

        # Will hold the final quantized modules after checking if user supplied any
        # custom quantized functions.
        self.quant_map = set()

    def prepare_state(self, float_module_list=None, custom_map=None):
        """
        Prepare the internal variables that would used in the monkey patching mechanism later.
        1. Set up the list of quantized modules that are supported by the tool for torch.nn.
        2. Set up the custom mapping for modules other than torch.nn.
        3. Use the float_module_list to switch off the monkey patching replacement for user indicated modules
        """

        # For the default quantized modules supported, generate the quant_map
        for item in self.default_quant_map:
            if float_module_list is not None and item.mod_name in float_module_list:
                # Skip this module if this is present in the float_module_list
                continue
            else:
                # append the modules into the variable that will be used in monkey patching
                self.quant_map.add(item)
                # also store the original module to be used in reverse monkey patching
                self.orginal_func_map.add(_quant_entry(item.orig_mod, item.mod_name,
                                                       getattr(item.orig_mod, item.mod_name)))

        # Add custom modules to the quant_map
        if custom_map is not None:
            for item in custom_map:
                # append the custom modules to the list that will be used in monkey patching
                # Note that we convert a tuple to a named tuple here
                self.quant_map.add(_quant_entry(item[0], item[1], item[2]))
                # also store the original module in another list which will be used to reverse monkey patching
                self.orginal_func_map.add(_quant_entry(item[0], item[1], getattr(item[0], item[1])))

    def apply_quant_modules(self):
        """
        For the modules registered in the quant_map, simply monkey patch them and also store the
        original modules so that they could be later replaced back.
        """
        for entry in self.quant_map:
            setattr(entry.orig_mod, entry.mod_name, entry.replace_mod)

    def restore_float_modules(self):
        """
        Reverse the effect of monkey patch by using the orginal_func_map to replace back the
        original modules.
        """
        for entry in self.orginal_func_map:
            setattr(entry.orig_mod, entry.mod_name, entry.replace_mod)

def initialize(float_module_list=None, custom_quant_modules=None):
    """Dynamic module replacement using monkey patching.

    Dynamically monkey patches the modules with their quantized versions. Internally, the
    state is maintained by a helper class object which helps in replacing the original
    modules back.

    Args:
        float_module_list: A list. User supplied list which indicates which modules to not monkey patch.
        custom_quant_modules: A dict. A mapping provided by user to indicate any other module apart
            from torch.nn and its corresponding quantized version.

    Returns:
        nothing.

    Typical usage example:

        # Define the deny list for torch.nn modules and custom map for modules other than torch.nn.
        float_module_list = ["Linear"]
        custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]
        ## Monkey patch the modules
        pytorch_quantization.quant_modules.initialize(float_module_list, custom_modules)
        ## Use the quantized modules
        pytorch_quantization.quant_modules.deactivate()
    """
    _quant_module_helper_object.prepare_state(float_module_list, custom_quant_modules)
    _quant_module_helper_object.apply_quant_modules()

def deactivate():
    """Dynamic module replacement which reverses the monkey patching.

    Dynamically replaces back the original modules that were monkey patched earlier
    in the initialize() function call using helper class object which maintains the state.
    """
    _quant_module_helper_object.restore_float_modules()

# Global object that maintains the state of the modules that are replaced.
_quant_module_helper_object = QuantModuleReplacementHelper()
