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


"""Helper functions for quant optimizer/trainer"""

import re

from absl import logging

def match_parameters(model, patterns):
    """Returns an generator over module parameters if name matches key

    It is useful to group parameters, and apply different functions to different group. This function provides an easy
    way to group them.

    Args:
        model: A Module
        patterns: A list of strings that will be used to match parameter names. If parameter name contains any pattern,
            it will be yield

    Yields:
        param: Module parameters
    """
    for name, param in model.named_parameters():
        for pattern in patterns:
            if re.search(pattern, name):
                yield param

def group_parameters(model, patterns_list, lrs=None, momentums=None, weight_decays=None):
    """Group parameters for using per-parameters option in optimizer

    Returns a list of dict that matches Pytorch optimizer fashion, see
    https://pytorch.org/docs/stable/optim.html#per-parameter-options for more details.

    Example:
        >>> [
        >>>    {'params': model.base.parameters()},
        >>>    {'params': model.classifier.parameters(), 'lr': 1e-3}
        >>> ]
    Parameters will be grouped w.r.t first level of the keys_list. e.g. `keys_list=[['conv1', 'conv2'], ['conv3']]` will
    return 2 groups, one with `conv1` and `conv2` in name, and the other with `conv3` in name.

    If lr, momentum or weight_decay are supplied, they will be added to the group as well.


    Args:
        model: A module
        patterns_list: A list of list of strings. WARNING: patters must be EXCLUSIVE, the function doesn't
            perform exclusive check.
        lrs: A list of float with same length as keys_list or None.
        momentums: A list of float with same length as keys_list or None.
        weight_decays: A list of float with same length as keys_list or None.

    Returns:
        param_group: A list of dict

    """
    param_groups = []
    for pattern in patterns_list:
        if not isinstance(pattern, list):
            raise TypeError("patterns_list must be list of list of patterns")
        param_groups.append({'params': match_parameters(model, pattern)})

    if lrs is not None:
        if len(lrs) != len(patterns_list):
            raise TypeError("len(lrs) must match len(patterns_list)")
        for i, lr in enumerate(lrs):
            param_groups[i]['lr'] = lr

    if momentums is not None:
        if len(momentums) != len(patterns_list):
            raise TypeError("len(momentums) must match len(patterns_list)")
        for i, momentum in enumerate(momentums):
            param_groups[i]['momentum'] = momentum

    if weight_decays is not None:
        if len(weight_decays) != len(patterns_list):
            raise TypeError("len(weight_decays) must match len(patterns_list)")
        for i, weight_decay in enumerate(weight_decays):
            param_groups[i]['weight_decay'] = weight_decay

    return param_groups

def freeze_parameters(model, patterns):
    """Set requires_grad to False if patterns match name

    Args:
        model: A Module
        patterns: A list of strings that will be used to match parameter names. If parameter name contains any pattern,
            it will be frozen.
    """
    for name, param in model.named_parameters():
        for pattern in patterns:
            if re.search(pattern, name):
                logging.warning("Freeze %s.", name)
                param.requires_grad = False

def quant_weight_inplace(model):
    """Make quantization inplace

    Search for quantized modules including QuantConvNd and QuantLinear, make weight quantization in place using
    weight_quantizer.

    Most publications of quantization aware training uses STE by default, which is really an approximation of
    derivative of the nondifferentiable quantization function, which works to some extended but by no means the F=ma of
    the problem.
    Inplace quantization can be used to implement relax-and-round, which is a common method in Discrete Optimization's
    or Integer Programming.
    """
    for name, module in model.named_modules():
        if hasattr(module, '_weight_quantizer') and module.weight_quantizer is not None:
            if not module.weight_quantizer.fake_quant:
                logging.warning(("In-place real quantization is VERY dangerous and should be used for inference only. "
                                 "Make sure that is the desired behavior."))
            logging.warning("In-place quantize weight of %s", name)
            module.weight.data.copy_(module.weight_quantizer(module.weight))
