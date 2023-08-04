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


"""Tests of Quant Module Replacement"""
import pytest
import numpy as np

import torch

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.quant_modules import QuantModuleReplacementHelper
import tests.utils as test_utils
from tests.fixtures import verbose

# pylint:disable=missing-docstring, no-self-use

class TestQuantModuleReplace():

    def test_simple_default_args(self):
        replacement_helper = QuantModuleReplacementHelper()
        replacement_helper.prepare_state()
        replacement_helper.apply_quant_modules()

        # Linear module should not be replaced with its quantized version
        assert(type(quant_nn.QuantLinear(16, 256, 3)) == type(torch.nn.Linear(16, 256, 3)))
        assert(type(quant_nn.QuantConv2d(16, 256, 3)) == type(torch.nn.Conv2d(16, 256, 3)))

        replacement_helper.restore_float_modules()

    def test_with_no_replace_list(self):
        no_replace_list = ["Linear"]
        custom_quant_modules = None
        replacement_helper = QuantModuleReplacementHelper()
        replacement_helper.prepare_state(no_replace_list, custom_quant_modules)
        replacement_helper.apply_quant_modules()

        # Linear module should not be replaced with its quantized version
        assert(type(quant_nn.QuantLinear(16, 256, 3)) != type(torch.nn.Linear(16, 256, 3)))
        assert(type(quant_nn.QuantConv2d(16, 256, 3)) == type(torch.nn.Conv2d(16, 256, 3)))

        replacement_helper.restore_float_modules()

    def test_with_custom_quant_modules(self):
        no_replace_list = ["Linear"]
        custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]
        replacement_helper = QuantModuleReplacementHelper()
        replacement_helper.prepare_state(no_replace_list, custom_quant_modules)
        replacement_helper.apply_quant_modules()

        # Although no replace list indicates Linear module should not be replaced with its
        # quantized version, since the custom_quant_modules still contains the Linear module's
        # mapping, it will replaced.
        assert(type(quant_nn.QuantLinear(16, 256, 3)) == type(torch.nn.Linear(16, 256, 3)))
        assert(type(quant_nn.QuantConv2d(16, 256, 3)) == type(torch.nn.Conv2d(16, 256, 3)))

        replacement_helper.restore_float_modules()

    def test_initialize_deactivate(self):
        no_replace_list = ["Linear"]
        custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]

        quant_modules.initialize(no_replace_list, custom_quant_modules)

        assert(type(quant_nn.QuantLinear(16, 256, 3)) == type(torch.nn.Linear(16, 256, 3)))
        assert(type(quant_nn.QuantConv2d(16, 256, 3)) == type(torch.nn.Conv2d(16, 256, 3)))

        quant_modules.deactivate()
