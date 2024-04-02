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


"""Tests of helper functions for quant optimizer"""

import numpy as np
import pytest


import torch.optim as optim

from pytorch_quantization.optim import helper
from pytorch_quantization.tensor_quant import QuantDescriptor
from .fixtures.models import QuantLeNet
from .fixtures.models import resnet18

# pylint:disable=missing-docstring, no-self-use

class TestMatchParameters():

    def test_single_key(self, resnet18):
        param = helper.match_parameters(resnet18, ['downsample.0.weight'])
        assert len(list(param)) == 3

    def test_multi_keys(self, resnet18):
        param = list(helper.match_parameters(resnet18, ['conv1', 'downsample']))
        assert len(param) == 18

    def test_regex(self, resnet18):
        param = helper.match_parameters(resnet18, ['downsample.*.weight$'])
        assert len(list(param)) == 6

        param = helper.match_parameters(resnet18, ['downsample.*.wei$'])
        assert not list(param)


class TestGroupParameters():

    def test_single_key(self, resnet18):
        param_groups = helper.group_parameters(resnet18, [['downsample.1.weight']])
        assert len(list(param_groups[0]['params'])) == 3

    def test_lr_momentum_decay(self, resnet18):
        lrs = [0.01, 0.001]
        momentums = [0.02, 0.002]
        weight_decays = [0.03, 0.003]
        param_groups = helper.group_parameters(
            resnet18, [['conv1.*weight'], ['downsample.*.weight']], lrs, momentums, weight_decays)

        assert param_groups[0]['lr'] == lrs[0]
        assert param_groups[1]['lr'] == lrs[1]
        assert param_groups[0]['momentum'] == momentums[0]
        assert param_groups[1]['momentum'] == momentums[1]
        assert param_groups[0]['weight_decay'] == weight_decays[0]
        assert param_groups[1]['weight_decay'] == weight_decays[1]

    def test_optimizer_feed(self, resnet18):
        """Feed grouped parameters to optimizer, see what happens"""
        lrs = [0.01, 0.001]
        momentums = [0.02, 0.002]
        weight_decays = [0.03, 0.003]
        param_groups = helper.group_parameters(
            resnet18, [['conv1.*weight'], ['downsample.*.weight']], lrs, momentums, weight_decays)
        optimizer = optim.SGD(param_groups)
        optimizer.step()

    def test_raises(self):
        with pytest.raises(TypeError, match="must be list of list of patterns"):
            helper.group_parameters(None, [['downsample.1.weight'], 'conv1'])

        with pytest.raises(TypeError, match="must match"):
            helper.group_parameters(None, [['downsample.1.weight'], ['conv1']], lrs=[0.1])

        with pytest.raises(TypeError, match="must match"):
            helper.group_parameters(None, [['downsample.1.weight'], ['conv1']], momentums=[0.1])

        with pytest.raises(TypeError, match="must match"):
            helper.group_parameters(None, [['downsample.1.weight'], ['conv1']], weight_decays=[0.1])


class TestFreezeParameters():

    def test_simple(self, resnet18):
        helper.freeze_parameters(resnet18, ['downsample.0.weight'])
        for name, param in resnet18.named_parameters():
            if 'downsample.0.weight' in name:
                assert not param.requires_grad

class TestQuantWeightInPlace():

    def test_simple(self):
        quant_lenet = QuantLeNet(
            quant_desc_input=QuantDescriptor(),
            quant_desc_weight=QuantDescriptor())
        quant_lenet.eval()
        helper.quant_weight_inplace(quant_lenet)
