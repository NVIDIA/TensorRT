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


"""tests of integrating Quant layers into a network"""

import pytest

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from tests.fixtures.models import LeNet, QuantLeNet
from tests.fixtures import verbose

np.random.seed(12345)  # seed 1234 causes 1 number mismatch at 6th decimal in one of the tests

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# pylint:disable=missing-docstring, no-self-use

class TestNetwork():
    """test basic operations of quantized network"""

    def test_simple_build(self):
        """test instantiation"""
        quant_model = QuantLeNet(quant_desc_input=QuantDescriptor(), quant_desc_weight=QuantDescriptor())
        for name, module in quant_model.named_modules():
            if "quantizer" in name:
                module.disable()

        input_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        weight_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        quant_model = QuantLeNet(quant_desc_input=input_desc, quant_desc_weight=weight_desc)

        input_desc = QuantDescriptor(amax=6.)
        weight_desc = QuantDescriptor(amax=1.)
        quant_model = QuantLeNet(quant_desc_input=input_desc, quant_desc_weight=weight_desc)


    def test_forward(self):
        """test forward pass with random data"""
        input_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        weight_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        quant_model = QuantLeNet(quant_desc_input=input_desc, quant_desc_weight=weight_desc)
        output = quant_model(torch.empty(16, 1, 28, 28))

    def test_backward(self):
        """test one iteration with random data and labels"""
        input_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        weight_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        quant_model = QuantLeNet(quant_desc_input=input_desc, quant_desc_weight=weight_desc)
        optimizer = optim.SGD(quant_model.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = quant_model(torch.empty(16, 1, 28, 28))
        loss = F.nll_loss(output, torch.randint(10, (16,), dtype=torch.int64))
        loss.backward()
        optimizer.step()

    def test_native_amp_fp16(self):
        """test one iteration with random data and labels"""
        input_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        weight_desc = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
        model = QuantLeNet(quant_desc_input=input_desc, quant_desc_weight=weight_desc)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(torch.empty(16, 1, 28, 28))
            loss = F.nll_loss(output, torch.randint(10, (16,), dtype=torch.int64))
        loss.backward()
        optimizer.step()
        assert loss.dtype == torch.float32

    def test_asp(self):
        """test Sparsity (ASP) and QAT toolkits together"""
        try:
            from apex.contrib.sparsity import ASP
        except ImportError:
            pytest.skip("ASP is not available.")

        quant_modules.initialize()
        model = LeNet()
        quant_modules.deactivate()

        optimizer = optim.SGD(model.parameters(), lr=0.01)

        ASP.init_model_for_pruning(
            model,
            mask_calculator="m4n2_1d",
            verbosity=2,
            whitelist=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, quant_nn.modules.quant_linear.QuantLinear],
            allow_recompute_mask=False,
            custom_layer_dict={
                quant_nn.QuantConv1d: ['weight'],
                quant_nn.QuantConv2d: ['weight'],
                quant_nn.QuantConv3d: ['weight'],
                quant_nn.QuantConvTranspose1d: ['weight'],
                quant_nn.QuantConvTranspose2d: ['weight'],
                quant_nn.QuantConvTranspose3d: ['weight'],
                quant_nn.QuantLinear: ['weight']
            },
            allow_permutation=False)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()

        model = model.to('cuda')
        output = model(torch.empty(16, 1, 28, 28).to('cuda'))
        optimizer.zero_grad()
        loss = F.nll_loss(output, torch.randint(10, (16,), dtype=torch.int64))
        loss.backward()
        optimizer.step()

    def test_quant_module_replacement(self):
        """test monkey patching of modules with their quantized versions"""
        lenet = LeNet()
        qlenet = QuantLeNet()

        mod_list = [type(mod) for name, mod in lenet.named_modules()]
        mod_list = mod_list[1:]    
        qmod_list = [type(mod) for name, mod in qlenet.named_modules()]
        qmod_list = qmod_list[1:]  

        # Before any monkey patching, the networks should be different
        assert(mod_list != qmod_list)

        # Monkey patch the modules
        no_replace_list = ["Linear"]
        custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]

        quant_modules.initialize(no_replace_list, custom_quant_modules)

        lenet = LeNet()
        qlenet = QuantLeNet()
    
        mod_list = [type(mod) for name, mod in lenet.named_modules()]
        mod_list = mod_list[1:]    
        qmod_list = [type(mod) for name, mod in qlenet.named_modules()]
        qmod_list = qmod_list[1:]

        # After monkey patching, the networks should be same
        assert(mod_list == qmod_list)

        # Reverse monkey patching
        quant_modules.deactivate()

        lenet = LeNet()
        qlenet = QuantLeNet()
    
        mod_list = [type(mod) for name, mod in lenet.named_modules()]
        mod_list = mod_list[1:]    
        qmod_list = [type(mod) for name, mod in qlenet.named_modules()]
        qmod_list = qmod_list[1:]

        # After reversing monkey patching, the networks should again be different
        assert(mod_list != qmod_list)

    def test_calibration(self):
        quant_model = QuantLeNet(quant_desc_input=QuantDescriptor(), quant_desc_weight=QuantDescriptor()).cuda()

        for name, module in quant_model.named_modules():
            if name.endswith("_quantizer"):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
                print(F"{name:40}: {module}")

        quant_model(torch.rand(16, 1, 224, 224, device="cuda"))

        # Load calib result and disable calibration
        for name, module in quant_model.named_modules():
            if name.endswith("_quantizer"):
                if module._calibrator is not None:
                    module.load_calib_amax()
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
        quant_model.cuda()

