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


"""Model used for tests"""

import pytest
import torch.nn as nn
import torch.nn.functional as F
from pytorch_quantization.nn import QuantConv2d, QuantLinear
from pytorch_quantization.tensor_quant import QuantDescriptor

class LeNet(nn.Module):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, **kwargs)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, **kwargs)
        self.fc1 = nn.Linear(320, 50, **kwargs)
        self.fc2 = nn.Linear(50, 10, **kwargs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class QuantLeNet(nn.Module):
    def __init__(self, **kwargs):
        super(QuantLeNet, self).__init__()
        self.conv1 = QuantConv2d(1, 10, kernel_size=5, **kwargs)
        self.conv2 = QuantConv2d(10, 20, kernel_size=5, **kwargs)
        self.fc1 = QuantLinear(320, 50, **kwargs)
        self.fc2 = QuantLinear(50, 10, **kwargs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

@pytest.fixture
def resnet18():
    import torchvision
    return torchvision.models.resnet18()

@pytest.fixture
def quant_lenet():
    return QuantLeNet(quant_desc_input=QuantDescriptor(), quant_desc_weight=QuantDescriptor())
