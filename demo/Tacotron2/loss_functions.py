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

import torch
import torch.nn as nn
from tacotron2.loss_function import Tacotron2Loss
from waveglow.loss_function import WaveGlowLoss


def get_loss_function(loss_function, sigma=1.0):
    if loss_function == 'Tacotron2':
        loss = Tacotron2Loss()
    elif loss_function == 'WaveGlow':
        loss = WaveGlowLoss(sigma=sigma)
    else:
        raise NotImplementedError(
            "unknown loss function requested: {}".format(loss_function))

    loss.cuda()
    return loss
