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


"""tests of supportive functions"""

import pytest
import numpy as np

import torch

import pytorch_quantization.nn.functional as QF

np.random.seed(1234)
torch.manual_seed(1234)

# pylint:disable=missing-docstring, no-self-use

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class TestClip():

    def test_simple_run(self):
        x_np = np.random.rand(1023).astype(np.float32)
        x_torch = torch.Tensor(x_np)
        clip_x_np = np.clip(x_np, 0.3, 0.7)
        clip_x_torch = QF.clip(x_torch, torch.tensor(0.3), torch.tensor(0.7))
        np.testing.assert_array_equal(clip_x_torch.cpu().numpy(), clip_x_np)

    def test_raise(self):
        x = torch.randn(3, 7, requires_grad=True)

        min_value = torch.Tensor(3, 7)
        max_value = torch.Tensor(3, 7)
        min_value.requires_grad = True
        max_value.requires_grad = True
        clip_x = QF.clip(x, min_value, max_value)

        labels = torch.randint(6, (3,)).type(torch.LongTensor).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(clip_x, labels)
        with pytest.raises(ValueError, match="can only be scalar"):
            loss.backward()

    def test_broadcast(self):
        """Test broadcast behavior by randomly picked shuffling of np.random.rand"""
        x_np = np.random.rand(1023, 4, 5, 6).astype(np.float32) - 0.5
        x_torch = torch.Tensor(x_np)
        min_value = np.random.rand(1, 4, 1, 1).astype(np.float32) * 0.1 - 0.2
        max_value = np.random.rand(1, 4, 1, 1).astype(np.float32) * 10 + 0.5
        clip_x_np = np.clip(x_np, min_value, max_value)
        clip_x_torch = QF.clip(x_torch, torch.tensor(min_value), torch.tensor(max_value))
        np.testing.assert_array_equal(clip_x_torch.cpu().numpy(), clip_x_np)

    def test_backward(self):
        x = torch.randn(3, 1025, requires_grad=True)
        x.retain_grad()

        min_value = torch.tensor(0.3)
        max_value = torch.tensor(0.7)
        min_value.requires_grad = True
        max_value.requires_grad = True
        min_value.retain_grad()
        max_value.retain_grad()
        clip_x = QF.clip(x, min_value, max_value)
        clip_x.retain_grad()

        labels = torch.randint(6, (3,)).type(torch.LongTensor).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(clip_x, labels)
        loss.backward()

        np.testing.assert_array_almost_equal(
            clip_x.grad[x < min_value].sum().cpu().numpy(), min_value.grad.cpu().numpy(), decimal=6)
        np.testing.assert_array_almost_equal(
            clip_x.grad[x > max_value].sum().cpu().numpy(), max_value.grad.cpu().numpy(), decimal=6)
        assert x.grad.cpu()[x.cpu() < min_value.cpu()].sum() == 0
        assert x.grad.cpu()[x.cpu() > max_value.cpu()].sum() == 0
        assert torch.equal(clip_x.grad[(x > min_value) & (x < max_value)], x.grad[(x > min_value) & (x < max_value)])
