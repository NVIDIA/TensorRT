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


"""tests of Clip module."""

import pytest
import numpy as np

import torch

from pytorch_quantization.nn.modules import clip

# make everything run on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

np.random.seed(1234)
torch.manual_seed(1234)

# pylint:disable=missing-docstring, no-self-use


class TestClip():

    def test_simple_run(self):
        x_np = np.random.rand(1023).astype(np.float32)
        x_torch = torch.Tensor(x_np)
        clip_op = clip.Clip(torch.tensor(0.3), torch.tensor(0.7))
        clip_x_np = np.clip(x_np, 0.3, 0.7)
        clip_x_torch = clip_op(x_torch)
        np.testing.assert_array_equal(clip_x_torch.cpu().numpy(), clip_x_np)

    def test_raise(self):
        with pytest.raises(ValueError, match="must be scalar"):
            clip_op = clip.Clip(torch.tensor(0.3), torch.tensor(0.7), learn_min=True)

    def test_backward(self):
        x = torch.randn(3, 7, requires_grad=True)
        x.retain_grad()

        min_value = 0.3
        max_value = 0.7
        clip_op = clip.Clip(min_value, max_value, learn_min=True, learn_max=True)
        clip_x = clip_op(x)
        clip_x.retain_grad()

        labels = torch.randint(6, (3,)).type(torch.LongTensor).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(clip_x, labels)

        loss.backward()

        assert x.grad.cpu()[x.cpu() < min_value].sum() == 0
        assert x.grad.cpu()[x.cpu() > max_value].sum() == 0
        assert torch.equal(clip_x.grad[(x > min_value) & (x < max_value)], x.grad[(x > min_value) & (x < max_value)])
