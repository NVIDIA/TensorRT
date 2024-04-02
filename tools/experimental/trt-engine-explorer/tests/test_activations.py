#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from .util import plan
from trex import create_activations, Activation

def test_create_activations(plan):
    pd_series = plan.get_layers_by_type('Convolution').iloc[0]
    inputs, outputs = create_activations(pd_series)
    assert isinstance(inputs[0], (Activation))
    assert isinstance(outputs[0], (Activation))

class TestActivation:
    def test_initialization(self, plan):
        pd_series = plan.df.iloc[0].Inputs[0]
        # Test the input pandas series
        assert isinstance(pd_series, (dict))
        assert pd_series['Name'] == 'input1'
        assert pd_series['Dimensions'] == [1, 3, 224, 224]
        # Test the Activation object
        activation = Activation(pd_series)
        assert activation.name == 'input1'
        assert activation.shape == [1, 3, 224, 224]
        assert activation.format == 'FP32 NCHW'
        assert activation.precision == 'FP32'
        assert activation.data_size == 4
        assert activation.size_bytes == 602112
