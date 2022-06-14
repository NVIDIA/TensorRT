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

import os
import json
from trex import *
import pytest


def test_ai():
    # computing a 3x3 convolution on a 256x56x56x64 input tensor, producing a 256x56x56x128 output,
    # all in half-precision, has an arithmetic intensity of 383.8 FLOPS/byte.
    R, S = 3, 3
    N = 256
    C, H, W = 64,  56, 56
    K, P, Q = 128, 56, 56
    G = 1
    weights_size = K * C * R * S
    n_ops = N * K * P * Q * C * R * S / G
    n_bytes = (N * C * H * W + weights_size + N * K * P * Q)
    ai = n_ops / n_bytes
    print(f"ai = {ai}")


@pytest.fixture
def plan():
    engine_prefix = "tests/inputs/mobilenet.qat.onnx.engine"
    plan = EnginePlan(f'{engine_prefix}.graph.json', f'{engine_prefix}.profile.json')
    return plan


def test_summary(plan):
    assert pytest.approx(plan.total_runtime, 0.001) == 0.470
    assert len(plan.layers) == 72
    assert plan.total_weights_size == 3469760
    assert plan.total_act_size == 16629344
    d = summary_dict(plan)
    assert d["Inputs"] == "[input1: [1, 3, 224, 224]xFP32 NCHW]"
    assert d["Average time"] == "0.470 ms"
    assert d["Layers"] == "72"
    assert d["Weights"] == "3.3 MB"
    assert d["Activations"] == "15.9 MB"


def test_layer_types(plan):
    assert set(plan.df['type']) == {'Convolution', 'Pooling', 'Reformat'}
    df = plan.get_layers_by_type('Convolution')
    assert pytest.approx(df['latency.pct_time'].sum(), 0.01) == 81.49
    assert len(df) == 53
    df = plan.get_layers_by_type('Pooling')
    assert pytest.approx(df['latency.pct_time'].sum(), 0.01) == 0.96
    assert len(df) == 1
    df = plan.get_layers_by_type('Reformat')
    assert pytest.approx(df['latency.pct_time'].sum(), 0.01) == 17.55
    assert len(df) == 18


def test_values(plan):
    # Random samples
    assert plan.df['weights_size'][1] == 864
    assert plan.df['weights_size'][6] == 2304
    assert plan.df['total_io_size_bytes'][11] == 225792
    assert plan.df['total_footprint_bytes'][5] == 1506144


def test_convolutions(plan):
    convs = plan.get_layers_by_type('Convolution')
    assert len(convs) == 53

    conv0 = convs.iloc[0]
    assert conv0['Name'] == " + ".join([
        "features.0.0.weight",
        "QuantizeLinear_8",
        "Conv_12 + PWN(Clip_16)"])

    G = 1
    assert conv0['attr.groups'] == G
    R, S = (3, 3)
    assert conv0['attr.kernel'] == (R, S)
    assert conv0['attr.out_maps'] == 32

    # Test Activations
    raw_inputs = conv0.Inputs
    assert raw_inputs[0]['Dimensions'] == [1, 3, 224, 224]
    inputs, outputs = create_activations(conv0)
    assert len(inputs)  > 0
    assert raw_inputs[0]['Dimensions'] == inputs[0].shape
    N, C, H, W = inputs[0].shape

    raw_outputs = conv0.Outputs
    assert raw_outputs[0]['Dimensions'] ==  [1, 32, 112, 112]
    assert len(outputs)  > 0
    assert raw_outputs[0]['Dimensions'] == outputs[0].shape
    N, K, P, Q = outputs[0].shape

    weights_size = K * C * R * S
    assert conv0['Weights']['Count'] == weights_size
    assert conv0['attr.M'] == N * P * Q
    assert conv0['attr.K'] == C * R * S
    assert conv0['attr.N'] == K

    # Arithmetic intensity
    n_ops = N * K * P * Q * C * R * S / G
    n_bytes = (N * C * H * W + weights_size + N * K * P * Q)
    ai = n_ops / n_bytes
    assert conv0['attr.arithmetic_intensity'] == ai


if __name__ == "__main__":
    engine_prefix = "tests/inputs/mobilenet.qat.onnx.engine"
    p = EnginePlan(f'{engine_prefix}.graph.json', f'{engine_prefix}.profile.json')
    test_summary(p)
