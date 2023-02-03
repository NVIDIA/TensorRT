#!/usr/bin/env python3
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
"""
Parses an ONNX model, and then extends it with an Identity layer.
"""
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath

parse_onnx = NetworkFromOnnxPath("identity.onnx")

# If we define a function called `load_network`, polygraphy can
# use it directly in place of using a model file.
#
# TIP: If our function isn't called `load_network`, we can explicitly specify
# the name with the model argument, separated by a colon. For example, `define_network.py:my_func`.
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    # NOTE: func.extend() causes the signature of this function to be `() -> (builder, network, parser)`
    # For details on how this works, see examples/api/03_interoperating_with_tensorrt

    # Append an identity layer to the network
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)

    output = network.add_identity(prev_output).get_output(0)
    network.mark_output(output)

    # Notice that we don't need to return anything - `extend()` takes care of that for us!
