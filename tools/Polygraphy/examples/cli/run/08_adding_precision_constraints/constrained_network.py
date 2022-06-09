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
Parses an ONNX model, then adds precision constraints so specific layers run in FP32.
"""

from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
import tensorrt as trt

# Load the model, which implements the following network:
#
# x -> MatMul (I_rot90) -> Add (FP16_MAX) -> Sub (FP16_MAX) -> MatMul (I_rot90) -> out
#
# Without constraining the subgraph (Add -> Sub) to FP32, this model may
# produce incorrect results when run with FP16 optimziations enabled.
parse_network_from_onnx = NetworkFromOnnxPath("./needs_constraints.onnx")


@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    """The below function traverses the parsed network and constrains precisions
    for specific layers to FP32.

    See examples/cli/run/04_defining_a_tensorrt_network_or_config_manually
    for more examples using network scripts in Polygraphy.
    """
    for layer in network:
        # Set computation precision for Add and Sub layer to FP32
        if layer.name in ("Add", "Sub"):
            layer.precision = trt.float32

        # Set the output precision for the Add layer to FP32.  Without this,
        # the intermediate output data of the Add may be stored as FP16 even
        # though the computation itself is performed in FP32.
        if layer.name == "Add":
            layer.set_output_type(0, trt.float32)
