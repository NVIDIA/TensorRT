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
Postprocessing script to add precision constraints to a TensorRT network.
"""

import tensorrt as trt

def postprocess(network):
    """
    Traverses the parsed network and constrains precisions
    for specific layers to FP32.

    Args:
        network (trt.INetworkDefinition): The network to modify.

    Returns:
        None
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
