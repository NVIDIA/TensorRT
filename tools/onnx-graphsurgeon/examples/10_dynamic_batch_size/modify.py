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

import onnx
import onnx_graphsurgeon as gs

# Load ONNX model
graph = gs.import_onnx(onnx.load("model.onnx"))

# Update input shape
for input in graph.inputs:
    input.shape[0] = 'N'

# Update 'Reshape' nodes (if they exist)
reshape_nodes = [node for node in graph.nodes if node.op == "Reshape"]
for node in reshape_nodes:
    # The batch dimension in the input shape is hard-coded to a static value in the original model.
    # To make the model work with our dynamic batch size, we can use a `-1`, which indicates that the
    # dimension should be automatically determined.
    node.inputs[1].values[0] = -1

# Save dynamic model
onnx.save(gs.export_onnx(graph), "dynamic.onnx")
