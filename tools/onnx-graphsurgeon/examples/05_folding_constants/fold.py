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

import onnx_graphsurgeon as gs
import onnx

print("Graph.fold_constants Help:\n{}".format(gs.Graph.fold_constants.__doc__))

graph = gs.import_onnx(onnx.load("model.onnx"))

# Fold constants in the graph using ONNX Runtime. This will replace
# expressions that can be evaluated prior to runtime with constant tensors.
# The `fold_constants()` function will not, however, remove the nodes that
# it replaced - it simply changes the inputs of subsequent nodes.
# To remove these unused nodes, we can follow up `fold_constants()` with `cleanup()`
graph.fold_constants().cleanup()

onnx.save(gs.export_onnx(graph), "folded.onnx")
