#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import onnx_graphsurgeon as gs

from demo_diffusion import utils_modelopt


def create_graph_with_fp16_resize() -> gs.Graph:
    """Return a gs.Graph with a single Resize node with FP16 input."""
    return gs.Graph(
        nodes=[
            gs.Node(
                op="Resize",
                name="my_resize_node",
                inputs=[
                    gs.Variable(name="X", dtype=np.float16),
                    gs.Variable(name="roi", dtype=np.float16),
                    gs.Variable(name="scales", dtype=np.float16),
                    gs.Variable(name="sizes", dtype=np.int64),
                ],
                outputs=[gs.Variable(name="Y", dtype=np.float16)],
            )
        ]
    )


def test_should_cast_resize_to_fp32() -> None:
    """Test that `cast_resize_to_fp32` correctly casts all Resize nodes to FP32."""
    # Precondition.
    graph = create_graph_with_fp16_resize()

    # Under test.
    utils_modelopt.cast_resize_io(graph)

    # Postcondition.
    has_resize = False
    for node in graph.nodes:
        if node.op == "Resize":
            has_resize = True
            x, roi, scales, sizes = node.inputs
            assert x.dtype == np.float32
            assert roi.dtype == np.float32
            assert scales.dtype == np.float32
            assert sizes.dtype == np.int64  # "sizes" is the exception input that cannot be cast to FP32.

    assert has_resize
