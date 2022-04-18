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
import argparse

COORD_CONV_AC_OP_TYPE = 'CoordConvAC'

# Here we'll register a function to do all the subgraph-replacement heavy-lifting
@gs.Graph.register()
def replace_with_coordconvac(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op=COORD_CONV_AC_OP_TYPE, inputs=inputs, outputs=outputs)


def main():
    '''
    Replace each unfolded CoordConv graph with a single CoordConv node.  
    From
    ... -> (CoordConv subgraph) -> Conv -> Relu -> (CoordConv subgraph) -> ...
    To
    ... -> CoordConv -> Conv -> Relu -> CoordConv -> ...
    '''
    # Configurable parameters from command line
    parser = argparse.ArgumentParser(description='ONNX Modifying Example')
    parser.add_argument('--onnx', default="mnist_cc.onnx",
                        help='onnx file to modify')
    parser.add_argument('--output', default="mnist_with_coordconv.onnx",
                        help='input batch size for testing (default: output.onnx)')
    args = parser.parse_args()
    
    # Load ONNX file
    graph = gs.import_onnx(onnx.load(args.onnx))

    tmap = graph.tensors()
    # You can figure out the input and output tensors using Netron.
    inputs = [tmap["conv1"]]
    outputs = [tmap["90"]]
    graph.replace_with_coordconvac(inputs, outputs)

    inputs = [tmap["92"]]
    outputs = [tmap["170"]]
    graph.replace_with_coordconvac(inputs, outputs)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

    # Save the modified model.
    onnx.save(gs.export_onnx(graph), "mnist_with_coordconv.onnx")

if __name__ == '__main__':
    main()

    