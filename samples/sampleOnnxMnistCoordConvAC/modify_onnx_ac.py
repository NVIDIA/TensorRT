##########################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://github.com/onnx/onnx/issues/2259
# onnx-graphsurgeon
##########################################################################



import torch
import onnx
import argparse

COORD_CONV_AC_OP_TYPE = 'CoordConvAC'
COORD_CONV_OP_TYPE = 'Conv'


def main():
    # Configurable parameters from command line
    parser = argparse.ArgumentParser(description='ONNX Modifying Example')
    parser.add_argument('--onnx', default="mnist_cc.onnx",
                        help='onnx file to modify')
    parser.add_argument('--output', default="mnist_with_coordconv.onnx",
                        help='input batch size for testing (default: output.onnx)')
    args = parser.parse_args()
    
    # Load ONNX file
    model = onnx.load(args.onnx)
    
    # Retrieve graph_def
    graph = model.graph

    node_input_new = 'conv1' 

    counter_conv_nodes_updated = 0
    nodes_to_delete = []

    # Iterate through all the nodes
    for i, node in enumerate(graph.node):
        
        if counter_conv_nodes_updated == 2:
            break

        if node.op_type == 'Conv':            
            # Update inputs of any Conv node and converting Conv->CoordConv
            graph.node[i].input.remove(graph.node[i].input[0])
            graph.node[i].input.insert(0, node_input_new)
            graph.node[i].op_type = COORD_CONV_OP_TYPE
            counter_conv_nodes_updated += 1
        elif node.op_type == 'Relu':
            # Saving output of previous node 
            node_input_new = graph.node[i].output[0]
        else:
            # Add node to list of removable nodes
            nodes_to_delete.append(i)
    
    for i in nodes_to_delete[::-1]:
        # Remove unnecessary nodes
        n = graph.node[i]
        graph.node.remove(n)
    
    # insert AC nodes 
    i = 0
    while i < len(graph.node):
        if graph.node[i].op_type == COORD_CONV_OP_TYPE:
            # Create an ac node
            node_ac = onnx.NodeProto()
            node_ac.op_type = COORD_CONV_AC_OP_TYPE
            node_ac.output.insert(0, f"ac_output_{i}")
            node_ac.input.insert(0, graph.node[i].input[0])
            graph.node[i].input[0] = f"ac_output_{i}"
            graph.node.insert(i, node_ac)
            i += 1
        i += 1

    # Generate model_cropped from modified graph
    model_cropped = onnx.helper.make_model(graph)

    print(onnx.helper.printable_graph(model_cropped.graph))

    print("Inputs:",  model_cropped.graph.node[0].input,
          "Outputs:", model_cropped.graph.node[-1].output)

    # Save the serialized model
    onnx.save(model_cropped, args.output)
        
if __name__ == '__main__':
    main()
