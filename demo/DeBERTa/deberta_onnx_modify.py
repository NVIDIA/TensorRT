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

'''
Modify original ONNX exported from HuggingFace to for TensorRT engine building.

The original HuggingFace implementation has uint8 Cast operations that TensorRT doesn't support, which needs to be removed from the ONNX model. After this step, the ONNX model can run in TensorRT.

Further, to use the DeBERTa plugin optimizations, the disentangled attention module needs to be replaced by node named `DisentangledAttention_TRT`.

Optional: generate model that has per-layer intermediate outputs for correctness check purpose.

These modifications are automated in this script.

Usage:
    python deberta_onnx_modify.py xx.onnx # for original TRT-compatible model, `xx_original.onnx`
    python deberta_onnx_modify.py xx.onnx --plugin # for TRT-compatible model with plugin nodes, `xx_plugin.onnx`
    python deberta_onnx_modify.py xx.onnx --correctness-check # for correctness check
'''

import onnx
from onnx import TensorProto
import onnx_graphsurgeon as gs
import argparse, os
import numpy as np

parser = argparse.ArgumentParser(description="Modify DeBERTa ONNX model to prepare for TensorRT engine building. If none of --plugin or --correctness-check flag is passed, it will just save the uint8 cast removed model.")
parser.add_argument('input', type=str, help='Path to the input ONNX model')
parser.add_argument('--plugin', action='store_true', help="Generate model with plugin")
parser.add_argument('--correctness-check', action='store_true', help="Generate model that has per-layer intermediate outputs for correctness check purpose")
parser.add_argument('--output', type=str, help="Path to the output ONNX model. If not set, default to the input file name with a suffix of '_original' or '_plugin' ")
args = parser.parse_args()

model_input = args.input
use_plugin = args.plugin
correctness_check = args.correctness_check

if args.output is None:
    model_output = os.path.splitext(model_input)[0] + ("_plugin" if use_plugin else "_original") + os.path.splitext(model_input)[-1]
else:
    model_output = args.output 

def remove_uint8_cast(graph):
    '''
    Remove all uint8 Cast nodes since TRT doesn't support UINT8 cast op.
    Ref: https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/06_removing_nodes
    '''
    nodes = [node for node in graph.nodes if node.op == 'Cast' and node.attrs["to"] == TensorProto.UINT8] # find by op name and attribute

    for node in nodes:
        # [ONNX's Cast operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast) will exactly have 1 input and 1 output
        # reconnect tensors
        input_node = node.i()
        input_node.outputs = node.outputs
        node.outputs.clear()

        # an alternative way is to just not cast to uint8
        # node.attrs["to"] = TensorProto.INT64

    return graph
 
@gs.Graph.register()
def insert_disentangled_attention(self, inputs, outputs, factor, span):
    '''
    Fuse disentangled attention module (Add + Gather + Gather + Transpose + Add + Div)

    inputs: list of plugin inputs
    outputs: list of plugin outputs
    factor: scaling factor of disentangled attention, sqrt(3d), converted from a division factor to a multiplying factor 
    span: relative distance span, k
    '''
    # disconnect previous output from flow (the previous subgraph still exists but is effectively dead since it has no link to an output tensor, and thus will be cleaned up)
    [out.inputs.clear() for out in outputs]
    # add plugin layer
    attrs = {
        "factor": 1/factor,
        "span": span
    }
    self.layer(op='DisentangledAttention_TRT', inputs=inputs, outputs=outputs, attrs=attrs)

def insert_disentangled_attention_all(graph):
    '''
    Insert disentangled attention plugin nodes for all layers
    '''
    nodes = [node for node in graph.nodes if node.op == 'GatherElements'] # find entry points by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    for l, (left,right) in enumerate(layers):
        print(f"Fusing layer {l}")
        
        # CAVEAT! MUST cast to list() when setting the inputs & outputs. graphsurgeon's default for X.inputs and X.outputs is `onnx_graphsurgeon.util.misc.SynchronizedList`, i.e. 2-way node-tensor updating mechanism. If not cast, when we remove the input nodes of a tensor, the tensor itself will be removed as well...
        
        # inputs: (data0, data1, data2), input tensors for c2c add and 2 gathers
        inputs = list(left.o().o().o().o().i().inputs)[0:1] + list(left.inputs)[0:1] + list(right.inputs)[0:1]
        
        # outputs: (result), output tensors after adding 3 gather results
        outputs = list(left.o().o().o().o().outputs)
        
        # constants: scaling factor, relative distance span
        factor = left.o().inputs[1].inputs[0].attrs["value"].values.item()
        span = right.i(1,0).i().i().i().inputs[1].inputs[0].attrs["value"].values.item()

        # insert plugin layer        
        graph.insert_disentangled_attention(inputs, outputs, factor, span) 

    return graph

def correctness_check_models(graph):
    '''
    Add output nodes at the plugin exit point for both the original model and the model with plugin
    '''

    seq_len = graph.inputs[0].shape[1]

    ## for original graph
    # make a copy of the graph first
    graph_raw = graph.copy()
    nodes = [node for node in graph_raw.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    original_output_all = []
    for l, (left,right) in enumerate(layers):
        # outputs: (result), output tensors after adding 3 gather results
        # add the output tensor to the graph outputs list. Don't create any new tensor!
        end_node = left.o().o().o().o()
        end_node.outputs[0].dtype = graph_raw.outputs[0].dtype # need to explicitly specify dtype and shape of graph output tensor
        end_node.outputs[0].shape = ['batch_size*num_heads', seq_len, seq_len]
        original_output_all.append(end_node.outputs[0])
      
    graph_raw.outputs = graph_raw.outputs + original_output_all # add plugin outputs to graph output

    ## for modified graph with plugin
    nodes = [node for node in graph.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    plugin_output_all = []
    for l, (left,right) in enumerate(layers):
        # inputs: (data0, data1, data2), input tensors for c2c add and 2 gathers
        inputs = list(left.o().o().o().o().i().inputs)[0:1] + list(left.inputs)[0:1] + list(right.inputs)[0:1]
        # outputs: (result), output tensors after adding 3 gather results
        outputs = list(left.o().o().o().o().outputs)
        end_node = left.o().o().o().o()
        end_node.outputs[0].dtype = graph.outputs[0].dtype # need to explicitly specify dtype and shape of graph output tensor
        end_node.outputs[0].shape = ['batch_size*num_heads', seq_len, seq_len]
        plugin_output_all.append(end_node.outputs[0]) # add to graph output (outside this loop)

        # constants: scaling factor, relative distance span
        factor = left.o().inputs[1].inputs[0].attrs["value"].values.item()
        span = right.i(1,0).i().i().i().inputs[1].inputs[0].attrs["value"].values.item()

        # insert plugin layer        
        graph.insert_disentangled_attention(inputs, outputs, factor, span) 

    graph.outputs = graph.outputs + plugin_output_all # add plugin outputs to graph output

    return graph_raw, graph
        
def check_model(model_name):
    # Load the ONNX model
    model = onnx.load(model_name)

    # Check that the model is well formed
    onnx.checker.check_model(model)

# load onnx
graph = gs.import_onnx(onnx.load(model_input))

# first, remove uint8 cast nodes
graph = remove_uint8_cast(graph)

if use_plugin:
    # save the modified model with plugin nodes

    # replace Add + Gather + Gather + Transpose + Add + Div (c2c and c2p and p2c) with DisentangledAttention_TRT node
    graph = insert_disentangled_attention_all(graph)

    # remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort()

    # export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), model_output)
    print(f"Saving modified model to {model_output}")

    # don't check model because 'DisentangledAttention_TRT' is not a registered op

elif correctness_check: 
    # correctness check, save two models (original and w/ plugin) with intermediate output nodes inserted
    graph_raw, graph = correctness_check_models(graph)

    # remove unused nodes, and topologically sort the graph.
    graph_raw.cleanup().toposort()
    graph.cleanup().toposort()

    # export the onnx graph from graphsurgeon
    model_output1 = os.path.splitext(model_input)[0] + "_correctness_check_original" + os.path.splitext(model_input)[-1]
    model_output2 = os.path.splitext(model_input)[0] + "_correctness_check_plugin" + os.path.splitext(model_input)[-1]
    onnx.save_model(gs.export_onnx(graph_raw), model_output1)
    onnx.save_model(gs.export_onnx(graph), model_output2)
    
    print(f"Saving models for correctness check to {model_output1} (original) and {model_output2} (with plugin)")

    check_model(model_output1)
    # don't check model_output2 because 'DisentangledAttention_TRT' is not a registered op

else:
    # no flag passed, save model with just uint8 cast removed
    graph.cleanup().toposort()
    onnx.save_model(gs.export_onnx(graph), model_output)
    print(f"Saving modified model to {model_output}")
    check_model(model_output)
