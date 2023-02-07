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
import sys
import os
import numpy as np
import argparse

def insert_decoder_loop(decoder_iter_onnx_path, output_dir, decoder_out_name, fp16):
    float_prec = np.float16 if fp16 else np.float32

    # Modify loop body so that it has 2+N inputs: (iteration_num, condition, loop carried dependencies...)
    # and 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...)

    # In this case, the loop carried dependencies include the following IN ORDER
    #    - decoder_output/decoder_input
    #    - attention_hidden
    #    - attention_cell
    #    - decoder_hidden
    #    - decoder_cell
    #    - attention_weights
    #    - attention_weights_cum
    #    - attention_context
    #    - not_finished (bool tensor, initialized to all True)
    #    - mel_lengths

    # The following are NOT loop carried dependencies (they remain constant through the loop), and must be moved to be inputs outside of the loop body
    #    - memory
    #    - processed_memory
    #    - mask

    # The scan outputs are
    #    - mel_outputs (which scans across decoder_output)
    #    - gate_outputs (scans across gate_prediction)
    #    - alignments (scans across attention_weights)


    loop_body = gs.import_onnx(onnx.load(decoder_iter_onnx_path))
    loop_tensors = loop_body.tensors()

    iteration_num = gs.Variable("iteration_num", dtype=np.int64, shape=())
    cond_in = gs.Variable("cond_in", dtype=bool, shape=())
    cond_out = gs.Variable("cond_out", dtype=bool, shape=())
    not_finished_in = gs.Variable("not_finished_in", shape=('batch_size', 1), dtype=bool)
    not_finished_out = gs.Variable("not_finished_out", shape=('batch_size', 1), dtype=bool)
    mel_lengths_in = gs.Variable("mel_lengths_in", shape=('batch_size', 1), dtype=np.int32)
    mel_lengths_out = gs.Variable("mel_lengths_out", shape=('batch_size', 1), dtype=np.int32)


    # Set loop body inputs in the correct order
    loop_body.inputs = [iteration_num, cond_in, loop_tensors["decoder_input"], loop_tensors["attention_hidden"], loop_tensors["attention_cell"], loop_tensors["decoder_hidden"], loop_tensors["decoder_cell"], loop_tensors["attention_weights"], loop_tensors["attention_weights_cum"], loop_tensors["attention_context"], not_finished_in, mel_lengths_in]

    # Set loop body outputs in the correct order
    loop_body.outputs = [cond_out, loop_tensors["decoder_output"], loop_tensors["out_attention_hidden"], loop_tensors["out_attention_cell"], loop_tensors["out_decoder_hidden"], loop_tensors["out_decoder_cell"], loop_tensors["out_attention_weights"], loop_tensors["out_attention_weights_cum"], loop_tensors["out_attention_context"], not_finished_out, mel_lengths_out, loop_tensors["decoder_output"], loop_tensors["gate_prediction"], loop_tensors["out_attention_weights"]]

    # The loop stop condition is given by the following lines in PyTorch
    #     dec = torch.le(torch.sigmoid(decoder_outputs[8]), gate_threshold).to(torch.int32).squeeze(1)
    #     not_finished = not_finished*dec
    #     if torch.sum(not_finished) == 0:
    #         break

    # To compute cond_out, we can essentially follow the same steps. Using Less instead of Greater+Not for now

    gate_threshold = gs.Constant("gate_threshold", np.array([0.5], dtype=float_prec))
    gate_sigmoid = gs.Variable("gate_sigmoid", dtype=float_prec, shape=())
    sigmoid = loop_body.nodes.append(gs.Node(op="Sigmoid", inputs=[loop_tensors["gate_prediction"]], outputs=[gate_sigmoid]))

    leq_output = gs.Variable("leq_output", dtype=bool)
    leq = loop_body.nodes.append(gs.Node(op="Less", inputs=[gate_sigmoid, gate_threshold], outputs=[leq_output]))

    loop_body.nodes.append(gs.Node(op="And", inputs=[not_finished_in, leq_output], outputs=[not_finished_out]))

    cast_output = gs.Variable("cast_output", dtype=np.int32)
    loop_body.nodes.append(gs.Node(op="Cast", inputs=[not_finished_out], outputs=[cast_output], attrs={"to": 6})) # int32

    reduce_output = gs.Variable("reduce_output", dtype=np.int32)
    loop_body.nodes.append( gs.Node(op="ReduceSum", inputs=[cast_output], outputs=[reduce_output], attrs={"axes": [0], "keepdims": 0}))

    unsqueezed_cond_out = gs.Variable("unsqueezed_cond_out", dtype=bool)
    loop_body.nodes.append(gs.Node(op="Equal", inputs=[reduce_output, gs.Constant("zero", np.array(0, dtype=np.int32))], outputs=[unsqueezed_cond_out]))

    squeezed_cond_out = gs.Variable("squeezed_cond_out", dtype=bool)
    loop_body.nodes.append(gs.Node(op="Squeeze", inputs=[unsqueezed_cond_out], outputs=[squeezed_cond_out], attrs={"axes": [0]}))

    loop_body.nodes.append(gs.Node(op="Not", inputs=[squeezed_cond_out], outputs=[cond_out]))

    # Compute mel_lengths
    #  from PyTorch:  mel_lengths += not_finished

    loop_body.nodes.append(gs.Node(op="Add", inputs=[mel_lengths_in, cast_output], outputs=[mel_lengths_out]))

    memory = gs.Variable("memory", dtype=float_prec, shape=('batch_size', 'seq_len', 512))
    processed_memory = gs.Variable("processed_memory", dtype=float_prec, shape=('batch_size', 'seq_len', 128))
    mask = gs.Variable("mask", dtype=bool, shape=('batch_size', 'seq_len'))

    loop_body.toposort()
    onnx.save(gs.export_onnx(loop_body), os.path.join(output_dir, "loop_body_{prec}.onnx".format(prec="fp16" if float_prec == np.float16 else "fp32")))

    # Create outer graph

    # Inputs to outer graph are the following (suffixed with _0 to signify initial states)
    #    - decoder_input_0
    #    - attention_hidden_0
    #    - attention_cell_0
    #    - decoder_hidden_0
    #    - decoder_cell_0
    #    - attention_weights_0
    #    - attention_weights_cum_0
    #    - attention_context_0
    #    - memory
    #    - processed_memory
    #    - mask

    # Outputs are the following
    #    - mel_outputs
    #    - mel_lengths

    # Note: alignments and gate_outputs are scan outputs, but don't seem to be used later in the PyTorch implementation. For now, we will make them intermediate tensors that are not outputted

    graph = gs.Graph()

    decoder_input_0 = gs.Variable("decoder_input_0", dtype=float_prec, shape=('batch_size', 80))
    attention_hidden_0 = gs.Variable("attention_hidden_0", dtype=float_prec, shape=('batch_size', 1024))
    attention_cell_0 = gs.Variable("attention_cell_0", dtype=float_prec, shape=('batch_size', 1024))
    decoder_hidden_0 = gs.Variable("decoder_hidden_0", dtype=float_prec, shape=('batch_size', 1024))
    decoder_cell_0 = gs.Variable("decoder_cell_0", dtype=float_prec, shape=('batch_size', 1024))
    attention_weights_0 = gs.Variable("attention_weights_0", dtype=float_prec, shape=('batch_size', 'seq_len'))
    attention_weights_cum_0 = gs.Variable("attention_weights_cum_0", dtype=float_prec, shape=('batch_size', 'seq_len'))
    attention_context_0 = gs.Variable("attention_context_0", dtype=float_prec, shape=('batch_size', 512))
    not_finished_0 = gs.Variable("not_finished_0", dtype=bool)
    mel_lengths_0 = gs.Variable("mel_lengths_0", dtype=np.int32)

    # For not_finished, we need to generate a tensor of shape (batch_size) that is all 1s
    # We can use the ONNX ConstantOfShape op to do this
    not_finished_shape = gs.Variable("not_finished_shape", dtype=np.int64)
    reduced = gs.Variable("reduced", dtype=float_prec)
    graph.nodes.append(gs.Node(op="ReduceSum", inputs=[decoder_input_0], outputs=[reduced], attrs={"axes":[1], "keepdims": 1}))
    graph.nodes.append(gs.Node(op="Shape", inputs=[reduced], outputs=[not_finished_shape]))
    before_cast = gs.Variable("before_cast", dtype=np.int32)
    graph.nodes.append(gs.Node(op="ConstantOfShape", inputs=[not_finished_shape], outputs=[before_cast], attrs={"value":gs.Constant("one", np.array([1], dtype=np.int32))}))
    graph.nodes.append(gs.Node(op="Cast", inputs=[before_cast], outputs=[not_finished_0], attrs={"to": 9}))

    # Same thing for mel_lengths, but we need all 0s
    graph.nodes.append(gs.Node(op="ConstantOfShape", inputs=[not_finished_shape], outputs=[mel_lengths_0], attrs={"value":gs.Constant("zero", np.array([0], dtype=np.int32))}))

    # Loop carried dependecies at the end of the loop
    decoder_input_t = gs.Variable("decoder_input_t", dtype=float_prec, shape=('batch_size', 80))
    attention_hidden_t = gs.Variable("attention_hidden_t", dtype=float_prec, shape=('batch_size', 1024))
    attention_cell_t = gs.Variable("attention_cell_t", dtype=float_prec, shape=('batch_size', 1024))
    decoder_hidden_t = gs.Variable("decoder_hidden_t", dtype=float_prec, shape=('batch_size', 1024))
    decoder_cell_t = gs.Variable("decoder_cell_t", dtype=float_prec, shape=('batch_size', 1024))
    attention_weights_t = gs.Variable("attention_weights_t", dtype=float_prec, shape=('batch_size', 'seq_len'))
    attention_weights_cum_t = gs.Variable("attention_weights_cum_t", dtype=float_prec, shape=('batch_size', 'seq_len'))
    attention_context_t = gs.Variable("attention_context_t", dtype=float_prec, shape=('batch_size', 512))
    not_finished_t = gs.Variable("not_finished_t", dtype=bool)
    mel_lengths_t = gs.Variable("mel_lengths_t", dtype=np.int32, shape=('batch_size', 1))

    # Scan outputs
    mel_outputs_raw = gs.Variable("mel_outputs_raw", dtype=float_prec, shape=(-1, 'batch_size', 80))
    gate_outputs = gs.Variable("gate_outputs", dtype=float_prec, shape=(-1, 'batch_size', 1))
    alignments = gs.Variable("alignments", dtype=float_prec, shape=(-1, 1, 'seq_len'))

    mel_outputs = gs.Variable("mel_outputs", dtype=float_prec, shape=('batch_size', 80, -1))

    graph.inputs = [decoder_input_0, attention_hidden_0, attention_cell_0, decoder_hidden_0, decoder_cell_0, attention_weights_0, attention_weights_cum_0, attention_context_0, memory, processed_memory, mask]
    graph.outputs = [mel_outputs, mel_lengths_t]

    trip_count = gs.Constant("trip_count", np.array(0, dtype=np.int64)) # In ONNX, this is an optional parameter, but I don't think ONNX-GS supports optional inputs. To fix this, after we export the ONNX ModelProto from GS, we replace this input with ""
    initial_cond = gs.Constant("initial_cond", np.array(True, dtype=bool))
    loop_inputs = [trip_count, initial_cond, decoder_input_0, attention_hidden_0, attention_cell_0, decoder_hidden_0, decoder_cell_0, attention_weights_0, attention_weights_cum_0, attention_context_0, not_finished_0, mel_lengths_0]
    loop_outputs = [decoder_input_t, attention_hidden_t, attention_cell_t, decoder_hidden_t, decoder_cell_t, attention_weights_t, attention_weights_cum_t, attention_context_t, not_finished_t, mel_lengths_t, mel_outputs_raw, gate_outputs, alignments]
    decoder_loop = gs.Node(op="Loop", name="decoder_loop", inputs=loop_inputs, outputs=loop_outputs, attrs={"body": loop_body})
    graph.nodes.append(decoder_loop)

    graph.nodes.append(gs.Node(op="Transpose", inputs=[mel_outputs_raw], outputs=[mel_outputs], attrs={"perm": [1, 2, 0]})) # Output needs to have loop dimension as inner-most dim

    graph.toposort()
    exported_graph = gs.export_onnx(graph)
    [x for x in exported_graph.graph.node if x.name == "decoder_loop"][0].input[0] = "" # Remove trip count input

    onnx.save(exported_graph, os.path.join(output_dir, decoder_out_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='path to original decoder_iter ONNX model')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--decoder_out', type=str, help='Filename of the exported decoder with outer loop')
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()

    if args.decoder_out == None:
        args.decoder_out = "decoder_with_outer_loop_{}.onnx".format("fp16" if args.fp16 else "fp32")

    insert_decoder_loop(args.model_path, args.output_dir, args.decoder_out, args.fp16)