#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import onnx
import logging
import os
import numpy as np
import onnx_graphsurgeon as gs
from onnx import helper, TensorProto, numpy_helper, version_converter

'''
This script is converting TE ONNX models (cast + CustomOp Q) and (CustomOp DQ + cast) pairs to Opset19 ONNX Q/DQ
usage:
python3 convert_te_onnx_to_trt_onnx.py --onnx_model_path <folder/file> 

This script requires onnx 1.14 and above
'''

def find_node_by_tensor(graph, search_tensor, is_node_input, search_node_type=None):
    for idx, node in enumerate(graph.node):
        search_container = node.output
        if is_node_input:
            search_container = node.input
        for node_tensor in search_container:
            if search_node_type and node.op_type != search_node_type:
                continue
            if node_tensor == search_tensor:
                return node, idx

    return None, None

def redirect_quantize_input(graph, q_node):
    assert(q_node.op_type == 'QuantizeLinear')
    q_input = q_node.input[0]
    cast_node, cast_node_idx = find_node_by_tensor(graph, q_input, False, 'Cast')
    if cast_node:
        q_node.input[0] = cast_node.input[0]
        return [cast_node_idx]
    return []

def redirect_dequantize_output(graph, dq_node):
    assert(dq_node.op_type == 'DequantizeLinear')
    dq_output = dq_node.output[0]
    cast_node, cast_node_idx = find_node_by_tensor(graph, dq_output, True, 'Cast')
    if cast_node:
        dq_node.output[0] = cast_node.output[0]
        return [cast_node_idx]
    return []

def get_attr_numpy_tensor(attr):
    assert(attr.type == onnx.AttributeProto.TENSOR)
    return numpy_helper.to_array(attr.t)

def get_attr(node, search_attr_name):
    for idx, attr in enumerate(node.attribute):
        if attr.name == search_attr_name:
            return attr, idx

    return None, None

def cast_scale(graph, qdq_node, cast_to):
    assert(cast_to in ['fp32', 'fp16'])
    assert(qdq_node.op_type in ['QuantizeLinear', 'DequantizeLinear'])
    constant_node_idx = None
    scale_tensor = qdq_node.input[1]
    constant_node, constant_node_idx = find_node_by_tensor(graph, scale_tensor, False, 'Constant')
    scale_cast_to_dtype = None
    onnx_cast_to_dtype = None
    if cast_to == 'fp16':
        scale_cast_to_dtype = np.dtype(np.float32)
        onnx_cast_to_dtype = onnx.TensorProto.FLOAT16
    elif cast_to == 'fp32':
        scale_cast_to_dtype = np.dtype(np.float32)
        onnx_cast_to_dtype = onnx.TensorProto.FLOAT

    if constant_node:
        scale_attr, _ = get_attr(constant_node, 'value')
        assert(scale_attr)
        numpy_scale = get_attr_numpy_tensor(scale_attr)
        logging.info(type(numpy_scale.dtype))
        logging.info(type(scale_cast_to_dtype))
        if numpy_scale.dtype != scale_cast_to_dtype:
            logging.debug(f'Change {qdq_node.name} scale from {numpy_scale.dtype} to {scale_cast_to_dtype}')
            numpy_scale = numpy_scale.astype(scale_cast_to_dtype)
            tensor_name = constant_node.name + '_casted'
            create_constant_tensor(graph, tensor_name, onnx_cast_to_dtype, numpy_scale)
            qdq_node.input[1] = tensor_name
    else:
        logging.warning(f'No constant node connected to {qdq_node} as scale')

    if constant_node_idx:
        return [constant_node_idx]
    return []

def create_constant_tensor(graph, name, dtype, np_tensor):
    tensor_value_info = helper.make_tensor_value_info(name, dtype, np_tensor.shape)
    graph.input.append(tensor_value_info)
    helper.make_tensor(name, data_type=dtype, dims=(), vals=[0])
    
    tensor_initializer = helper.make_tensor(name, dtype, np_tensor.shape, np_tensor.flatten().tolist())
    graph.initializer.append(tensor_initializer)

'''
Convert custom operators to opset19
'''
def custom_op_to_opset19(graph, node, use_int32_quantization, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision):
    assert(node.op_type in ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear'])
    is_dq = node.op_type == 'TRT_FP8DequantizeLinear'
    logging.debug(f'Convert {node.name} to Opset19')
    orig_node_name = node.name
    new_node_name = orig_node_name + '_converted'

    quant_to = TensorProto.FLOAT8E4M3FN
    if use_int32_quantization:
        quant_to = TensorProto.INT32

    #add zero point to the node
    tensor_name = new_node_name + '_zero_point'
    create_constant_tensor(graph, tensor_name, quant_to, np.array([0]))
    node.input.append(tensor_name)

    node.domain = ""
    node.op_type = "QuantizeLinear"

    node_idxs_to_delete = []
    if is_dq:
        node.op_type = "DequantizeLinear"
        if remove_cast_after_dq:
            node_idxs_to_delete += redirect_dequantize_output(graph, node)
            if change_qdq_scale_precision:
                node_idxs_to_delete += cast_scale(graph, node, change_qdq_scale_precision)
    else:
        if remove_cast_before_q:
            node_idxs_to_delete += redirect_quantize_input(graph, node)
            if change_qdq_scale_precision:
                node_idxs_to_delete += cast_scale(graph, node, change_qdq_scale_precision)

    node.name = new_node_name
    logging.debug(f'Convert Done\n')
    return node_idxs_to_delete

def check_model(graph):
    converted_qdq_ops = ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear']
    passed_check = True
    for node in graph.node:
        if node.op_type in converted_qdq_ops:
            logging.error(f'Node \"{node.name}\" of type {node.op_type} should have been removed')
            passed_check = False
    return passed_check

def update_quantize_node_type(model):
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == "TRT_FP8QuantizeLinear":
            for out in node.outputs:
                out.dtype = TensorProto.FLOAT8E4M3FN
    return gs.export_onnx(graph)

'''
Converts onnx files from TE to TRT
'''
def replace_customop_qdq_with_onnx_qdq(te_onnx_files, results_path, create_netron_compatible_model, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision):
    # store mappings from original ONNX name to new ONNX name.
    file_mappings = {}
    for te_onnx_file in te_onnx_files:
        logging.debug('Loading model')
        model = onnx.load(te_onnx_file, load_external_data=False)
        # update QuantizeLinear output dtype
        model = update_quantize_node_type(model)
        # change model opset to 19
        model.opset_import[0].version = 19
        graph = model.graph
        logging.debug('Loading model finished')
        converted_qdq_ops = ['TRT_FP8QuantizeLinear', 'TRT_FP8DequantizeLinear']

        try:
            node_idxs_to_delete = []
            converted = False
            for node in graph.node:
                if node.op_type in converted_qdq_ops:
                    converted = True
                    node_idxs_to_delete += custom_op_to_opset19(graph, node, create_netron_compatible_model, remove_cast_before_q, remove_cast_after_dq, change_qdq_scale_precision)

            if converted:
                assert(check_model(graph))
                node_idxs_to_delete = reversed(sorted(node_idxs_to_delete))
                for node_idx in node_idxs_to_delete:
                    del(graph.node[node_idx])
                suffix = '.opset19'
                if create_netron_compatible_model:
                    suffix += '.netron'
                suffix += '.onnx'
                new_model_filename = os.path.join(results_path, os.path.splitext(os.path.split(te_onnx_file)[1])[0] + suffix)
                onnx.save_model(model, new_model_filename)
                logging.info(f'The converted model is saved at {new_model_filename}!')
                file_mappings[te_onnx_file] = new_model_filename
            else:
                logging.info(f'No conversion was done with {te_onnx_file}!')
                file_mappings[te_onnx_file] = te_onnx_file
        except Exception as ex:
            logging.error(f'Failed: {ex}')
            file_mappings[te_onnx_file] = None
    return file_mappings

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_path', required=True, help="Path of model or a folder of models. When using a folder, this script will convert all \'.onnx\' files")
    parser.add_argument('--results_path', required=False, help="Path for generated models, when not set, the generated model(s) will be next ot the origianl model(s)")
    parser.add_argument('--create_netron_compatible_model', action='store_true', required=False, help="When set, the script will use int32 quantization. "
        "This enables the user to view the graph with Netron, until it adds support for opset19. The generated model isn't TRT compatible.")
    parser.add_argument('--remove_casts', required=False, help="Controls whether to remove casts around q/dq nodes. "
        "For example, when set to \'dq\', remove casts only after dq. Default is \'keep_all\'", choices=['q', 'dq', 'qdq', 'keep_all'], default='keep_all')
    parser.add_argument('--change_qdq_scale_precision', required=False, help="When set controls q/dq nodes scales data type.", choices=['fp32', 'fp16'])
    args = parser.parse_args()

    results_path = args.results_path
    if results_path and os.path.isdir(results_path) == False:
        logging.error(f'\'--results_path\' set to \'{results_path}\', but the folder doesn\'t exist, exiting')
        exit(-1)

    if results_path is None:
        results_path = args.onnx_model_path
        if os.path.isfile(results_path):
            results_path = os.path.split(results_path)[0]

    remove_cast_after_dq = False
    remove_cast_before_q = False
    if args.remove_casts == 'q':
        remove_cast_before_q = True
    elif args.remove_casts == 'dq':
        remove_cast_after_dq = True
    elif args.remove_casts == 'qdq':
        remove_cast_after_dq = True
        remove_cast_before_q = True

    onnx_files = []
    if os.path.isdir(args.onnx_model_path):
        logging.info(f"Got folder: {args.onnx_model_path}")
        onnx_files = [os.path.join(args.onnx_model_path, filename) for filename in os.listdir(args.onnx_model_path) if filename.endswith('.onnx')==True and filename.endswith('.opset19.onnx')==False]

    else:
        logging.info(f"Got file: {args.onnx_model_path}")
        onnx_files = [args.onnx_model_path]

    replace_customop_qdq_with_onnx_qdq(onnx_files, results_path, args.create_netron_compatible_model, remove_cast_before_q, remove_cast_after_dq, args.change_qdq_scale_precision)
