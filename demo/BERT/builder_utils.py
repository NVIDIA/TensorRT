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

import re 
import pickle

import numpy as np
import onnx
import tensorrt as trt
import torch

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError as err:
    import sys
    sys.stderr.write("""Error: Failed to import tensorflow module ({})\n""".format(err))
    sys.exit()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

"""
Attentions Keys
"""
WQ = "self_query_kernel"
BQ = "self_query_bias"
WK = "self_key_kernel"
BK = "self_key_bias"
WV = "self_value_kernel"
BV = "self_value_bias"
WQKV = "self_qkv_kernel"
BQKV = "self_qkv_bias"

"""
Transformer Keys
"""
W_AOUT = "attention_output_dense_kernel"
B_AOUT = "attention_output_dense_bias"
AOUT_LN_BETA = "attention_output_layernorm_beta"
AOUT_LN_GAMMA = "attention_output_layernorm_gamma"
W_MID = "intermediate_dense_kernel"
B_MID = "intermediate_dense_bias"
W_LOUT = "output_dense_kernel"
B_LOUT = "output_dense_bias"
LOUT_LN_BETA = "output_layernorm_beta"
LOUT_LN_GAMMA = "output_layernorm_gamma"

"""
Squad Output Keys
"""
SQD_W = "squad_output_weights"
SQD_B = "squad_output_bias"


def load_tf_weights(inputbase, config):
    """
    Load the weights from the tensorflow checkpoint
    """
    weights_dict = dict()

    try:
        reader = tf.train.NewCheckpointReader(inputbase)
        tensor_dict = reader.get_variable_to_shape_map()

        # There might be training-related variables in the checkpoint that can be discarded
        param_names = [key for key in sorted(tensor_dict) if "adam" not in key and "global_step" not in key and "pooler" not in key]
        count = len(param_names)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(count))

        for pn in param_names:
            toks = pn.lower().split("/")
            if "encoder" in pn:
                assert ("layer" in pn)
                l = (re.findall("\d+", pn))[0]
                outname = "l{}_".format(l) + "_".join(toks[3:])
            else:
                outname = "_".join(toks)

            tensor = reader.get_tensor(pn)
            shape = tensor.shape
            if pn.find("kernel") != -1:
                weights_dict[outname + "_notrans"] = trt.Weights(np.ascontiguousarray(tensor).flatten())

                TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Transposing {}\n".format(np))
                tensor = np.transpose(tensor)

            shape = tensor.shape
            flat_tensor = tensor.flatten()
            shape_str = "{} ".format(len(shape)) + " ".join([str(d) for d in shape])
            weights_dict[outname] = trt.Weights(flat_tensor)

            TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Original name: {:}, TensorRT name: {:}, shape: {:}".format(pn, outname, shape_str))

        N = config.num_attention_heads
        H = config.head_size

        additional_dict = dict()
        for key, value in weights_dict.items():
            pos = key.find(BQ)
            if pos != -1:
                hidden_size = value.size
                prefix = key[:pos]

                Bq_ = value
                Bk_ = weights_dict[prefix + BK]
                Bv_ = weights_dict[prefix + BV]
                Wq_ = weights_dict[prefix + WQ]
                Wk_ = weights_dict[prefix + WK]
                Wv_ = weights_dict[prefix + WV]

                mat_size = hidden_size * hidden_size
                wcount = 3 * mat_size
                Wall = np.zeros(wcount, np.float32)
                bcount = 3 * hidden_size
                Ball = np.zeros(bcount, np.float32)
                Wall[0:mat_size] = Wq_.numpy()[0:mat_size]
                Wall[mat_size:2*mat_size] = Wk_.numpy()[0:mat_size]
                Wall[2*mat_size:3*mat_size] = Wv_.numpy()[0:mat_size]
                Ball[0:hidden_size] = Bq_.numpy()[0:hidden_size]
                Ball[hidden_size:2*hidden_size] = Bk_.numpy()[0:hidden_size]
                Ball[2*hidden_size:3*hidden_size] = Bv_.numpy()[0:hidden_size]

                if config.use_int8 and getattr(config, 'interleaved', False):
                    Wall = np.ascontiguousarray(Wall.reshape((3, N, H, N, H)), dtype=np.float32)
                    Ball = np.ascontiguousarray(Ball.reshape((3, N, H)), dtype=np.float32)
                else:
                    Wall = np.ascontiguousarray(Wall.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)), dtype=np.float32)
                    Ball = np.ascontiguousarray(Ball.reshape((3, N, H)).transpose((1, 0, 2)), dtype=np.float32)

                additional_dict[prefix + WQKV] = trt.Weights(Wall)
                additional_dict[prefix + BQKV] = trt.Weights(Ball)
                additional_dict[prefix + WQKV + "_notrans"] = trt.Weights(np.ascontiguousarray(Wall.T))

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    weights_dict.update(additional_dict)
    return weights_dict

def onnx_to_trt_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    qkv_strings = {'key', 'value', 'query', 'query_key_value'}
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    if toks[0] == 'bert': #embeddings or encoder
        if toks[1] == 'encoder': #transformer
            # Token conversions for sparse checkpoints
            if toks[-2] == 'dense_act': 
                toks[-2] = 'dense'
            elif toks[-3] == 'dense_act':
                if toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'
                elif toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                toks[-3] = 'dense'
            elif toks[-2].startswith('matmul'):
                toks[-2] = {
                    'matmul_q_quantizer': 'qv_a_input_quantizer',
                    'matmul_k_quantizer': 'qv_b_input_quantizer',
                    'matmul_v_quantizer': 'av_b_input_quantizer',
                    'matmul_a_quantizer': 'av_a_input_quantizer',
                }[toks[-2].replace('input_', '')]

            # Token conversions for all checkpoints
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in qkv_strings) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in qkv_strings) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'
            
            if 'final_input_quantizer' not in toks[2]:
                ind = toks.index('layers')+1 if 'layers' in toks else 3
                toks = toks[ind:]
                toks[0] = 'l{}'.format(int(toks[0]))
        else:
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else: #embeddings: drop "_weight" suffix
                if toks[-1] == 'amax':
                    toks[-2] = 'amax'
                toks = toks[:-1]
    elif 'qa' in onnx_name:
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        print("Encountered unknown case:", onnx_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed

def get_onnx_weight_dict(tensor_dict, config):
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    weights_dict = dict()
    for outname, tensor in tensor_dict.items():
        if outname.find("_amax") != -1:
            weights_dict[outname] = tensor
        elif outname.find(BQ) != -1:
            prefix = outname[:outname.find(BQ)]

            Wqkv = np.zeros((3, hidden_size, hidden_size), np.float32)
            Bqkv = np.zeros((3, hidden_size), np.float32)

            Wqkv[0,:,:] = tensor_dict[prefix + WQ]
            Wqkv[1,:,:] = tensor_dict[prefix + WK]
            Wqkv[2,:,:] = tensor_dict[prefix + WV]
            Bqkv[0,:] = tensor
            Bqkv[1,:] = tensor_dict[prefix + BK]
            Bqkv[2,:] = tensor_dict[prefix + BV]
    
            if config.use_int8 and getattr(config, 'interleaved', False):
                Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)))
                Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)))
            else:
                Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1,0,2,3,4)))
                Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1,0,2)))
    
            weights_dict[prefix + WQKV] = trt.Weights(Wqkv)
            weights_dict[prefix + BQKV] = trt.Weights(Bqkv)
            weights_dict[prefix + WQKV + "_notrans"] = trt.Weights(np.ascontiguousarray(Wqkv.T))

        elif outname.find(BK) != -1 or outname.find(BV) != -1 or outname.find(WQ) != -1 or outname.find(WK) != -1 or outname.find(WV) != -1:
            pass
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten()
            weights_dict[outname] = trt.Weights(flat_tensor)

            if outname.find("kernel") != -1:
                tensor = np.transpose(tensor)
                weights_dict[outname + "_notrans"] = trt.Weights(np.ascontiguousarray(tensor).flatten())

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(weights_dict)))
    return weights_dict

def load_onnx_weights_and_quant(path, config):
    """
    Load the weights from the onnx checkpoint
    """
    model = onnx.load(path)
    weights = model.graph.initializer
    tensor_dict = dict((onnx_to_trt_name(w.name), np.frombuffer(w.raw_data, np.int8).reshape(w.dims))
                       if w.name.split('_')[-1] == 'mask' else 
                       (onnx_to_trt_name(w.name), np.frombuffer(w.raw_data, np.float32).reshape(w.dims))
                       for w in weights)
    return get_onnx_weight_dict(tensor_dict, config)

def load_pytorch_weights_and_quant(path, config):
    """
    Load the weights from the pytorch checkpoint
    """
    state_dict = torch.load(path, map_location='cpu')["model"]
    tensor_dict = {onnx_to_trt_name(name):val.numpy() for name, val in state_dict.items()}
    return get_onnx_weight_dict(tensor_dict, config)

def load_megatron_pickle_weights(path, config):
    N = config.num_attention_heads
    H = config.head_size

    with open(path, 'rb') as f:
        tensor_dict = pickle.load(f)

    weight_dict = {}
    for name, tensor in tensor_dict.items():
        if 'scale' in name:
            continue

        name = (onnx_to_trt_name(name)
                .replace('embedding_', 'embeddings_')
                .replace('tokentype_', 'token_type_')
                .replace('_av', '_self_av')
                .replace('_qv', '_self_qv')
                .replace('query_key_value', 'self_qkv'))

        if name.endswith('self_qkv_kernel'):
            tensor = np.ascontiguousarray(tensor.reshape((3, N, H, N, H))).astype(np.float32)
            weight_dict[name] = trt.Weights(tensor)
        elif name.endswith('self_qkv_bias'):
            tensor = np.ascontiguousarray(tensor.reshape((3, N, H))).astype(np.float32)
            weight_dict[name] = trt.Weights(tensor)
        elif name == 'l{}_output_layernorm_output_quantizer_amax'.format(config.num_hidden_layers-1):
            weight_dict['bert_encoder_final_input_quantizer_amax'] = tensor
        elif name.endswith('_amax'):
            weight_dict[name] = tensor
            if name.endswith('_qkv_input_amax'):
                weight_dict[name.replace('_qkv_input_amax', '_query_input_amax')] = tensor
                weight_dict[name.replace('_qkv_input_amax', '_key_input_amax')] = tensor
                weight_dict[name.replace('_qkv_input_amax', '_value_input_amax')] = tensor
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten().astype(np.float32)
            weight_dict[name] = trt.Weights(flat_tensor)

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(weight_dict)))
    return weight_dict
