#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorrt as trt
import ctypes
import argparse
import numpy as np
import json
import sys
import re
import os

try:
    from tensorflow.python import pywrap_tensorflow as pyTF
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})""".format(err))
    sys.exit()

nvinfer = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
cm = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode=ctypes.RTLD_GLOBAL)
pg = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)


"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPlugin", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPlugin", "1", "")
gelu_plg_creator = plg_registry.get_plugin_creator("CustomGeluPlugin", "1", "")
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPlugin", "1", "")


"""
Attentions Keys
"""
WQ = "query_kernel"
BQ = "query_bias"
WK = "key_kernel"
BK = "key_bias"
WV = "value_kernel"
BV = "value_bias"
WQKV = "qkv_kernel"
BQKV = "qkv_bias"


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


class BertConfig:
    def __init__(self, bert_config_path):
        with open(bert_config_path, 'r') as f:
            data = json.load(f)
            self.num_attention_heads = data['num_attention_heads']
            self.hidden_size = data['hidden_size']
            self.intermediate_size = data['intermediate_size']
            self.num_hidden_layers = data['num_hidden_layers']
            self.use_fp16 = True


def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_layer_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    assert(len(input_tensor.shape) == 4)
    S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)
    set_layer_name(mult_all, prefix, "qkv_mult")

    has_mask = imask != None

    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_S = trt.PluginField("S", np.array([S], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_S, pf_has_mask])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0), imask]
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    set_layer_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx


def skipln(prefix, init_dict, network, input_tensor, skip):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    assert len(idims) == 4
    hidden_size = idims[1]

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection([pf_ld, pf_beta, pf_gamma])
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer


def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 4
    hidden_size = idims[1]

    context_transposed = attention_layer_opt(prefix + "attention_self_", config, init_dict, network, input_tensor, imask)
    attention_heads = context_transposed.get_output(0)

    W_aout = init_dict[prefix + W_AOUT]
    B_aout = init_dict[prefix + B_AOUT]
    attention_out_fc = network.add_fully_connected(attention_heads, hidden_size, W_aout, B_aout)

    skiplayer = skipln(prefix + "attention_output_layernorm_", init_dict, network, attention_out_fc.get_output(0), input_tensor)
    attention_ln = skiplayer.get_output(0)

    W_mid = init_dict[prefix + W_MID]
    B_mid = init_dict[prefix + B_MID]
    mid_dense = network.add_fully_connected(attention_ln, config.intermediate_size, W_mid, B_mid)

    mid_dense_out = mid_dense.get_output(0)

    pfc = trt.PluginFieldCollection()
    plug = gelu_plg_creator.create_plugin("gelu", pfc)

    gelu_layer = network.add_plugin_v2([mid_dense_out], plug)

    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")

    # Dense to hidden size
    W_lout = init_dict[prefix + W_LOUT]
    B_lout = init_dict[prefix + B_LOUT]

    out_dense = network.add_fully_connected(intermediate_act, hidden_size, W_lout, B_lout)
    set_layer_name(out_dense, prefix + "output_", "dense")
    out_layer = skipln(prefix + "output_layernorm_", init_dict, network, out_dense.get_output(0), attention_ln)
    out_ln = out_layer.get_output(0)

    set_tensor_name(out_ln, prefix + "output_", "reshape")

    return out_ln


def bert_model(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        prev_input = transformer_layer_opt(ss, config,  init_dict, network, prev_input, input_mask)
    return prev_input


def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 4
    S, hidden_size, _, _ = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    W = network.add_constant((1, hidden_size, 2), W_out)
    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)
    set_layer_name(dense, prefix, "dense")
    return dense


def load_weights(inputbase):
    """
    Load the weights from the tensorflow checkpoint
    """
    weights_dict = dict()

    try:
        reader = pyTF.NewCheckpointReader(inputbase)
        tensor_dict = reader.get_variable_to_shape_map()

        # There might be training-related variables in the checkpoint that can be discarded
        param_names = [key for key in sorted(tensor_dict) if 'adam' not in key and 'global_step' not in key and 'pooler' not in key]
        count = len(param_names)
        TRT_LOGGER.log(TRT_LOGGER.INFO, str(count))

        for pn in param_names:
            toks = pn.lower().split('/')
            if 'encoder' in pn:
                assert ('layer' in pn)
                l = (re.findall('\d+', pn))[0]
                outname = 'l{}_'.format(l) + '_'.join(toks[3:])
            else:
                outname = '_'.join(toks)

            tensor = reader.get_tensor(pn)
            shape = tensor.shape
            if pn.find('kernel') != -1:
                TRT_LOGGER.log(TRT_LOGGER.INFO, "Transposing {}\n".format(np))
                tensor = np.transpose(tensor)

            shape = tensor.shape
            flat_tensor = tensor.flatten()
            shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
            weights_dict[outname] = trt.Weights(flat_tensor)

            TRT_LOGGER.log(TRT_LOGGER.INFO, "Orig.name: {:}, TRT name: {:}, shape: {:}".format(pn, outname, shape_str))

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

                additional_dict[prefix + WQKV] = trt.Weights(Wall)
                additional_dict[prefix + BQKV] = trt.Weights(Ball)

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    weights_dict.update(additional_dict)
    return weights_dict


def main(inputbase, B, S, bert_path, outputbase):
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    TRT_LOGGER.log(TRT_LOGGER.INFO, bert_config_path)
    config = BertConfig(bert_config_path)

    # Load weights from checkpoint file
    init_dict = load_weights(inputbase)

    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_batch_size = B
        builder.max_workspace_size = 5000 * (1024 * 1024)
        builder.fp16_mode = True
        builder.strict_type_constraints = False

        ty = trt.PluginFieldType.FLOAT32

        w = init_dict["bert_embeddings_layernorm_beta"]
        wbeta = trt.PluginField("bert_embeddings_layernorm_beta", w.numpy(), ty)

        w = init_dict["bert_embeddings_layernorm_gamma"]
        wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", w.numpy(), ty)

        w = init_dict["bert_embeddings_word_embeddings"]
        wwordemb = trt.PluginField("bert_embeddings_word_embeddings", w.numpy(), ty)

        w = init_dict["bert_embeddings_token_type_embeddings"]
        wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", w.numpy(), ty)

        w = init_dict["bert_embeddings_position_embeddings"]
        wposemb = trt.PluginField("bert_embeddings_position_embeddings", w.numpy(), ty)

        pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb])
        fn = emln_plg_creator.create_plugin("embeddings", pfc)

        with builder.create_network() as network:
            input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(S, ))
            segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(S, ))
            input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(S, ))
            inputs = [input_ids, segment_ids, input_mask]
            emb_layer = network.add_plugin_v2(inputs, fn)

            embeddings = emb_layer.get_output(0)
            mask_idx = emb_layer.get_output(1)

            bert_out = bert_model(config, init_dict, network, embeddings, mask_idx)

            squad_logits = squad_output("cls_", config, init_dict, network, bert_out)
            squad_logits_out = squad_logits.get_output(0)
            network.mark_output(squad_logits_out)

            engine = builder.build_cuda_engine(network)

            TRT_LOGGER.log(TRT_LOGGER.INFO, "Serializing the engine....")
            serialized_engine = engine.serialize()
            TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving the engine....")
            with open(outputbase, 'wb') as fout:
                fout.write(serialized_engine)
            TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT BERT Sample')
    parser.add_argument('-m', '--model', required=True,
                        help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001)-> model.ckpt-766908')
    parser.add_argument('-o', '--output', required=True, help='The bert engine file, ex bert.engine')
    parser.add_argument('-b', '--batchsize', required=False, default=1, help='Batch size (default=1)')
    parser.add_argument('-s', '--sequence', required=False, default=384, help='Sequence length of the BERT model (default=384)')
    parser.add_argument('-c', '--config', required=True,
                        help='The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google')

    opt = parser.parse_args()

    inputbase = opt.model
    outputbase = opt.output
    B = int(opt.batchsize)
    S = int(opt.sequence)
    bert_path = opt.config
    main(inputbase, B, S, bert_path, outputbase)
    # Required to work around a double free issue in TRT 5.1
    os._exit(0)
