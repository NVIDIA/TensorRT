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

import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import onnx
import pycuda.autoinit

# TensorRT
import tensorrt as trt
from builder_utils import load_tf_weights, load_pytorch_weights_and_quant, load_onnx_weights_and_quant, load_megatron_pickle_weights
from builder_utils import WQKV, BQKV  # Attention Keys
from builder_utils import W_AOUT, B_AOUT, W_MID, B_MID, W_LOUT, B_LOUT  # Transformer Keys
from builder_utils import SQD_W, SQD_B  # SQuAD Output Keys

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_version = [int(n) for n in trt.__version__.split('.')]

# Import necessary plugins for demoBERT
plugin_lib_name = "nvinfer_plugin.dll" if sys.platform == "win32" else "libnvinfer_plugin.so"
env_name_to_add_path = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `{}` on your {}?".format(plugin_lib_name, env_name_to_add_path))

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
emln_plg_creator2 = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "2", "")
mha_plg_creator2 = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "2", "")
skln_plg_creator2 = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "2", "")

mha_plg_creator3 = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "3", "")
skln_plg_creator3 = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "3", "")

# Megatron Plugins
emln_plg_creator3 = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "3", "")
skln_plg_creator4 = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "4", "")

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8, use_qat, interleaved, timing_cache, use_sparsity, use_megatron):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_qat = use_qat
            self.interleaved = interleaved
            self.timing_cache = timing_cache
            self.use_sparsity = use_sparsity
            self.use_megatron = use_megatron

    def get_trt_dtype(self):
        dtype = trt.float32
        if self.use_fp16:
            dtype = trt.float16
        if self.use_int8:
            dtype = trt.int8
        return dtype

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def attention_layer_opt(prefix, config, init_dict, network, input_tensor, mask_idx, cu_seqlens, max_seqlen):
    """
    Add the attention layer
    """
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    # FC_attention
    if config.use_int8:
        mult_all = network.add_convolution_nd(input_tensor, 3 * hidden_size, (1, 1), Wall, Ball)
    else:
        mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)

    if config.use_qat:
        dr_qkv = max(
            init_dict[prefix + 'self_qv_a_input_quantizer_amax'],
            init_dict[prefix + 'self_qv_b_input_quantizer_amax'],
            init_dict[prefix + 'self_av_b_input_quantizer_amax'],
        )
        set_output_range(mult_all, dr_qkv)
    set_output_name(mult_all, prefix, "qkv_mult")

    # QKV2CTX
    dtype = config.get_trt_dtype()

    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
    pf_var_seqlen =  trt.PluginField("var_seqlen", np.array([int(1)], np.int32), trt.PluginFieldType.FLOAT32)

    if config.use_qat:
        dr_probs = init_dict[prefix + 'self_av_a_input_quantizer_amax']
        dq_probs = dr_probs / 127.0
        pf_dq_probs =  trt.PluginField("dq_probs", np.array([dq_probs], np.float32), trt.PluginFieldType.FLOAT32)
        fields = [pf_hidden_size, pf_num_heads, pf_dq_probs]
    else:
        fields = [pf_hidden_size, pf_num_heads]

    if config.use_int8 and config.interleaved:
        pfc = trt.PluginFieldCollection(fields)
        qkv2ctx_plug = mha_plg_creator3.create_plugin("qkv2ctx", pfc)
        qkv_in = [mult_all.get_output(0), cu_seqlens, max_seqlen]
    else:
        fields.append(pf_has_mask)
        fields.append(pf_type)
        fields.append(pf_var_seqlen)
        pfc = trt.PluginFieldCollection(fields)
        qkv2ctx_plug = mha_plg_creator2.create_plugin("qkv2ctx", pfc)
        qkv_in = [mult_all.get_output(0), mask_idx, cu_seqlens, max_seqlen]
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    qkv2ctx.name = prefix + 'qkv_to_ctx'

    if config.use_qat:
        dr_ctx = init_dict[prefix + 'output_dense_input_amax']
        set_output_range(qkv2ctx, dr_ctx)
    set_output_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx

def skipln(prefix, config, init_dict, network, input_tensor, skip, is_last_skipln=False):
    """
    Add the skip layer
    """
    hidden_size = config.hidden_size
    dtype = config.get_trt_dtype()

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)

    if config.use_int8 and config.interleaved:
        pfc = trt.PluginFieldCollection([pf_beta, pf_gamma])
        creator = skln_plg_creator3 if not config.use_megatron or is_last_skipln else skln_plg_creator4
        skipln_plug = creator.create_plugin("skipln", pfc)
    else:
        pfc = trt.PluginFieldCollection([pf_ld, pf_beta, pf_gamma, pf_type])
        skipln_plug = skln_plg_creator2.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer

def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, residual, mask_idx, cu_seqlens, max_seqlen):
    """
    Add the transformer layer
    """
    hidden_size = config.hidden_size

    if config.use_qat:
        dr_input = init_dict[prefix + 'attention_self_query_input_amax']
        assert(dr_input ==init_dict[prefix + 'attention_self_key_input_amax'] )
        assert(dr_input ==init_dict[prefix + 'attention_self_value_input_amax'] )
        input_tensor.set_dynamic_range(-dr_input, dr_input)

    context_transposed = attention_layer_opt(prefix + "attention_", config, init_dict, network, input_tensor, mask_idx, cu_seqlens, max_seqlen)
    attention_heads = context_transposed.get_output(0)

    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    W_aout = init_dict[prefix + W_AOUT]
    if config.use_int8:
        attention_out_fc = network.add_convolution_nd(attention_heads, hidden_size, (1, 1), W_aout, B_aout)
    else:
        attention_out_fc = network.add_fully_connected(attention_heads, hidden_size, W_aout, B_aout)
    if config.use_int8 and config.use_qat:
        dr_fc_aout = init_dict[prefix + 'attention_output_add_local_input_quantizer_amax']
        set_output_range(attention_out_fc, dr_fc_aout)

    if config.use_megatron:
        dr_skln1_res_in = init_dict[prefix + "attention_output_add_residual_input_quantizer_amax"]
        residual.set_dynamic_range(-dr_skln1_res_in, dr_skln1_res_in)
        skip = residual
    else:
        skip = input_tensor
    skiplayer = skipln(prefix + "attention_output_layernorm_", config, init_dict, network, attention_out_fc.get_output(0), skip)
    attention_ln = skiplayer.get_output(0)
    if config.use_qat:
        dr_skln1 = init_dict[prefix + 'intermediate_dense_input_amax']
        set_output_range(skiplayer, dr_skln1)

    # FC1 + GELU
    B_mid = init_dict[prefix + B_MID]
    W_mid = init_dict[prefix + W_MID]
    if config.use_int8:
        mid_dense = network.add_convolution_nd(attention_ln, config.intermediate_size, (1, 1), W_mid, B_mid)
    else:
        mid_dense = network.add_fully_connected(attention_ln, config.intermediate_size, W_mid, B_mid)

    gelu_layer = add_gelu(network, mid_dense.get_output(0))

    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")
    if config.use_int8:
        if config.use_qat:
            dr_gelu = init_dict[prefix + 'output_dense_input_amax']
            set_output_range(gelu_layer, dr_gelu)
        else:
            # use gelu10 according to whitepaper http://arxiv.org/abs/2004.09602
            set_output_range(gelu_layer, 10)

    # FC2
    # Dense to hidden size
    B_lout = init_dict[prefix + B_LOUT]
    W_lout = init_dict[prefix + W_LOUT]

    if config.use_int8:
        out_dense = network.add_convolution_nd(intermediate_act, hidden_size, (1, 1), W_lout, B_lout)
    else:
        out_dense = network.add_fully_connected(intermediate_act, hidden_size, W_lout, B_lout)
    if config.use_int8 and config.use_qat:
        dr_fc_out = init_dict[prefix + 'output_add_local_input_quantizer_amax']
        set_output_range(out_dense, dr_fc_out)
    set_output_name(out_dense, prefix + "output_", "dense")

    if config.use_megatron:
        dr_skln2_res_in = init_dict[prefix + 'output_add_residual_input_quantizer_amax']
        set_output_range(skiplayer, dr_skln2_res_in, out_idx=1)
        skip = skiplayer.get_output(1)
    else:
        skip = attention_ln

    is_last_skipln = prefix.startswith('l{}'.format(config.num_hidden_layers-1))
    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), skip, is_last_skipln)
    set_output_name(out_layer, prefix + "output_", "reshape")

    return out_layer

def add_gelu(network, input_tensor):
    """
    Adds elementwise GELU, and will trigger FC+GELU fusion in TRT
    """
    shape = (1, ) * len(input_tensor.shape)
    POW = network.add_constant(shape, trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant(shape, trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant(shape, trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(input_tensor, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(input_tensor, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), input_tensor, trt.ElementWiseOperation.PROD)

    # enable elementwise fusing for int8 && fp16
    POW.precision = trt.DataType.FLOAT
    MULTIPLY.precision = trt.DataType.FLOAT
    SQRT.precision = trt.DataType.FLOAT
    ONE.precision = trt.DataType.FLOAT
    HALF.precision = trt.DataType.FLOAT
    X_pow.precision = trt.DataType.FLOAT
    X_mul.precision = trt.DataType.FLOAT
    X_add.precision = trt.DataType.FLOAT
    X_sqrt.precision = trt.DataType.FLOAT
    X_tanh.precision = trt.DataType.FLOAT
    X_one.precision = trt.DataType.FLOAT
    CDF.precision = trt.DataType.FLOAT
    gelu_layer.precision = trt.DataType.FLOAT
    return gelu_layer


def bert_model(config, init_dict, network, input_tensor, residual, mask_idx, cu_seqlens, max_seqlen):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        out_layer = transformer_layer_opt(ss, config, init_dict, network, prev_input, residual, mask_idx, cu_seqlens, max_seqlen)
        prev_input = out_layer.get_output(0)
        # Skip reading residual from final layer
        if config.use_megatron and (layer != config.num_hidden_layers - 1):
            residual = out_layer.get_output(1)

    if config.use_qat:
        dr_out = init_dict["bert_encoder_final_input_quantizer_amax"]
        set_output_range(out_layer, dr_out)

    squad_logits = squad_output("cls_", config, init_dict, network, prev_input)
    squad_logits_out = squad_logits.get_output(0)
    network.mark_output(squad_logits_out)


def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """
    hidden_size = config.hidden_size

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    if config.use_int8:
        dense = network.add_convolution_nd(input_tensor, 2, (1, 1), W_out, B_out)
    else:
        dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)

    OUT = network.add_shuffle(dense.get_output(0))
    if config.use_int8 and config.interleaved:
        OUT.second_transpose = (1, 2, 0, 3)
    else:
        OUT.second_transpose = (1, 0, 2, 3)
    set_output_name(OUT, prefix, "squad_logits")
    return OUT

def emb_layernorm(builder, network, config, weights_dict, builder_config, max_sequence_length, batch_sizes):
    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1,))
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1,))
    cu_seqlens = network.add_input(name="cu_seqlens", dtype=trt.int32, shape=(-1,))
    max_seqlen = network.add_input(name="max_seqlen", dtype=trt.int32, shape=(-1,))

    for batch_size in batch_sizes:
        # Specify profiles
        profile = builder.create_optimization_profile()
        min_shape = (1,)
        shape = (max_sequence_length*batch_size,)
        profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
        profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
        profile.set_shape("cu_seqlens", min=min_shape, opt=(batch_size+1,), max=(batch_size+1,))
        profile.set_shape("max_seqlen", min=min_shape, opt=(max_sequence_length,), max=(max_sequence_length,))
        builder_config.add_optimization_profile(profile)

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"].numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"].numpy(), trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    output_fp16 = trt.PluginField("output_fp16", np.array([1 if config.use_fp16 or config.use_int8 else 0]).astype(np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
    fn = (emln_plg_creator3 if config.use_megatron else emln_plg_creator2).create_plugin("embeddings", pfc)

    inputs = [input_ids, segment_ids, cu_seqlens, max_seqlen]
    emb_layer = network.add_plugin_v2(inputs, fn)

    if config.use_int8 and config.use_qat:
        dr_input = weights_dict['l0_attention_self_query_input_amax']
        set_output_range(emb_layer, dr_input, out_idx=0)
        
        if config.use_megatron:
            dr_skln1_res_in = weights_dict['l0_attention_output_add_residual_input_quantizer_amax']
            set_output_range(emb_layer, dr_skln1_res_in, out_idx=1)
    
    set_output_name(emb_layer, "embeddings_", "output")
    return emb_layer, cu_seqlens, max_seqlen

def build_engine(batch_sizes, workspace_size, sequence_length, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num, verbose):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = workspace_size * (1024 * 1024)
        builder_config.avg_timing_iterations = 8
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.use_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            if not config.use_qat:
                raise RuntimeError("Post training calibration is not supported in variable-length BERT.")

        if verbose:
            builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # speed up the engine build for trt major version >= 8 
        # 1. disable cudnn tactic
        # 2. load global timing cache
        if trt_version[0] >= 8:
            tactic_source = builder_config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
            builder_config.set_tactic_sources(tactic_source)
            if config.timing_cache != None:
                if os.path.exists(config.timing_cache):
                    with open(config.timing_cache, "rb") as f:
                        cache = builder_config.create_timing_cache(f.read())
                        builder_config.set_timing_cache(cache, ignore_mismatch = False)
                else:
                    cache = builder_config.create_timing_cache(b"")
                    builder_config.set_timing_cache(cache, ignore_mismatch = False)

        if config.use_sparsity:
            TRT_LOGGER.log(TRT_LOGGER.INFO, "Setting sparsity flag on builder_config.")
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # Create the network
        emb_layer, cu_seqlens, max_seqlen = emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_length, batch_sizes)
        embeddings = emb_layer.get_output(0)
        if config.use_int8 and config.interleaved:
            shuffle = network.add_shuffle(embeddings)
            shuffle.second_transpose = (2, 1, 0, 3)
            embeddings = shuffle.get_output(0)
            mask_idx = None
        else:
            mask_idx = emb_layer.get_output(1)
            
        if config.use_megatron:  # megatron currently only supports int8 and interleaved
            shuffler = network.add_shuffle(emb_layer.get_output(1))
            shuffler.second_transpose = (2, 1, 0, 3)
            residual = shuffler.get_output(0)

            dr_emb = weights_dict['l0_attention_self_query_input_amax']
            embeddings.set_dynamic_range(-dr_emb, dr_emb)
            dr_skln1_res_in = weights_dict['l0_attention_output_add_residual_input_quantizer_amax']
            residual.set_dynamic_range(-dr_skln1_res_in, dr_skln1_res_in)
        else:
            residual = None

        bert_model(config, weights_dict, network, embeddings, residual, mask_idx, cu_seqlens, max_seqlen)

        build_start_time = time.time()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))

        # save global timing cache
        if trt_version[0] >= 8 and config.timing_cache != None:
            cache = builder_config.get_timing_cache()
            with cache.serialize() as buffer:
                with open(config.timing_cache, "wb") as f:
                    f.write(buffer)
                    f.flush()
                    os.fsync(f)

        return engine

def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--ckpt", required=False,
                        help="The checkpoint file basename, e.g.: basename(model.ckpt-766908.data-00000-of-00001) is model.ckpt-766908")
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-pt", "--pytorch", required=False, help="The PyTorch checkpoint file path.")
    parser.add_argument("-pkl", "--pickle", required=False, help="The Pickle weights dictionary file path for the Megatron variant of BERT.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--max-batch-size", default=[], action="append", help="Max batch size. The engine will be usable with any input with (batch-size * sequence-length) below (max-batch-size * max-sequence-length). Can be specified multiple times to build optimization profiles for more than one batch size.", type=int)
    parser.add_argument("-s", "--max-sequence-length", default=128, help="Max sequence length of the BERT model. The engine will be usable with any input with (batch-size * sequence-length) below (max-batch-size * max-sequence-length).", type=int)
    parser.add_argument("-c", "--config-dir", required=True,
                        help="The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-w", "--workspace-size", default=1200, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-j", "--squad-json", default="squad/dev-v1.1.json", help="squad json dataset used for int8 calibration", required=False)
    parser.add_argument("-v", "--vocab-file", default="./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt", help="Path to file containing entire understandable vocab", required=False)
    parser.add_argument("-n", "--calib-num", default=100, help="calibration batch numbers", type=int)
    parser.add_argument("-p", "--calib-path", help="calibration cache path", required=False)
    parser.add_argument("-il", "--interleaved", action="store_true", help="use interleaved format, only valid in INT8 precision", required=False)
    parser.add_argument("-tcf", "--timing-cache-file", help="Path to tensorrt build timeing cache file, only available for tensorrt 8.0 and later", required=False)
    parser.add_argument("-sp", "--sparse", action="store_true", help="Indicates that model is sparse", required=False)
    parser.add_argument("--megatron", action="store_true", help="Indicates that model is the Megatron-style architecture", required=False)
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose logger and set profiling verbosity to verbose", required=False)

    args, _ = parser.parse_known_args()
    args.max_batch_size = args.max_batch_size or [1]

    if args.verbose:
        TRT_LOGGER.min_severity = TRT_LOGGER.VERBOSE

    cc = pycuda.autoinit.device.compute_capability()
    if cc[0] * 10 + cc[1] < 72:
        raise RuntimeError("This variable-length BERT demo only support Xavier+ GPU.")
        
    if args.megatron: 
        if not (args.interleaved and args.int8):
            raise RuntimeError("Megatron BERT currently only supports int8 and interleaved.")
        if not args.pickle:
            raise RuntimeError("Megatron BERT currently only supports loading a pickle weights dictionary.")

    bert_config_path = os.path.join(args.config_dir, "bert_config.json")
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))

    config = BertConfig(bert_config_path, args.fp16, args.int8, args.int8 and (args.onnx or args.pytorch or args.pickle), args.interleaved, args.timing_cache_file, args.sparse, args.megatron)

    if args.calib_path != None:
        calib_cache = args.calib_path
    else:
        calib_cache = "BertSquadL{}H{}A{}S{}CalibCache".format(config.num_hidden_layers, config.head_size, config.num_attention_heads, args.max_sequence_length)

    if args.onnx != None:
        weights_dict = load_onnx_weights_and_quant(args.onnx, config)
    elif args.pytorch != None:
        weights_dict = load_pytorch_weights_and_quant(args.pytorch, config)
    elif args.ckpt != None:
        weights_dict = load_tf_weights(args.ckpt, config)
    elif args.pickle != None:
        weights_dict =  load_megatron_pickle_weights(args.pickle, config)
    else:
        raise RuntimeError("You need either specify TF checkpoint using option --ckpt, ONNX using option --onnx, "
                           "PyTorch using option --pytorch, or Pickle weight dictionary using option --pickle "
                           "to build TRT BERT model.")

    with build_engine(args.max_batch_size, args.workspace_size, args.max_sequence_length, config, weights_dict, args.squad_json, args.vocab_file, calib_cache, args.calib_num, args.verbose) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

if __name__ == "__main__":
    main()
