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
from helpers.calibrator import BertCalibrator as BertCalibrator
from builder_utils import load_tf_weights, load_pytorch_weights_and_quant, load_onnx_weights_and_quant
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
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "1", "")
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "1", "")
fc_plg_creator = plg_registry.get_plugin_creator("CustomFCPluginDynamic", "1", "")

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8, use_strict, use_fc2_gemm, use_int8_skipln, use_int8_multihead, use_qat, use_sparsity, timing_cache):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_fc2_gemm = use_fc2_gemm
            self.use_strict = use_strict
            self.use_int8_skipln = use_int8_skipln
            self.use_int8_multihead = use_int8_multihead
            self.is_calib_mode = False
            self.use_qat = use_qat
            self.use_sparsity = use_sparsity
            self.timing_cache = timing_cache

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    if config.use_int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)

def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    assert(len(input_tensor.shape) == 5)
    B, S, hidden_size, _, _ = input_tensor.shape
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

    has_mask = imask is not None

    # QKV2CTX
    pf_type = trt.PluginField("type_id", np.array([get_mha_dtype(config)], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)
    if config.use_qat:
        dr_probs = init_dict[prefix + 'self_av_a_input_quantizer_amax']
        dq_probs = dr_probs / 127.0
        pf_dq_probs =  trt.PluginField("dq_probs", np.array([dq_probs], np.float32), trt.PluginFieldType.FLOAT32)
        pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_dq_probs])
    else:
        pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0)]
    if has_mask:
        qkv_in.append(imask)
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)

    if config.use_qat:
        dr_ctx = init_dict[prefix + 'output_dense_input_amax']
        set_output_range(qkv2ctx, dr_ctx)
    set_output_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx

def skipln(prefix, config, init_dict, network, input_tensor, skip, bias=None):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    # Skip layernorm doesn't use INT8 inputs and output by default unless it is specified.
    if config.use_int8 and config.use_int8_skipln and not config.is_calib_mode:
        dtype = trt.int8

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)

    fields = [pf_ld, pf_beta, pf_gamma, pf_type ]

    if bias:
        pf_bias = trt.PluginField("bias", bias.numpy(), trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer

# Custom FC plugin is faster than native FC only on older architectures.
def use_custom_fc():
    cc = pycuda.autoinit.device.compute_capability()
    return cc[0] * 10 + cc[1] <= 70

def custom_fc(config, network, input_tensor, out_dims, W):
    pf_out_dims = trt.PluginField("out_dims", np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("W", W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense

def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    if config.use_qat:
        dr_input = init_dict[prefix + 'attention_self_query_input_amax']
        assert(dr_input ==init_dict[prefix + 'attention_self_key_input_amax'] )
        assert(dr_input ==init_dict[prefix + 'attention_self_value_input_amax'] )
        input_tensor.set_dynamic_range(-dr_input, dr_input)

    context_transposed = attention_layer_opt(prefix + "attention_", config, init_dict, network, input_tensor, imask)
    attention_heads = context_transposed.get_output(0)

    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    if config.use_int8:
        W_aout = init_dict[prefix + W_AOUT]
        attention_out_fc = network.add_convolution_nd(attention_heads, hidden_size, (1, 1), W_aout, B_aout)
        B_aout = None

        if not config.use_int8_skipln:
            attention_out_fc.set_output_type(0, trt.DataType.HALF if config.use_fp16 else trt.DataType.FLOAT)

        if config.use_qat:
            dr_fc_aout = init_dict[prefix + 'attention_output_add_local_input_quantizer_amax']
            set_output_range(attention_out_fc, dr_fc_aout)
    elif use_custom_fc():
        W_aoutT = init_dict[prefix + W_AOUT + "_notrans"]
        attention_out_fc = custom_fc(config, network, attention_heads, hidden_size, W_aoutT)
    else:
        W_aout = init_dict[prefix + W_AOUT]
        attention_out_fc = network.add_fully_connected(attention_heads, hidden_size, W_aout, B_aout)
        B_aout = None

    skiplayer = skipln(prefix + "attention_output_layernorm_",config, init_dict, network, attention_out_fc.get_output(0), input_tensor, B_aout)
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

    mid_dense_out = mid_dense.get_output(0)
    POW = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(mid_dense_out, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(mid_dense_out, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), mid_dense_out, trt.ElementWiseOperation.PROD)

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
    if config.use_int8 and not config.use_fc2_gemm:
        W_lout = init_dict[prefix + W_LOUT]
        out_dense = network.add_convolution_nd(intermediate_act, hidden_size, (1, 1), W_lout, B_lout)
        B_lout = None

        if not config.use_int8_skipln:
            out_dense.set_output_type(0, trt.DataType.HALF if config.use_fp16 else trt.DataType.FLOAT)
    elif use_custom_fc():
        W_loutT = init_dict[prefix + W_LOUT + "_notrans"]
        out_dense = custom_fc(config, network, intermediate_act, hidden_size, W_loutT)
    else:
        W_lout = init_dict[prefix + W_LOUT]
        out_dense = network.add_fully_connected(intermediate_act, hidden_size, W_lout, B_lout)
        B_lout = None

    if config.use_qat:
        dr_fc_out = init_dict[prefix + 'output_add_local_input_quantizer_amax']
        set_output_range(out_dense, dr_fc_out)
    set_output_name(out_dense, prefix + "output_", "dense")

    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), attention_ln, B_lout)
    set_output_name(out_layer, prefix + "output_", "reshape")

    return out_layer

def bert_model(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        out_layer = transformer_layer_opt(ss, config,  init_dict, network, prev_input, input_mask)
        prev_input = out_layer.get_output(0)

    if config.use_qat:
        dr_out = init_dict["bert_encoder_final_input_quantizer_amax"]
        set_output_range(out_layer, dr_out)
    return prev_input

def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    W = network.add_constant((1, hidden_size, 2), W_out)
    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)

    OUT = network.add_shuffle(dense.get_output(0))
    OUT.second_transpose = (1, 0, 2, 3, 4)
    set_output_name(OUT, prefix, "squad_logits")
    return OUT

def emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes):
    # int8 only support some of the sequence length, we dynamic on sequence length is not allowed.
    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))
    input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(-1 if len(batch_sizes) > 1 else batch_sizes[0], -1 if len(sequence_lengths) > 1 else sequence_lengths[0]))

    # Specify profiles for the batch sizes we're interested in.
    # Make sure the profile also works for all sizes not covered by the previous profile.

    if len(sequence_lengths) > 1 or len(batch_sizes) > 1:
        for batch_size in sorted(batch_sizes):
            if len(sequence_lengths) == 1:
                profile = builder.create_optimization_profile()
                min_shape = (1, sequence_lengths[0])
                shape = (batch_size, sequence_lengths[0])
                profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
                profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
                profile.set_shape("input_mask", min=min_shape, opt=shape, max=shape)
                builder_config.add_optimization_profile(profile)
            else:
                for sequence_length in sorted(sequence_lengths):
                    profile = builder.create_optimization_profile()
                    min_shape = (1, sequence_length)
                    shape = (batch_size, sequence_length)
                    profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
                    profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
                    profile.set_shape("input_mask", min=min_shape, opt=shape, max=shape)
                    builder_config.add_optimization_profile(profile)

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"].numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"].numpy(), trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)

    output_fp16 = trt.PluginField("output_fp16", np.array([1 if config.use_fp16 else 0]).astype(np.int32), trt.PluginFieldType.INT32)
    mha_type = trt.PluginField("mha_type_id", np.array([get_mha_dtype(config)], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16, mha_type])
    fn = emln_plg_creator.create_plugin("embeddings", pfc)

    input_ids = network.add_shuffle(input_ids)
    input_ids.second_transpose = (1, 0)
    segment_ids = network.add_shuffle(segment_ids)
    segment_ids.second_transpose = (1, 0)
    input_mask = network.add_shuffle(input_mask)
    input_mask.second_transpose = (1, 0)
    inputs = [input_ids.get_output(0),
              segment_ids.get_output(0),
              input_mask.get_output(0)]
    emb_layer = network.add_plugin_v2(inputs, fn)

    if config.use_qat:
        set_output_range(emb_layer, 1, 1)
    set_output_name(emb_layer, "embeddings_", "output")
    return emb_layer

def build_engine(batch_sizes, workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num, verbose):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = workspace_size * (1024 * 1024)
        builder_config.avg_timing_iterations = 8
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.use_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            if not config.use_qat:
                calibrator = BertCalibrator(squad_json, vocab_file, calibrationCacheFile, 1, sequence_lengths[-1], calib_num)
                builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                builder_config.int8_calibrator = calibrator
        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if verbose:
            builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if config.use_sparsity:
            TRT_LOGGER.log(TRT_LOGGER.INFO, "Setting sparsity flag on builder_config.")
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

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

        # only use the largest sequence when in calibration mode
        if config.is_calib_mode:
            sequence_lengths = sequence_lengths[-1:]

        # Create the network
        emb_layer = emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes)
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)

        bert_out = bert_model(config, weights_dict, network, embeddings, mask_idx)

        squad_logits = squad_output("cls_", config, weights_dict, network, bert_out)
        squad_logits_out = squad_logits.get_output(0)

        network.mark_output(squad_logits_out)

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

        if config.use_int8 and not config.use_qat:
            calibrator.free()
        return engine

def generate_calibration_cache(sequence_lengths, workspace_size, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num):
    """
    BERT demo needs a separate engine building path to generate calibration cache.
    This is because we need to configure SLN and MHA plugins in FP32 mode when
    generating calibration cache, and INT8 mode when building the actual engine.
    This cache could be generated by examining certain training data and can be
    reused across different configurations.
    """
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    if not config.use_int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.use_fp16
    config.use_fp16 = False
    config.is_calib_mode = True

    with build_engine([1], workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num, False) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.use_fp16 = saved_use_fp16
    config.is_calib_mode = False

def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--ckpt", required=False,
                        help="The checkpoint file basename, e.g.: basename(model.ckpt-766908.data-00000-of-00001) is model.ckpt-766908")
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-pt", "--pytorch", required=False, help="The PyTorch checkpoint file path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--batch-size", default=[], action="append", help="Batch size(s) to optimize for. The engine will be usable with any batch size below this, but may not be optimal for smaller sizes. Can be specified multiple times to optimize for more than one batch size.", type=int)
    parser.add_argument("-s", "--sequence-length", default=[], action="append", help="Sequence length of the BERT model", type=int)
    parser.add_argument("-c", "--config-dir", required=True,
                        help="The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=1200, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-j", "--squad-json", default="squad/dev-v1.1.json", help="squad json dataset used for int8 calibration", required=False)
    parser.add_argument("-v", "--vocab-file", default="./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt", help="Path to file containing entire understandable vocab", required=False)
    parser.add_argument("-n", "--calib-num", default=100, help="calibration batch numbers", type=int)
    parser.add_argument("-p", "--calib-path", help="calibration cache path", required=False)
    parser.add_argument("-g", "--force-fc2-gemm", action="store_true", help="Force use gemm to implement FC2 layer", required=False)
    parser.add_argument("-iln", "--force-int8-skipln", action="store_true", help="Run skip layernorm with INT8 (FP32 or FP16 by default) inputs and output", required=False)
    parser.add_argument("-imh", "--force-int8-multihead", action="store_true", help="Run multi-head attention with INT8 (FP32 or FP16 by default) input and output", required=False)
    parser.add_argument("-sp", "--sparse", action="store_true", help="Indicates that model is sparse", required=False)
    parser.add_argument("-tcf", "--timing-cache-file", help="Path to tensorrt build timeing cache file, only available for tensorrt 8.0 and later", required=False)
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose logger and set profiling verbosity to DETAILED", required=False)

    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]
    args.sequence_length = args.sequence_length or [128]

    cc = pycuda.autoinit.device.compute_capability()
    if cc[0] * 10 + cc[1] < 75 and args.force_int8_multihead:
        raise RuntimeError("--force-int8-multihead option is only supported on Turing+ GPU.")
    if cc[0] * 10 + cc[1] < 72 and args.force_int8_skipln:
        raise RuntimeError("--force-int8-skipln option is only supported on Xavier+ GPU.")

    if args.verbose:
        TRT_LOGGER.min_severity = TRT_LOGGER.VERBOSE

    bert_config_path = os.path.join(args.config_dir, "bert_config.json")
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))

    config = BertConfig(bert_config_path, args.fp16, args.int8, args.strict, args.force_fc2_gemm, args.force_int8_skipln, args.force_int8_multihead, args.int8 and args.onnx != None, args.sparse, args.timing_cache_file)

    if args.calib_path != None:
        calib_cache = args.calib_path
    else:
        calib_cache = "BertSquadL{}H{}A{}S{}CalibCache".format(config.num_hidden_layers, config.head_size, config.num_attention_heads, "-".join(str(len) for len in args.sequence_length))

    if args.onnx != None:
        weights_dict = load_onnx_weights_and_quant(args.onnx, config)
    elif args.pytorch != None:
        weights_dict = load_pytorch_weights_and_quant(args.pytorch, config)
    elif args.ckpt != None:
        weights_dict = load_tf_weights(args.ckpt, config)
        generate_calibration_cache(args.sequence_length, args.workspace_size, config, weights_dict, args.squad_json, args.vocab_file, calib_cache, args.calib_num)
    else:
        raise RuntimeError("You need either specify TF checkpoint using option --ckpt or ONNX using option --onnx to build TRT BERT model.")

    with build_engine(args.batch_size, args.workspace_size, args.sequence_length, config, weights_dict, args.squad_json, args.vocab_file, calib_cache, args.calib_num, args.verbose) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

if __name__ == "__main__":
    main()
