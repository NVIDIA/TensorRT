/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "attention_opt.hpp"
#include "gelu_plugin.hpp"
#include "skip_layer_norm_plugin.hpp"
#include "transformer_keys.hpp"
namespace bert
{

ITensor* skipln(
    const std::string& prefix, WeightDict& init_dict, INetworkDefinition* network, ITensor* input_tensor, ITensor* skip)
{

    Dims idims = input_tensor->getDimensions();
    assert(idims.nbDims == 5);
    int hidden_size = idims.d[2];

    Weights& wbeta = init_dict.at(prefix + "beta");
    Weights& wgamma = init_dict.at(prefix + "gamma");
    SkipLayerNormPlugin skipln_plug("skipln", hidden_size, wbeta, wgamma);
    ITensor* skipln_inputs[2] = {input_tensor, skip};

    IPluginV2Layer* layer = network->addPluginV2(skipln_inputs, 2, skipln_plug);
    return layer->getOutput(0);
}

ITensor* transformer_layer_opt(const std::string& prefix, const BertConfig& config, WeightDict& init_dict,
    INetworkDefinition* network, ITensor* input_tensor, ITensor* imask = nullptr, TensorDict* dbg = nullptr)
{

    assert(input_tensor);
    assert(network);

    Dims idims = input_tensor->getDimensions();
    assert(idims.nbDims == 5);
    int hidden_size = idims.d[2];

    ITensor* attention_heads
        = attention_layer_opt(prefix + "attention_self_", config, init_dict, network, input_tensor, imask, dbg);

    Weights W_aout = init_dict.at(prefix + W_AOUT);
    Weights B_aout = init_dict.at(prefix + B_AOUT);

    IFullyConnectedLayer* attention_out_fc = network->addFullyConnected(*attention_heads, hidden_size, W_aout, B_aout);

    ITensor* attention_ln = skipln(
        prefix + "attention_output_layernorm_", init_dict, network, attention_out_fc->getOutput(0), input_tensor);

    Weights W_mid = init_dict.at(prefix + W_MID);
    Weights B_mid = init_dict.at(prefix + B_MID);

    IFullyConnectedLayer* mid_dense = network->addFullyConnected(*attention_ln, config.intermediate_size, W_mid, B_mid);

    // gelu
    auto plug = GeluPlugin("gelu");
    ITensor* mid_dense_out = mid_dense->getOutput(0);
    IPluginV2Layer* gelu_layer = network->addPluginV2(&mid_dense_out, 1, plug);
    ITensor* intermediate_act = gelu_layer->getOutput(0);
    assert(intermediate_act);
    set_name(intermediate_act, prefix, "gelu");

    // dense to hidden size
    Weights W_lout = init_dict.at(prefix + W_LOUT);
    Weights B_lout = init_dict.at(prefix + B_LOUT);

    IFullyConnectedLayer* out_dense = network->addFullyConnected(*intermediate_act, hidden_size, W_lout, B_lout);
    set_name(out_dense, prefix + "output_", "dense");

    ITensor* out_ln = skipln(prefix + "output_layernorm_", init_dict, network, out_dense->getOutput(0), attention_ln);

    assert(out_ln);

    set_name(out_ln, prefix + "output_", "reshape");

    return out_ln;
}
}
