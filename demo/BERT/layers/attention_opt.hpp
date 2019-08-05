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

#include "attention_keys.hpp"
#include "qkv2context_plugin.hpp"
namespace bert
{

ITensor* attention_layer_opt(const std::string& prefix, const BertConfig& config, WeightDict& init_dict,
    INetworkDefinition* network, ITensor* input_tensor, ITensor* input_mask = nullptr, TensorDict* dbg = nullptr,
    int verbose = 0)
{

    assert(input_tensor);
    assert(network);
    Dims idims = input_tensor->getDimensions();

    assert(idims.nbDims == 5);
    int B = idims.d[0];
    int S = idims.d[1];
    int hidden_size = idims.d[2];

    const int num_heads = config.num_attention_heads;

    assert(hidden_size % num_heads == 0);
    int head_size = hidden_size / num_heads;

    Weights Wall_ = init_dict.at(prefix + WQKV);
    Weights Ball_ = init_dict.at(prefix + BQKV);

    IFullyConnectedLayer* mult_all = network->addFullyConnected(*input_tensor, 3 * hidden_size, Wall_, Ball_);
    set_name(mult_all, prefix, "qkv_mult");

    ITensor* shuffle_out = mult_all->getOutput(0);

    bool has_mask = input_mask != nullptr;
    QKV2ContextPlugin qkv2ctx_plug("qkv2ctx", hidden_size, num_heads, B, S, has_mask);
    ITensor* qkv_in[2] = {shuffle_out, input_mask};
    IPluginV2Layer* qkv2ctx = network->addPluginV2(qkv_in, 1 + has_mask, qkv2ctx_plug);
    set_name(qkv2ctx, prefix, "context_layer");
    ITensor* context = qkv2ctx->getOutput(0);

    assert(context);

    return context;
}
}
