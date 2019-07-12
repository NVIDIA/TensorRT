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
namespace bert{

void qkv2context(const std::string& prefix, const BertConfig& config, WeightDict& init_dict,
    INetworkDefinition* network, ITensor* input_tensor, ITensor*& context, ITensor* input_mask = nullptr,
    TensorDict* dbg = nullptr)
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

    IShuffleLayer* shuffle_all = network->addShuffle(*mult_all->getOutput(0));
    shuffle_all->setReshapeDimensions(Dims{5, B, S, 3, num_heads, head_size});
    shuffle_all->setSecondTranspose({2, 0, 3, 1, 4});
    set_name(shuffle_all, prefix, "qkv_layer"); // 3xBxNxSxH
    ITensor* shuffle_out = shuffle_all->getOutput(0);

    bool has_mask = input_mask != nullptr;
    QKV2ContextPlugin qkv2ctx_plug("qkv2ctx", hidden_size, num_heads, B, S, has_mask);
    ITensor* qkv_in[2] = {shuffle_out, input_mask};
    IPluginV2Layer* qkv2ctx = network->addPluginV2(qkv_in, 1 + has_mask, qkv2ctx_plug);
    set_name(qkv2ctx, prefix, "context_layer");
    context = qkv2ctx->getOutput(0);
}

ITensor* attention_layer_opt(const std::string& prefix, const BertConfig& config, WeightDict& init_dict,
    INetworkDefinition* network, ITensor* input_tensor, ITensor* imask = nullptr, TensorDict* dbg = nullptr,
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

    ITensor* context = nullptr;

    qkv2context(prefix, config, init_dict, network, input_tensor, context, imask, dbg);

    assert(context);

    IShuffleLayer* context_transposed = network->addShuffle(*context);
    assert(context_transposed);
    set_name(context_transposed, prefix, "context_transposed");
    context_transposed->setFirstTranspose({0, 2, 1, 3});
    context_transposed->setReshapeDimensions(Dims{5, B, S, hidden_size, 1, 1});

    auto o_dims = context_transposed->getOutput(0)->getDimensions();
    auto i_dims = input_tensor->getDimensions();
    assert(o_dims.nbDims == i_dims.nbDims);
    for (int it = 0; it < i_dims.nbDims; it++)
    {
        assert(o_dims.d[it] == i_dims.d[it]);
    }

    return context_transposed->getOutput(0);
}
}
