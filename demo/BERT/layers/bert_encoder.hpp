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

#include "transformer_layer_opt.hpp"

namespace bert
{

ITensor* bert_model(const BertConfig& config, WeightDict& init_dict, INetworkDefinition* network, ITensor* input_tensor,
    ITensor* input_mask = nullptr, TensorDict* dbg = nullptr)
{

    ITensor* prev_input = input_tensor;

    for (int layer = 0; layer < config.num_hidden_layers; layer++)
    {
        std::stringstream ss;
        ss << "l" << layer << "_";

        prev_input = transformer_layer_opt(ss.str(), config, init_dict, network, prev_input, input_mask, dbg);
    }

    return prev_input;
}
}
