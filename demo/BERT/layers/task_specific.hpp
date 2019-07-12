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
namespace bert
{

const std::string SQD_W = "squad_output_weights";
const std::string SQD_B = "squad_output_bias";

ITensor* squad_output(const std::string& prefix, const BertConfig& config, WeightDict& init_dict,
    INetworkDefinition* network, ITensor* input_tensor, TensorDict* dbg = nullptr)
{

    assert(input_tensor);
    assert(network);

    Dims idims = input_tensor->getDimensions();
    assert(idims.nbDims == 5);
    int B = idims.d[0];
    int S = idims.d[1];
    int hidden_size = idims.d[2];

    Weights W_out = init_dict.at(prefix + SQD_W);
    Weights B_out = init_dict.at(prefix + SQD_B);

    IConstantLayer* W = network->addConstant(Dims3{1, hidden_size, 2}, W_out);
    assert(W);

    IFullyConnectedLayer* dense = network->addFullyConnected(*input_tensor, 2, W_out, B_out);
    assert(dense);
    set_name(dense, prefix, "dense");

    IShuffleLayer* squad_logits = network->addShuffle(*dense->getOutput(0));
    assert(squad_logits);
    set_name(squad_logits, prefix, "squad_logits");
    squad_logits->setReshapeDimensions(Dims3{B, S, 2});
    squad_logits->setSecondTranspose({2, 0, 1});

    return squad_logits->getOutput(0);
}
}
