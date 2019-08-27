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

#ifndef TRT_TRANSFORMER_KEYS_H
#define TRT_TRANSFORMER_KEYS_H

#include <string>

namespace bert
{
const std::string W_AOUT = "attention_output_dense_kernel";
const std::string B_AOUT = "attention_output_dense_bias";
const std::string AOUT_LN_BETA = "attention_output_layernorm_beta";
const std::string AOUT_LN_GAMMA = "attention_output_layernorm_gamma";
const std::string W_MID = "intermediate_dense_kernel";
const std::string B_MID = "intermediate_dense_bias";
const std::string W_LOUT = "output_dense_kernel";
const std::string B_LOUT = "output_dense_bias";
const std::string LOUT_LN_BETA = "output_layernorm_beta";
const std::string LOUT_LN_GAMMA = "output_layernorm_gamma";
}

#endif // TRT_TRANSFORMER_KEYS_H
