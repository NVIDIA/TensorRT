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

#ifndef TRT_ATTENTION_KEYS_H
#define TRT_ATTENTION_KEYS_H

#include <string>

namespace bert
{
const std::string WQ = "query_kernel";
const std::string BQ = "query_bias";
const std::string WK = "key_kernel";
const std::string BK = "key_bias";
const std::string WV = "value_kernel";
const std::string BV = "value_bias";

const std::string WQKV = "qkv_kernel";
const std::string BQKV = "qkv_bias";
}

#endif // TRT_ATTENTION_KEYS_H
