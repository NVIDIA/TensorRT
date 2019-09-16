/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_BERT_H
#define TRT_BERT_H

#include "driver.h"
#include <NvInfer.h>
#include <string>

constexpr const char* kMODEL_INPUT0_NAME = "input_ids";
constexpr const char* kMODEL_INPUT1_NAME = "segment_ids";
constexpr const char* kMODEL_INPUT2_NAME = "input_mask";

namespace bert
{

struct BERTDriver : DynamicDriver
{
    const int mNumHeads;

    BERTDriver(const int nbHeads, const bool useFp16, const size_t maxWorkspaceSize, const OptProfiles& optProfiles);

    BERTDriver(const std::string& enginePath);

    void buildNetwork(nvinfer1::INetworkDefinition* network, const HostTensorMap& params) override;
};
}

#endif // TRT_BERT_H
