/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseEltwise(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 2, 1))
    {
        return nullptr;
    }

    const trtcaffe::EltwiseParameter& p = msg.eltwise_param();

    ElementWiseOperation op = ElementWiseOperation::kSUM;
    switch (p.operation())
    {
    case trtcaffe::EltwiseParameter_EltwiseOp_SUM: op = ElementWiseOperation::kSUM; break;
    case trtcaffe::EltwiseParameter_EltwiseOp_PROD: op = ElementWiseOperation::kPROD; break;
    case trtcaffe::EltwiseParameter_EltwiseOp_MAX: op = ElementWiseOperation::kMAX; break;
    }

    return network.addElementWise(*tensors[msg.bottom(0)], *tensors[msg.bottom(1)], op);
}
} //namespace nvcaffeparser1
