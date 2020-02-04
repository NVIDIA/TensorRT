/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseResize(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }
    if (getInferLibVersion() <= 5000) {
      RETURN_AND_LOG_ERROR(nullptr, "IResizeLayer can be supported by TensorRT6 or Newer version");
      
    }

    const trtcaffe::ResizeParameter& p = msg.resize_param();

    auto mode = p.interp_mode(0);
    nvinfer1::ResizeMode resizeMode;
    if (mode == trtcaffe::ResizeParameter_Interp_mode_NEAREST) {
      resizeMode = nvinfer1::ResizeMode::kNEAREST;
    } else if (mode == trtcaffe::ResizeParameter_Interp_mode_LINEAR) {
      resizeMode = nvinfer1::ResizeMode::kLINEAR;
    } else {
      RETURN_AND_LOG_ERROR(nullptr, "This version of TensorRT only supports nearest or linear mode!");
    }
    float scales[] = {1, (float)(p.height_scale()), (float)(p.width_scale())};

    nvinfer1::IResizeLayer* layer = network.addResize(*tensors[msg.bottom(0)]);
    layer->setResizeMode(resizeMode);
    layer->setScales(scales, 3);
    return layer;
}
} //namespace nvcaffeparser1

