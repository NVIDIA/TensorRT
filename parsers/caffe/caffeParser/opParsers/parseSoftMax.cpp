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
ILayer* parseSoftMax(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::SoftmaxParameter& p = msg.softmax_param();

    // Caffe supports negative axis, indexing from the last dimension
    // However, there is a discrepancy in the internal tensor dimension in some cases.
    // For example. InnerProduct produces flat 1D blob in Caffe, while TensorRT still
    // produces CHW format. MNIST sample generates input to Softmax as,
    //     Caffe    = n x 10
    //     TensorRT = n x 10 x 1 x 1
    // To make sure we do not run into issues, negative axis won't be supported in TensorRT
    int nbDims = tensors[msg.bottom(0)]->getDimensions().nbDims;
    bool hasAxis = p.has_axis();       // optional parameter
    int axis = hasAxis ? p.axis() : 1; // default is 1

    bool axisAbort = (axis <= 0) || (axis > 3) || (axis > nbDims);

    if (axisAbort)
    {
        std::cout << "Caffe Parser: Invalid axis in softmax layer - Cannot perform softmax along batch size dimension and expects NCHW input. Negative axis is not supported in TensorRT, please use positive axis indexing" << std::endl;
        return nullptr;
    }

    auto softmax = network.addSoftMax(*tensors[msg.bottom(0)]);
    // Do this so that setAxes is not used when the default axis is needed
    // This is necessary to preserve correct roll-into-the-batch dimension behaviour for samples like FasterRCNN
    // NCHW -> default axis when setAxes is not called will be 1 (the C dimension)
    // NPCHW -> default axis when setAxes is not called will be 2 (the C dimension)
    if (hasAxis)
    {
        uint32_t axes = 1u << (axis - 1);
        softmax->setAxes(axes);
    }
    return softmax;
}
} //namespace nvcaffeparser1