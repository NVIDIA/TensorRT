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
ILayer* parseReduction(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    // The first axis to reduce to a scalar -- may be negative to index from the
    // end (e.g., -1 for the last axis).
    // (Currently, only reduction along ALL "tail" axes is supported; reduction
    // of axis M through N, where N < num_axes - 1, is unsupported.)
    // Suppose we have an n-axis bottom Blob with shape:
    //     (d0, d1, d2, ..., d(m-1), dm, d(m+1), ..., d(n-1)).
    // If axis == m, the output Blob will have shape
    //     (d0, d1, d2, ..., d(m-1)),
    // and the ReductionOp operation is performed (d0 * d1 * d2 * ... * d(m-1))
    // times, each including (dm * d(m+1) * ... * d(n-1)) individual data.
    // If axis == 0 (the default), the output Blob always has the empty shape
    // (count 1), performing reduction across the entire input --
    // often useful for creating new loss functions.
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    // operation == 1 is SUM -> ReduceOperation::kSUM
    const int SUM = 1;
    // operation == 2 is ASUM -> UnaryOperation::kABS and ReduceOperation::kSUM
    const int ASUM = 2;
    // operation == 3 is SUMSQ -> ElementWiseOperation::kPROD and ReduceOperation::kSUM
    const int SUMSQ = 3;
    // operation == 4 is MEAN -> ReduceOperation::kAVG
    const int MEAN = 4;

    const trtcaffe::ReductionParameter& p = msg.reduction_param();
    bool hasOperation = p.has_operation();              // optional parameter
    bool hasAxis = p.has_axis();                        // optional parameter
    bool hasCoeff = p.has_coeff();                      // optional parameter
    int operation = hasOperation ? p.operation() : SUM; // default is SUM
    int axis = hasAxis ? p.axis() : 0;                  // default is 0
    axis = (axis < 0) ? 4 + axis : axis;                // axis negative number correction
    float coeff = hasCoeff ? p.coeff() : 1.0;           // default is 1

    // With implicit batch dimensions:
    //   acceptable axis values: 1, 2, 3, -1, -2, -3
    //   unacceptable axis values: 0 and anything else
    //   acceptable corrected axis values: 1, 2, 3
    //   unacceptable corrected axis values: 0 and anything else
    //
    // With implicit batch dimensions:
    //   acceptable axis values: 1, 2, 3, 0, -1, -2, -3
    //   unacceptable axis values: anything else
    //   acceptable corrected axis values: 0, 1, 2, 3
    //   unacceptable corrected axis values: 0
    //
    // protect against "garbage" input arguments
    if (axis < 0 || axis > 3)
    {
        std::cout << "Caffe Parser: Invalid axis in reduction layer - can only reduce NCHW input." << std::endl;
        return nullptr;
    }
    if (network.hasImplicitBatchDimension() && axis == 0)
    {
        std::cout << "Caffe Parser: Invalid axis in reduction layer - cannot reduce over batch size dimension."
                  << std::endl;
        return nullptr;
    }

    ReduceOperation op = (operation == MEAN ? ReduceOperation::kAVG : ReduceOperation::kSUM);
    // For implicit batch, corrected axis values are 1, 2, 3
    // only reduction along tail dimensions is supported
    // 1 means 111 or 4 + 2 + 1 = 7
    // 2 means 110 or 4 + 2 = 6
    // 3 means 100 or 4
    // Let's employ a bit shift trick instead
    // 10000 = 16
    // axis == 1: 1u << axis is 2 and so (16 - 2) >> 1 = 7 or 111
    // axis == 2: 1u << axis is 4 and so (16 - 4) >> 1 = 6 or 110
    // axis == 3: 1u << axis is 8 and so (16 - 8) >> 1 = 4 or 100
    //
    // For explicit batch, corrected axis values are 0, 1, 2, 3
    // 1 means 1111 or 8 + 4 + 2 + 1 = 15
    // 2 means 1110 or 8 + 4 + 2 = 14
    // 3 means 1100 or 8 + 4 = 12
    // 4 means 1000 or 8
    // 10000 = 16
    // axis == 0: 1u << axis is 1 and so 16 - 1 = 15 or 1111
    // axis == 1: 1u << axis is 2 and so 16 - 2 = 14 or 1110
    // axis == 2: 1u << axis is 4 and so 16 - 4 = 12 or 1100
    // axis == 3: 1u << axis is 8 and so 16 - 8 = 8 or 1000
    uint32_t reduceAxes = (16 - (1u << axis)) >> static_cast<int>(network.hasImplicitBatchDimension());

    ITensor* input = tensors[msg.bottom(0)];
    ILayer* returnVal = nullptr;
    // need to add in layer before for ASUM and SUMSQ
    if (operation == ASUM)
    {
        returnVal = network.addUnary(*input, UnaryOperation::kABS);
        input = returnVal->getOutput(0);
        std::string layerName = msg.name() + std::string("/reductionLayer/unaryLayer");
        returnVal->setName(layerName.c_str());
    }
    else if (operation == SUMSQ)
    {
        returnVal = network.addElementWise(*input, *input, ElementWiseOperation::kPROD);
        input = returnVal->getOutput(0);
        std::string layerName = msg.name() + std::string("/reductionLayer/elementWiseLayer");
        returnVal->setName(layerName.c_str());
    }

// add in the actual reduce layer
#define GIE_3111 0
#if GIE_3111
    returnVal = network.addReduce(*input, op, reduceAxes, false);
#else
    returnVal = network.addReduce(*input, op, reduceAxes, true);
    // output a warning
    std::cout << "Warning: The Reduce layer does not discard reduced dimensions. The reduced dimensions are treated as dimensions of size one in the output of the Reduce layer." << std::endl;
#endif
    input = returnVal->getOutput(0);
    std::string reduceLayerName = msg.name() + std::string("/reductionLayer/reduceLayer");
    returnVal->setName(reduceLayerName.c_str());

    // need to add in layer after for coeff != 1.0
    if (coeff != 1.0f)
    {
        auto* shiftArr = (float*) malloc(sizeof(float));
        auto* scaleArr = (float*) malloc(sizeof(float));
        auto* powerArr = (float*) malloc(sizeof(float));
        weightFactory.getTmpAllocs().push_back(shiftArr);
        weightFactory.getTmpAllocs().push_back(scaleArr);
        weightFactory.getTmpAllocs().push_back(powerArr);
        *shiftArr = 0.0f;
        *scaleArr = coeff;
        *powerArr = 1.0f;

        Weights wShift, wScale, wPower;

        wShift = Weights{DataType::kFLOAT, shiftArr, 1};
        wScale = Weights{DataType::kFLOAT, scaleArr, 1};
        wPower = Weights{DataType::kFLOAT, powerArr, 1};

        returnVal = network.addScale(*input, ScaleMode::kUNIFORM, wShift, wScale, wPower);
        std::string layerName = msg.name() + std::string("/reductionLayer/scaleLayer");
        returnVal->setName(layerName.c_str());
    }

    return returnVal;
}
} //namespace nvcaffeparser1
