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
template <typename T>
inline bool bnConvertWrap(float scaleFactor, const Weights& variance, const Weights& mean,
                          const Weights& scaleBlob, const Weights& biasBlob,
                          Weights& shift, Weights& scale, float eps,
                          bool nvCaffe, CaffeWeightFactory& weightFactory)
{

    assert(shift.count == scale.count);
    if (nvCaffe)
    {
        if (scaleBlob.values == nullptr)
        {
            return false;
        }
        if (biasBlob.values == nullptr)
        {
            return false;
        }
    }
    T* shiftv = reinterpret_cast<T*>(malloc(sizeof(T) * shift.count));
    if (!shiftv)
    {
        return false;
    }

    T* scalev = reinterpret_cast<T*>(malloc(sizeof(T) * scale.count));
    if (!scalev)
    {
        free(shiftv);
        return false;
    }
    shift.values = shiftv;
    scale.values = scalev;
    weightFactory.getTmpAllocs().push_back(shiftv);
    weightFactory.getTmpAllocs().push_back(scalev);

    const T* m = reinterpret_cast<const T*>(mean.values);
    const T* v = reinterpret_cast<const T*>(variance.values);
    for (int i = 0; i < shift.count; i++)
    {
        scalev[i] = T(1.0f / std::sqrt(float(v[i]) * scaleFactor + eps));
        shiftv[i] = T(-(float(m[i]) * scaleFactor * float(scalev[i])));
    }

    if (nvCaffe)
    {
        const T* s = reinterpret_cast<const T*>(scaleBlob.values);
        const T* b = reinterpret_cast<const T*>(biasBlob.values);
        for (int i = 0; i < shift.count; i++)
        {
            scalev[i] = T(float(scalev[i]) * s[i]);
            shiftv[i] = T(float(shiftv[i]) * s[i]) + b[i];
        }
    }
    return true;
}

ILayer* parseBatchNormalization(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::BatchNormParameter& p = msg.batch_norm_param();
    bool nvCaffe = weightFactory.getBlobsSize(msg.name()) == 5;

    int C = parserutils::getC(tensors[msg.bottom(0)]->getDimensions());

    Weights mean{DataType::kFLOAT, nullptr, 0},
        variance{DataType::kFLOAT, nullptr, 0},
        scaleBlob{DataType::kFLOAT, nullptr, 0},
        biasBlob{DataType::kFLOAT, nullptr, 0},
        movingAverage{DataType::kFLOAT, nullptr, 0};

    // Because of the incompatible nature of the batch normalizations
    // between BLVC Caffe and nvCaffe, two different paths have to be
    // used.
    if (nvCaffe)
    {
        if (weightFactory.isInitialized())
        {
            mean = weightFactory(msg.name(), WeightType::kNVMEAN);
            variance = weightFactory(msg.name(), WeightType::kNVVARIANCE);
            scaleBlob = weightFactory(msg.name(), WeightType::kNVSCALE);
            biasBlob = weightFactory(msg.name(), WeightType::kNVBIAS);
        }
        else
        {
            mean = weightFactory.allocateWeights(C);
            variance = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            scaleBlob = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            biasBlob = weightFactory.allocateWeights(C);
        }
    }
    else
    {
        if (weightFactory.isInitialized())
        {
            mean = weightFactory(msg.name(), WeightType::kMEAN);
            variance = weightFactory(msg.name(), WeightType::kVARIANCE);
            movingAverage = weightFactory(msg.name(), WeightType::kMOVING_AVERAGE);
        }
        else
        {
            mean = weightFactory.allocateWeights(C);
            variance = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            movingAverage = weightFactory.allocateWeights(1, std::uniform_real_distribution<float>(0.99F, 1.01F));
        }
        assert(mean.count == variance.count && movingAverage.count == 1);
    }

    Weights shift{mean.type, nullptr, mean.count};
    Weights scale{mean.type, nullptr, mean.count};
    Weights power{mean.type, nullptr, 0};
    bool success{false};
    float scaleFactor{1.0f};
    if (!nvCaffe)
    {
        float average{0.0f};
        // Inside weightFactory, the weights are generated based off the type.
        if (mean.type == DataType::kFLOAT)
        {
            average = *(static_cast<const float*>(movingAverage.values));
        }
        else
        {
            average = *(static_cast<const float16*>(movingAverage.values));
        }
        if (average == 0.0f)
        {
            std::cout << "Batch normalization moving average is zero" << std::endl;
            return nullptr;
        }
        scaleFactor /= average;
    }
    if (mean.type == DataType::kFLOAT)
    {
        success = bnConvertWrap<float>(scaleFactor, variance, mean, scaleBlob, biasBlob, shift, scale, p.eps(), nvCaffe, weightFactory);
    }
    else
    {
        success = bnConvertWrap<float16>(scaleFactor, variance, mean, scaleBlob, biasBlob, shift, scale, p.eps(), nvCaffe, weightFactory);
    }

    if (!success)
    {
        return nullptr;
    }

    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kCHANNEL, shift, scale, power);
}
} //namespace nvcaffeparser1
