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

#include "caffeMacros.h"
#include "caffeWeightFactory.h"
#include "half.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

template <typename INPUT, typename OUTPUT>
void* convertInternal(void** ptr, int64_t count, bool* mOK)
{
    assert(ptr != nullptr);
    if (*ptr == nullptr)
    {
        return nullptr;
    }
    if (!count)
    {
        return nullptr;
    }
    auto* iPtr = static_cast<INPUT*>(*ptr);
    auto* oPtr = static_cast<OUTPUT*>(malloc(count * sizeof(OUTPUT)));
    for (int i = 0; i < count; ++i)
    {
        if (static_cast<OUTPUT>(iPtr[i]) > std::numeric_limits<OUTPUT>::max()
            || static_cast<OUTPUT>(iPtr[i]) < std::numeric_limits<OUTPUT>::lowest())
        {
            std::cout << "Error: Weight " << iPtr[i] << " is outside of [" << std::numeric_limits<OUTPUT>::max()
                      << ", " << std::numeric_limits<OUTPUT>::lowest() << "]." << std::endl;
            if (mOK)
            {
                (*mOK) = false;
            }
            break;
        }
        oPtr[i] = iPtr[i];
    }
    (*ptr) = oPtr;
    return oPtr;
}



CaffeWeightFactory::CaffeWeightFactory(const trtcaffe::NetParameter& msg, DataType dataType, std::vector<void*>& tmpAllocs, bool isInitialized)
    : mMsg(msg)
    , mTmpAllocs(tmpAllocs)
    , mDataType(dataType)
    , mInitialized(isInitialized)
{
    mRef = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
}

DataType CaffeWeightFactory::getDataType() const
{
    return mDataType;
}

size_t CaffeWeightFactory::getDataTypeSize() const
{
    switch (getDataType())
    {
    case DataType::kFLOAT:
    case DataType::kINT32:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    }
    return 0;
}

std::vector<void*>& CaffeWeightFactory::getTmpAllocs()
{
    return mTmpAllocs;
}

int CaffeWeightFactory::getBlobsSize(const std::string& layerName)
{
    for (int i = 0, n = mMsg.layer_size(); i < n; ++i)
    {
        if (mMsg.layer(i).name() == layerName)
        {
            return mMsg.layer(i).blobs_size();
        }
    }
    return 0;
}

const trtcaffe::BlobProto* CaffeWeightFactory::getBlob(const std::string& layerName, int index)
{
    if (mMsg.layer_size() > 0)
    {
        for (int i = 0, n = mMsg.layer_size(); i < n; i++)
        {
            if (mMsg.layer(i).name() == layerName && index < mMsg.layer(i).blobs_size())
            {
                return &mMsg.layer(i).blobs(index);
            }
        }
    }
    else
    {
        for (int i = 0, n = mMsg.layers_size(); i < n; i++)
        {
            if (mMsg.layers(i).name() == layerName && index < mMsg.layers(i).blobs_size())
            {
                return &mMsg.layers(i).blobs(index);
            }
        }
    }

    return nullptr;
}

std::vector<Weights> CaffeWeightFactory::getAllWeights(const std::string& layerName)
{
    std::vector<Weights> v;
    for (int i = 0;; i++)
    {
        auto b = getBlob(layerName, i);
        if (b == nullptr)
        {
            break;
        }
        auto weights = getWeights(*b, layerName);
        convert(weights, DataType::kFLOAT);
        v.push_back(weights);
    }
    return v;
}

Weights CaffeWeightFactory::operator()(const std::string& layerName, WeightType weightType)
{
    const trtcaffe::BlobProto* blobMsg = getBlob(layerName, int(weightType));
    if (blobMsg == nullptr)
    {
        std::cout << "Weights for layer " << layerName << " doesn't exist" << std::endl;
        RETURN_AND_LOG_ERROR(getNullWeights(), "ERROR: Attempting to access NULL weights");
        assert(0);
    }
    return getWeights(*blobMsg, layerName);
}

void CaffeWeightFactory::convert(Weights& weights, DataType targetType)
{
    void* tmpAlloc{nullptr};
    if (weights.type == DataType::kFLOAT && targetType == DataType::kHALF)
    {
        tmpAlloc = convertInternal<float, float16>(const_cast<void**>(&weights.values), weights.count, &mOK);
        weights.type = targetType;
    }
    if (weights.type == DataType::kHALF && targetType == DataType::kFLOAT)
    {
        tmpAlloc = convertInternal<float16, float>(const_cast<void**>(&weights.values), weights.count, &mOK);
        weights.type = targetType;
    }
    if (tmpAlloc)
    {
        mTmpAllocs.push_back(tmpAlloc);
    }
}

void CaffeWeightFactory::convert(Weights& weights)
{
    convert(weights, getDataType());
}

bool CaffeWeightFactory::isOK()
{
    return mOK;
}

bool CaffeWeightFactory::isInitialized()
{
    return mInitialized;
}

Weights CaffeWeightFactory::getNullWeights()
{
    return Weights{mDataType, nullptr, 0};
}

Weights CaffeWeightFactory::allocateWeights(int64_t elems, std::uniform_real_distribution<float> distribution)
{
    void* data = malloc(elems * getDataTypeSize());

    switch (getDataType())
    {
    case DataType::kFLOAT:
        for (int64_t i = 0; i < elems; ++i)
        {
            ((float*) data)[i] = distribution(generator);
        }
        break;
    case DataType::kHALF:
        for (int64_t i = 0; i < elems; ++i)
        {
            ((float16*) data)[i] = (float16)(distribution(generator));
        }
        break;
    default:
        break;
    }

    mTmpAllocs.push_back(data);
    return Weights{getDataType(), data, elems};
}

Weights CaffeWeightFactory::allocateWeights(int64_t elems, std::normal_distribution<float> distribution)
{
    void* data = malloc(elems * getDataTypeSize());

    switch (getDataType())
    {
    case DataType::kFLOAT:
        for (int64_t i = 0; i < elems; ++i)
        {
            ((float*) data)[i] = distribution(generator);
        }
        break;
    case DataType::kHALF:
        for (int64_t i = 0; i < elems; ++i)
        {
            ((float16*) data)[i] = (float16)(distribution(generator));
        }
        break;
    default:
        break;
    }

    mTmpAllocs.push_back(data);
    return Weights{getDataType(), data, elems};
}

trtcaffe::Type CaffeWeightFactory::getBlobProtoDataType(const trtcaffe::BlobProto& blobMsg)
{
    if (blobMsg.has_raw_data())
    {
        assert(blobMsg.has_raw_data_type());
        return blobMsg.raw_data_type();
    }
    if (blobMsg.double_data_size() > 0)
    {
        return trtcaffe::DOUBLE;
    }
    return trtcaffe::FLOAT;
}

size_t CaffeWeightFactory::sizeOfCaffeType(trtcaffe::Type type)
{
    if (type == trtcaffe::FLOAT)
    {
        return sizeof(float);
    }
    if (type == trtcaffe::FLOAT16)
    {
        return sizeof(uint16_t);
    }
    return sizeof(double);
}

// The size returned here is the number of array entries, not bytes
std::pair<const void*, size_t> CaffeWeightFactory::getBlobProtoData(const trtcaffe::BlobProto& blobMsg,
                                                                        trtcaffe::Type type, std::vector<void*>& tmpAllocs)
{
    // NVCaffe new binary format. It may carry any type.
    if (blobMsg.has_raw_data())
    {
        assert(blobMsg.has_raw_data_type());
        if (blobMsg.raw_data_type() == type)
        {
            return std::make_pair(&blobMsg.raw_data().front(),
                                    blobMsg.raw_data().size() / sizeOfCaffeType(type));
        }
    }
    // Old BVLC format.
    if (blobMsg.data_size() > 0 && type == trtcaffe::FLOAT)
    {
        return std::make_pair(&blobMsg.data().Get(0), blobMsg.data_size());
    }

    // Converting to the target type otherwise
    const int count = blobMsg.has_raw_data() ? blobMsg.raw_data().size() / sizeOfCaffeType(blobMsg.raw_data_type()) : (blobMsg.data_size() > 0 ? blobMsg.data_size() : blobMsg.double_data_size());

    if (count > 0)
    {
        void* new_memory = malloc(count * sizeOfCaffeType(type));
        tmpAllocs.push_back(new_memory);

        if (type == trtcaffe::FLOAT)
        {
            auto* dst = reinterpret_cast<float*>(new_memory);
            if (blobMsg.has_raw_data())
            {
                if (blobMsg.raw_data_type() == trtcaffe::FLOAT16)
                {
                    const auto* src = reinterpret_cast<const float16*>(&blobMsg.raw_data().front());
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float(src[i]);
                    }
                }
                else if (blobMsg.raw_data_type() == trtcaffe::DOUBLE)
                {
                    const auto* src = reinterpret_cast<const double*>(&blobMsg.raw_data().front());
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float(src[i]);
                    }
                }
            }
            else if (blobMsg.double_data_size() == count)
            {
                for (int i = 0; i < count; ++i)
                {
                    dst[i] = float(blobMsg.double_data(i));
                }
            }
            return std::make_pair(new_memory, count);
        }
        if (type == trtcaffe::FLOAT16)
        {
            auto* dst = reinterpret_cast<float16*>(new_memory);

            if (blobMsg.has_raw_data())
            {
                if (blobMsg.raw_data_type() == trtcaffe::FLOAT)
                {
                    const auto* src = reinterpret_cast<const float*>(&blobMsg.raw_data().front());
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float16(src[i]);
                    }
                }
                else if (blobMsg.raw_data_type() == trtcaffe::DOUBLE)
                {
                    const auto* src = reinterpret_cast<const double*>(&blobMsg.raw_data().front());
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float16(float(src[i]));
                    }
                }
            }
            else if (blobMsg.data_size() == count)
            {
                for (int i = 0; i < count; ++i)
                {
                    dst[i] = float16(blobMsg.data(i));
                }
            }
            else if (blobMsg.double_data_size() == count)
            {
                for (int i = 0; i < count; ++i)
                {
                    dst[i] = float16(float(blobMsg.double_data(i)));
                }
            }
            return std::make_pair(new_memory, count);
        }
    }
    return std::make_pair(nullptr, 0UL);
}

template <typename T>
bool CaffeWeightFactory::checkForNans(const void* values, int count, const std::string& layerName)
{
    const T* v = reinterpret_cast<const T*>(values);
    for (int i = 0; i < count; i++)
    {
        if (std::isnan(float(v[i])))
        {
            std::cout << layerName << ": Nan detected in weights" << std::endl;
            return false;
        }
    }
    return true;
}

Weights CaffeWeightFactory::getWeights(const trtcaffe::BlobProto& blobMsg, const std::string& layerName)
{
    // Always load weights into FLOAT format
    const auto blobProtoData = getBlobProtoData(blobMsg, trtcaffe::FLOAT, mTmpAllocs);

    if (blobProtoData.first == nullptr)
    {
        const int bits = mDataType == DataType::kFLOAT ? 32 : 16;
        std::cout << layerName << ": ERROR - " << bits << "-bit weights not found for "
                    << bits << "-bit model" << std::endl;
        mOK = false;
        return Weights{DataType::kFLOAT, nullptr, 0};
    }

    mOK &= checkForNans<float>(blobProtoData.first, int(blobProtoData.second), layerName);
    return Weights{DataType::kFLOAT, blobProtoData.first, int(blobProtoData.second)};
}
