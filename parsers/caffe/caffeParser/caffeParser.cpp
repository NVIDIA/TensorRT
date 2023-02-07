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

#include <iostream>

#include "caffeMacros.h"
#include "caffeParser.h"
#include "opParsers.h"
#include "parserUtils.h"
#include "readProto.h"
#include "binaryProtoBlob.h"
#include "google/protobuf/text_format.h"
#include "half.h"
#include "NvInferPluginUtils.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

CaffeParser::~CaffeParser()
{
    for (auto v : mTmpAllocs)
    {
        free(v);
    }
    for (auto p : mNewPlugins)
    {
        if (p)
        {
            p->destroy();
        }
    }
    delete mBlobNameToTensor;
}

std::vector<nvinfer1::PluginField> CaffeParser::parseNormalizeParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::NormalizeParameter& p = msg.norm_param();

    int* acrossSpatial = allocMemory<int32_t>();
    *acrossSpatial = p.across_spatial() ? 1 : 0;
    f.emplace_back("acrossSpatial", acrossSpatial, PluginFieldType::kINT32, 1);

    int* channelShared = allocMemory<int32_t>();
    *channelShared = p.channel_shared() ? 1 : 0;
    f.emplace_back("channelShared", channelShared, PluginFieldType::kINT32, 1);

    auto* eps = allocMemory<float>();
    *eps = p.eps();
    f.emplace_back("eps", eps, PluginFieldType::kFLOAT32, 1);

    std::vector<Weights> w;
    // If .caffemodel is not provided, need to randomize the weight
    if (!weightFactory.isInitialized())
    {
        int C = parserutils::getC(tensors[msg.bottom(0)]->getDimensions());
        w.emplace_back(weightFactory.allocateWeights(C, std::normal_distribution<float>(0.0F, 1.0F)));
    }
    else
    {
        // Use the provided weight from .caffemodel
        w = weightFactory.getAllWeights(msg.name());
    }

    for (auto weight : w)
    {
        f.emplace_back("weights", weight.values, PluginFieldType::kFLOAT32, weight.count);
    }

    int* nbWeights = allocMemory<int32_t>();
    *nbWeights = w.size();
    f.emplace_back("nbWeights", nbWeights, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField> CaffeParser::parsePriorBoxParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::PriorBoxParameter& p = msg.prior_box_param();

    int minSizeSize = p.min_size_size();
    auto* minSize = allocMemory<float>(minSizeSize);
    for (int i = 0; i < minSizeSize; ++i)
    {
        minSize[i] = p.min_size(i);
    }
    f.emplace_back("minSize", minSize, PluginFieldType::kFLOAT32, minSizeSize);

    int maxSizeSize = p.max_size_size();
    auto* maxSize = allocMemory<float>(maxSizeSize);
    for (int i = 0; i < maxSizeSize; ++i)
    {
        maxSize[i] = p.max_size(i);
    }
    f.emplace_back("maxSize", maxSize, PluginFieldType::kFLOAT32, maxSizeSize);

    int aspectRatiosSize = p.aspect_ratio_size();
    auto* aspectRatios = allocMemory<float>(aspectRatiosSize);
    for (int i = 0; i < aspectRatiosSize; ++i)
    {
        aspectRatios[i] = p.aspect_ratio(i);
    }
    f.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32, aspectRatiosSize);

    int varianceSize = p.variance_size();
    auto* variance = allocMemory<float>(varianceSize);
    for (int i = 0; i < varianceSize; ++i)
    {
        variance[i] = p.variance(i);
    }
    f.emplace_back("variance", variance, PluginFieldType::kFLOAT32, varianceSize);

    int* flip = allocMemory<int32_t>();
    *flip = p.flip() ? 1 : 0;
    f.emplace_back("flip", flip, PluginFieldType::kINT32, 1);

    int* clip = allocMemory<int32_t>();
    *clip = p.clip() ? 1 : 0;
    f.emplace_back("clip", clip, PluginFieldType::kINT32, 1);

    int* imgH = allocMemory<int32_t>();
    *imgH = p.has_img_h() ? p.img_h() : p.img_size();
    f.emplace_back("imgH", imgH, PluginFieldType::kINT32, 1);

    int* imgW = allocMemory<int32_t>();
    *imgW = p.has_img_w() ? p.img_w() : p.img_size();
    f.emplace_back("imgW", imgW, PluginFieldType::kINT32, 1);

    auto* stepH = allocMemory<float>();
    *stepH = p.has_step_h() ? p.step_h() : p.step();
    f.emplace_back("stepH", stepH, PluginFieldType::kFLOAT32, 1);

    auto* stepW = allocMemory<float>();
    *stepW = p.has_step_w() ? p.step_w() : p.step();
    f.emplace_back("stepW", stepW, PluginFieldType::kFLOAT32, 1);

    auto* offset = allocMemory<float>();
    *offset = p.offset();
    f.emplace_back("offset", offset, PluginFieldType::kFLOAT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField> CaffeParser::parseDetectionOutputParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::DetectionOutputParameter& p = msg.detection_output_param();
    const trtcaffe::NonMaximumSuppressionParameter& nmsp = p.nms_param();

    int* shareLocation = allocMemory<int32_t>();
    *shareLocation = p.share_location() ? 1 : 0;
    f.emplace_back("shareLocation", shareLocation, PluginFieldType::kINT32, 1);

    int* varianceEncodedInTarget = allocMemory<int32_t>();
    *varianceEncodedInTarget = p.variance_encoded_in_target() ? 1 : 0;
    f.emplace_back("varianceEncodedInTarget", varianceEncodedInTarget, PluginFieldType::kINT32, 1);

    int* backgroundLabelId = allocMemory<int32_t>();
    *backgroundLabelId = p.background_label_id();
    f.emplace_back("backgroundLabelId", backgroundLabelId, PluginFieldType::kINT32, 1);

    int* numClasses = allocMemory<int32_t>();
    *numClasses = p.num_classes();
    f.emplace_back("numClasses", numClasses, PluginFieldType::kINT32, 1);

    //nms
    int* topK = allocMemory<int32_t>();
    *topK = nmsp.top_k();
    f.emplace_back("topK", topK, PluginFieldType::kINT32, 1);

    int* keepTopK = allocMemory<int32_t>();
    *keepTopK = p.keep_top_k();
    f.emplace_back("keepTopK", keepTopK, PluginFieldType::kINT32, 1);

    auto* confidenceThreshold = allocMemory<float>();
    *confidenceThreshold = p.confidence_threshold();
    f.emplace_back("confidenceThreshold", confidenceThreshold, PluginFieldType::kFLOAT32, 1);

    //nms
    auto* nmsThreshold = allocMemory<float>();
    *nmsThreshold = nmsp.nms_threshold();
    f.emplace_back("nmsThreshold", nmsThreshold, PluginFieldType::kFLOAT32, 1);

    // input order = {0, 1, 2} in Caffe
    int* inputOrder = allocMemory<int32_t>(3);
    inputOrder[0] = 0;
    inputOrder[1] = 1;
    inputOrder[2] = 2;
    f.emplace_back("inputOrder", inputOrder, PluginFieldType::kINT32, 3);

    // confSigmoid = false for Caffe
    int* confSigmoid = allocMemory<int32_t>();
    *confSigmoid = 0;
    f.emplace_back("confSigmoid", confSigmoid, PluginFieldType::kINT32, 1);

    // isNormalized = true for Caffe
    int* isNormalized = allocMemory<int32_t>();
    *isNormalized = 1;
    f.emplace_back("isNormalized", isNormalized, PluginFieldType::kINT32, 1);

    // codeTypeSSD : from NvInferPlugin.h
    // CORNER = 0, CENTER_SIZE = 1, CORNER_SIZE = 2, TF_CENTER = 3
    int* codeType = allocMemory<int32_t>();
    switch (p.code_type())
    {
    case trtcaffe::PriorBoxParameter::CORNER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CENTER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CENTER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CORNER: // CORNER is default
    default:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER);
        break;
    }
    f.emplace_back("codeType", codeType, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField> CaffeParser::parseLReLUParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::ReLUParameter& p = msg.relu_param();
    auto* negSlope = allocMemory<float>();
    *negSlope = p.negative_slope();
    f.emplace_back("negSlope", negSlope, PluginFieldType::kFLOAT32, 1);
    return f;
}
std::vector<nvinfer1::PluginField> CaffeParser::parseRPROIParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::ROIPoolingParameter& p1 = msg.roi_pooling_param();
    const trtcaffe::RegionProposalParameter& p2 = msg.region_proposal_param();

    // Memory allocations for plugin field variables
    int* poolingH = allocMemory<int32_t>();
    int* poolingW = allocMemory<int32_t>();
    auto* spatialScale = allocMemory<float>();
    int* preNmsTop = allocMemory<int32_t>();
    int* nmsMaxOut = allocMemory<int32_t>();
    auto* iouThreshold = allocMemory<float>();
    auto* minBoxSize = allocMemory<float>();
    int* featureStride = allocMemory<int32_t>();
    int* anchorsRatioCount = allocMemory<int32_t>();
    int* anchorsScaleCount = allocMemory<int32_t>();
    int anchorsRatiosSize = p2.anchor_ratio_size();
    auto* anchorsRatios = allocMemory<float>(anchorsRatiosSize);
    int anchorsScalesSize = p2.anchor_scale_size();
    auto* anchorsScales = allocMemory<float>(anchorsScalesSize);

    // Intialize the plugin fields with values from the prototxt
    *poolingH = p1.pooled_h();
    f.emplace_back("poolingH", poolingH, PluginFieldType::kINT32, 1);

    *poolingW = p1.pooled_w();
    f.emplace_back("poolingW", poolingW, PluginFieldType::kINT32, 1);

    *spatialScale = p1.spatial_scale();
    f.emplace_back("spatialScale", spatialScale, PluginFieldType::kFLOAT32, 1);

    *preNmsTop = p2.prenms_top();
    f.emplace_back("preNmsTop", preNmsTop, PluginFieldType::kINT32, 1);

    *nmsMaxOut = p2.nms_max_out();
    f.emplace_back("nmsMaxOut", nmsMaxOut, PluginFieldType::kINT32, 1);

    *iouThreshold = p2.iou_threshold();
    f.emplace_back("iouThreshold", iouThreshold, PluginFieldType::kFLOAT32, 1);

    *minBoxSize = p2.min_box_size();
    f.emplace_back("minBoxSize", minBoxSize, PluginFieldType::kFLOAT32, 1);

    *featureStride = p2.feature_stride();
    f.emplace_back("featureStride", featureStride, PluginFieldType::kINT32, 1);

    *anchorsRatioCount = p2.anchor_ratio_count();
    f.emplace_back("anchorsRatioCount", anchorsRatioCount, PluginFieldType::kINT32, 1);

    *anchorsScaleCount = p2.anchor_scale_count();
    f.emplace_back("anchorsScaleCount", anchorsScaleCount, PluginFieldType::kINT32, 1);

    for (int i = 0; i < anchorsRatiosSize; ++i) {
        anchorsRatios[i] = p2.anchor_ratio(i);
}
    f.emplace_back("anchorsRatios", anchorsRatios, PluginFieldType::kFLOAT32, anchorsRatiosSize);

    for (int i = 0; i < anchorsScalesSize; ++i) {
        anchorsScales[i] = p2.anchor_scale(i);
}
    f.emplace_back("anchorsScales", anchorsScales, PluginFieldType::kFLOAT32, anchorsScalesSize);

    return f;
}

const IBlobNameToTensor* CaffeParser::parseBuffers(const uint8_t* deployBuffer,
                                                   std::size_t deployLength,
                                                   const uint8_t* modelBuffer,
                                                   std::size_t modelLength,
                                                   INetworkDefinition& network,
                                                   DataType weightType) noexcept
{
    mDeploy = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    google::protobuf::io::ArrayInputStream deployStream(deployBuffer, deployLength);
    if (!google::protobuf::TextFormat::Parse(&deployStream, mDeploy.get()))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    if (modelBuffer)
    {
        mModel = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
        google::protobuf::io::ArrayInputStream modelStream(modelBuffer, modelLength);
        google::protobuf::io::CodedInputStream codedModelStream(&modelStream);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        codedModelStream.SetTotalBytesLimit(modelLength);
#else
        // Note: This WARs the very low default size limit (64MB)
        codedModelStream.SetTotalBytesLimit(modelLength, -1);
#endif
        if (!mModel->ParseFromCodedStream(&codedModelStream))
        {
            RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
        }
    }

    return parse(network, weightType, modelBuffer != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            INetworkDefinition& network,
                                            DataType weightType) noexcept
{
    CHECK_NULL_RET_NULL(deployFile)

    // this is used to deal with dropout layers which have different input and output
    mModel = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (modelFile && !readBinaryProto(mModel.get(), modelFile, mProtobufBufferSize))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
    }

    mDeploy = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (!readTextProto(mDeploy.get(), deployFile))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    return parse(network, weightType, modelFile != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(INetworkDefinition& network,
                                            DataType weightType,
                                            bool hasModel)
{
    bool ok = true;
    CaffeWeightFactory weights(*mModel.get(), weightType, mTmpAllocs, hasModel);

    mBlobNameToTensor = new (BlobNameToTensor);

    // Get list of all available plugin creators
    int numCreators = 0;
    nvinfer1::IPluginCreator* const* tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);
    for (int k = 0; k < numCreators; ++k)
    {
        if (!tmpList[k])
        {
            std::cout << "Plugin Creator for plugin " << k << " is a nullptr." << std::endl;
            continue;
        }
        std::string pluginName = tmpList[k]->getPluginName();
        mPluginRegistry[pluginName] = tmpList[k];
    }

    for (int i = 0; i < mDeploy->input_size(); i++)
    {
        Dims dims;
        if (network.hasImplicitBatchDimension())
        {
            if (mDeploy->input_shape_size())
            {
                dims = Dims3{(int) mDeploy->input_shape().Get(i).dim().Get(1), (int) mDeploy->input_shape().Get(i).dim().Get(2), (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            }
            else
            {
                // Deprecated, but still used in a lot of networks
                dims = Dims3{(int) mDeploy->input_dim().Get(i * 4 + 1), (int) mDeploy->input_dim().Get(i * 4 + 2), (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        else
        {
            std::cout << "Warning, setting batch size to 1. Update the dimension after parsing due to using explicit batch size." << std::endl;
            if (mDeploy->input_shape_size())
            {
                dims = Dims4{1, (int) mDeploy->input_shape().Get(i).dim().Get(1), (int) mDeploy->input_shape().Get(i).dim().Get(2), (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            }
            else
            {
                // Deprecated, but still used in a lot of networks
                dims = Dims4{1, (int) mDeploy->input_dim().Get(i * 4 + 1), (int) mDeploy->input_dim().Get(i * 4 + 2), (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        ITensor* tensor = network.addInput(mDeploy->input().Get(i).c_str(), DataType::kFLOAT, dims);
        (*mBlobNameToTensor)[mDeploy->input().Get(i)] = tensor;
    }

    for (int i = 0; i < mDeploy->layer_size() && ok; i++)
    {
        const trtcaffe::LayerParameter& layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == trtcaffe::TEST)
        {
            continue;
        }

        // If there is a inplace operation and the operation is
        // modifying the input, emit an error as
        for (int j = 0; ok && j < layerMsg.top_size(); ++j)
        {
            for (int k = 0; ok && k < layerMsg.bottom_size(); ++k)
            {
                if (layerMsg.top().Get(j) == layerMsg.bottom().Get(k))
                {
                    auto iter = mBlobNameToTensor->find(layerMsg.top().Get(j).c_str());
                    if (iter != nullptr && iter->isNetworkInput())
                    {
                        ok = false;
                        std::cout << "TensorRT does not support in-place operations on input tensors in a prototxt file." << std::endl;
                    }
                }
            }
        }
        if (getInferLibVersion() >= 5000)
        {
            if (mPluginFactoryV2 && mPluginFactoryV2->isPluginV2(layerMsg.name().c_str()))
            {
                std::vector<Weights> w = weights.getAllWeights(layerMsg.name());
                nvinfer1::IPluginV2* plugin = mPluginFactoryV2->createPlugin(layerMsg.name().c_str(), w.empty() ? nullptr : &w[0], w.size(), mPluginNamespace.c_str());
                std::vector<ITensor*> inputs;
                for (int i = 0, n = layerMsg.bottom_size(); i < n; i++)
                {
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(i)]);
                }
                ILayer* layer = network.addPluginV2(&inputs[0], int(inputs.size()), *plugin);
                layer->setName(layerMsg.name().c_str());
                if (plugin->getNbOutputs() != layerMsg.top_size())
                {
                    std::cout << "Plugin layer output count is not equal to caffe output count" << std::endl;
                    ok = false;
                }
                for (int i = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); i < n; i++)
                {
                    (*mBlobNameToTensor)[layerMsg.top(i)] = layer->getOutput(i);
                }

                if (layer == nullptr)
                {
                    std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                    ok = false;
                }
                continue;
            }
            // Use the TRT5 plugin creator method to check for built-in plugin support


                std::string pluginName;
                nvinfer1::PluginFieldCollection fc;
                std::vector<nvinfer1::PluginField> f;
                if (layerMsg.type() == "Normalize")
                {
                    pluginName = "Normalize_TRT";
                    f = parseNormalizeParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "PriorBox")
                {
                    pluginName = "PriorBox_TRT";
                    f = parsePriorBoxParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "DetectionOutput")
                {
                    pluginName = "NMS_TRT";
                    f = parseDetectionOutputParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "RPROI")
                {
                    pluginName = "RPROI_TRT";
                    f = parseRPROIParam(layerMsg, weights, *mBlobNameToTensor);
                }

                if (mPluginRegistry.find(pluginName) != mPluginRegistry.end())
                {
                    // Set fc
                    fc.nbFields = f.size();
                    fc.fields = f.empty() ? nullptr : f.data();
                    nvinfer1::IPluginV2* pluginV2 = mPluginRegistry.at(pluginName)->createPlugin(layerMsg.name().c_str(), &fc);
                    assert(pluginV2);
                    mNewPlugins.push_back(pluginV2);

                    std::vector<ITensor*> inputs;
                    for (int i = 0, n = layerMsg.bottom_size(); i < n; i++)
                    {
                        inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(i)]);
                    }

                    auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), *pluginV2);
                    layer->setName(layerMsg.name().c_str());
                    if (pluginV2->getNbOutputs() != layerMsg.top_size())
                    {
                        std::cout << "Plugin layer output count is not equal to caffe output count" << std::endl;
                        ok = false;
                    }
                    for (int i = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); i < n; i++)
                    {
                        (*mBlobNameToTensor)[layerMsg.top(i)] = layer->getOutput(i);
                    }

                    if (layer == nullptr)
                    {
                        std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                        ok = false;
                    }
                    continue;
                }

        }

        if (layerMsg.type() == "Dropout")
        {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "Input")
        {
            const trtcaffe::InputParameter& p = layerMsg.input_param();
            for (int i = 0; i < layerMsg.top_size(); i++)
            {
                const trtcaffe::BlobShape& shape = p.shape().Get(i);
                if (shape.dim_size() != 4)
                {
                    RETURN_AND_LOG_ERROR(nullptr, "error parsing input layer, TensorRT only supports 4 dimensional input");
                }
                else
                {
                    Dims d;
                    if (network.hasImplicitBatchDimension())
                    {
                        d = Dims3{(int) shape.dim().Get(1), (int) shape.dim().Get(2), (int) shape.dim().Get(3)};
                    }
                    else
                    {
                        std::cout << "Warning, setting batch size to 1. Update the dimension after parsing due to "
                                     "using explicit batch size."
                                  << std::endl;
                        d = Dims4{1, (int) shape.dim().Get(1), (int) shape.dim().Get(2), (int) shape.dim().Get(3)};
                    }
                    ITensor* tensor = network.addInput(layerMsg.top(i).c_str(), DataType::kFLOAT, d);
                    (*mBlobNameToTensor)[layerMsg.top().Get(i)] = tensor;
                }
            }
            continue;
        }
        if (layerMsg.type() == "Flatten")
        {
            ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
            std::cout << "Warning: Flatten layer ignored. TensorRT implicitly"
                         " flattens input to FullyConnected layers, but in other"
                         " circumstances this will result in undefined behavior."
                      << std::endl;
            continue;
        }

        // Use parser table to lookup the corresponding parse function to handle the rest of the layers
        auto v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end())
        {
            std::cout << "could not parse layer type " << layerMsg.type() << std::endl;
            ok = false;
        }
        else
        {
            ILayer* layer = (*v->second)(network, layerMsg, weights, *static_cast<BlobNameToTensor*>(mBlobNameToTensor));
            if (layer == nullptr)
            {
                std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                ok = false;
            }
            else
            {
                layer->setName(layerMsg.name().c_str());
                (*mBlobNameToTensor)[layerMsg.top(0)] = layer->getOutput(0);
            }
        }
    }

    mBlobNameToTensor->setTensorNames();

    return ok && weights.isOK() && mBlobNameToTensor->isOK() ? mBlobNameToTensor : nullptr;
}

IBinaryProtoBlob* CaffeParser::parseBinaryProto(const char* fileName) noexcept
{
    CHECK_NULL_RET_NULL(fileName)
    using namespace google::protobuf::io;

    std::ifstream stream(fileName, std::ios::in | std::ios::binary);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not open file " + std::string{fileName});
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        codedInput.SetTotalBytesLimit(INT_MAX);
#else
        // Note: This WARs the very low default size limit (64MB)
        codedInput.SetTotalBytesLimit(INT_MAX, -1);
#endif

    trtcaffe::BlobProto blob;
    bool ok = blob.ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        RETURN_AND_LOG_ERROR(nullptr, "parseBinaryProto: Could not parse mean file");
    }

    Dims4 dims{1, 1, 1, 1};
    if (blob.has_shape())
    {
        int size = blob.shape().dim_size(), s[4] = {1, 1, 1, 1};
        for (int i = 4 - size; i < 4; i++)
        {
            assert(blob.shape().dim(i) < INT32_MAX);
            s[i] = static_cast<int>(blob.shape().dim(i));
        }
        dims = Dims4{s[0], s[1], s[2], s[3]};
    }
    else
    {
        dims = Dims4{blob.num(), blob.channels(), blob.height(), blob.width()};
    }

    const int dataSize = parserutils::volume(dims);
    assert(dataSize > 0);

    const trtcaffe::Type blobProtoDataType = CaffeWeightFactory::getBlobProtoDataType(blob);
    const auto blobProtoData = CaffeWeightFactory::getBlobProtoData(blob, blobProtoDataType, mTmpAllocs);

    if (dataSize != (int) blobProtoData.second)
    {
        std::cout << "CaffeParser::parseBinaryProto: blob dimensions don't match data size!!" << std::endl;
        return nullptr;
    }

    const int dataSizeBytes = dataSize * CaffeWeightFactory::sizeOfCaffeType(blobProtoDataType);
    void* memory = malloc(dataSizeBytes);
    memcpy(memory, blobProtoData.first, dataSizeBytes);
    return new BinaryProtoBlob(memory,
                               blobProtoDataType == trtcaffe::FLOAT ? DataType::kFLOAT : DataType::kHALF, dims);

    std::cout << "CaffeParser::parseBinaryProto: couldn't find any data!!" << std::endl;
    return nullptr;
}
