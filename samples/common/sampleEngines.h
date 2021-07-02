/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_SAMPLE_ENGINES_H
#define TRT_SAMPLE_ENGINES_H

#include <iostream>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "sampleOptions.h"
#include "sampleUtils.h"

namespace sample
{

struct Parser
{
    TrtUniquePtr<nvcaffeparser1::ICaffeParser> caffeParser;
    TrtUniquePtr<nvuffparser::IUffParser> uffParser;
    TrtUniquePtr<nvonnxparser::IParser> onnxParser;

    operator bool() const
    {
        return caffeParser || uffParser || onnxParser;
    }
};

//!
//! \brief Generate a network definition for a given model
//!
//! \return Parser The parser used to initialize the network and that holds the weights for the network, or an invalid
//! parser (the returned parser converts to false if tested)
//!
//! \see Parser::operator bool()
//!
Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err);

//!
//! \brief Set up network and config
//!
//! \return boolean Return true if network and config were successfully set
//!
bool setupNetworkAndConfig(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    INetworkDefinition& network, IBuilderConfig& config, std::ostream& err,
    std::vector<std::vector<char>>& sparseWeights);

//!
//! \brief Log refittable layers and weights of a refittable engine
//!
void dumpRefittable(nvinfer1::ICudaEngine& engine);

//!
//! \brief Load a serialized engine
//!
//! \return Pointer to the engine loaded or nullptr if the operation failed
//!
nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err);

//!
//! \brief Save an engine into a file
//!
//! \return boolean Return true if the engine was successfully saved
//!
bool saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& fileName, std::ostream& err);

//!
//! \brief Create an engine from model or serialized file, and optionally save engine
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
std::tuple<TrtUniquePtr<nvinfer1::ICudaEngine>, TrtUniquePtr<INetworkDefinition>, Parser> getEngineNetworkParserTuple(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

//!
//! \brief Create an engine from model or serialized file, and optionally save engine
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
inline TrtUniquePtr<nvinfer1::ICudaEngine> getEngine(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    return std::get<0>(getEngineNetworkParserTuple(model, build, sys, err));
}

//!
//! \brief Create a serialized network
//!
//! \return Pointer to a host memory for a serialized network
//!
IHostMemory* networkToSerialized(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    INetworkDefinition& network, std::ostream& err);

//!
//! \brief Tranfer model to a serialized network
//!
//! \return Pointer to a host memory for a serialized network
//!
IHostMemory* modelToSerialized(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

//!
//! \brief Serialize network and save it into a file
//!
//! \return boolean Return true if the network was successfully serialized and saved
//!
bool serializeAndSave(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

bool timeRefit(const INetworkDefinition& network, nvinfer1::ICudaEngine& engine);

//!
//! \brief Set tensor scales from a calibration table
//!
void setTensorScalesFromCalibration(nvinfer1::INetworkDefinition& network, const std::vector<IOFormat>& inputFormats,
        const std::vector<IOFormat>& outputFormats, const std::string& calibrationFile);

} // namespace sample

#endif // TRT_SAMPLE_ENGINES_H
