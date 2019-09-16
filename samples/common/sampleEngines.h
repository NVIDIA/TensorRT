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

#ifndef TRT_SAMPLE_ENGINES_H
#define TRT_SAMPLE_ENGINES_H

#include <iostream>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

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
//! \brief Create an engine for a network defintion
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
nvinfer1::ICudaEngine* networkToEngine(const BuildOptions& build, const SystemOptions& sys, nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, std::ostream& err);

//!
//! \brief Create an engine for a given model
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
nvinfer1::ICudaEngine* modelToEngine(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

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
TrtUniquePtr<nvinfer1::ICudaEngine> getEngine(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

} // namespace sample

#endif // TRT_SAMPLE_ENGINES_H
