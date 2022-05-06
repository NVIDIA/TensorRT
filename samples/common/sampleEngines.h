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

#ifndef TRT_SAMPLE_ENGINES_H
#define TRT_SAMPLE_ENGINES_H

#include <iostream>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferConsistency.h"
#include "NvInferSafeRuntime.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

namespace sample
{

struct Parser
{
    std::unique_ptr<nvcaffeparser1::ICaffeParser> caffeParser;
    std::unique_ptr<nvuffparser::IUffParser> uffParser;
    std::unique_ptr<nvonnxparser::IParser> onnxParser;

    operator bool() const
    {
        return caffeParser || uffParser || onnxParser;
    }
};

//!
//! \brief A helper class to hold a serialized engine (std or safe) and only deserialize it when being accessed.
//!
class LazilyDeserializedEngine
{
public:
    //!
    //! \brief Delete default constructor to make sure isSafe and DLACore are always set.
    //!
    LazilyDeserializedEngine() = delete;

    //!
    //! \brief Constructor of LazilyDeserializedEngine.
    //!
    LazilyDeserializedEngine(bool isSafe, int32_t DLACore) : mIsSafe(isSafe), mDLACore(DLACore)
    {
    }

    //!
    //! \brief Move from another LazilyDeserializedEngine.
    //!
    LazilyDeserializedEngine(LazilyDeserializedEngine&& other)
    {
        mIsSafe = other.mIsSafe;
        mDLACore = other.mDLACore;
        mEngineBlob = std::move(other.mEngineBlob);
        mEngine = std::move(other.mEngine);
        mSafeEngine = std::move(other.mSafeEngine);
    }

    //!
    //! \brief Delete copy constructor.
    //!
    LazilyDeserializedEngine(LazilyDeserializedEngine const& other) = delete;

    //!
    //! \brief Get the pointer to the ICudaEngine. Triggers deserialization if not already done so.
    //!
    nvinfer1::ICudaEngine* get();

    //!
    //! \brief Get the pointer to the ICudaEngine and release the ownership.
    //!
    nvinfer1::ICudaEngine* release();

    //!
    //! \brief Get the pointer to the safe::ICudaEngine. Triggers deserialization if not already done so.
    //!
    nvinfer1::safe::ICudaEngine* getSafe();

    //!
    //! \brief Get the underlying blob storing serialized engine.
    //!
    std::vector<uint8_t> const& getBlob() const
    {
        return mEngineBlob;
    }

    //!
    //! \brief Set the underlying blob storing serialized engine.
    //!
    void setBlob(void* data, size_t size)
    {
        mEngineBlob.resize(size);
        std::memcpy(mEngineBlob.data(), data, size);
        mEngine.reset();
        mSafeEngine.reset();
    }

    //!
    //! \brief Release the underlying blob without deleting the deserialized engine.
    //!
    void releaseBlob()
    {
        mEngineBlob.clear();
    }

    //!
    //! \brief Get if safe mode is enabled.
    //!
    bool isSafe()
    {
        return mIsSafe;
    }

private:
    bool mIsSafe{false};
    int32_t mDLACore{-1};
    std::vector<uint8_t> mEngineBlob;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::safe::ICudaEngine> mSafeEngine;
};

struct BuildEnvironment
{
    BuildEnvironment() = delete;
    BuildEnvironment(BuildEnvironment const& other) = delete;
    BuildEnvironment(BuildEnvironment&& other) = delete;
    BuildEnvironment(bool isSafe, int32_t DLACore) : engine(isSafe, DLACore)
    {
    }

    std::unique_ptr<INetworkDefinition> network;
    //! Parser that creates the network. Must be declared *after* network, so that when
    //! ~BuildEnvironment() executes, the parser is destroyed before the network is destroyed.
    Parser parser;
    LazilyDeserializedEngine engine;
};

//!
//! \brief Generate a network definition for a given model
//!
//! \return Parser The parser used to initialize the network and that holds the weights for the network, or an invalid
//! parser (the returned parser converts to false if tested)
//!
//! Constant input dimensions in the model must not be changed in the corresponding
//! network definition, because its correctness may rely on the constants.
//!
//! \see Parser::operator bool()
//!
Parser modelToNetwork(ModelOptions const& model, nvinfer1::INetworkDefinition& network, std::ostream& err);

//!
//! \brief Set up network and config
//!
//! \return boolean Return true if network and config were successfully set
//!
bool setupNetworkAndConfig(BuildOptions const& build, SystemOptions const& sys, IBuilder& builder,
    INetworkDefinition& network, IBuilderConfig& config, std::ostream& err,
    std::vector<std::vector<int8_t>>& sparseWeights);

//!
//! \brief Log refittable layers and weights of a refittable engine
//!
void dumpRefittable(nvinfer1::ICudaEngine& engine);

//!
//! \brief Load a serialized engine
//!
//! \return Pointer to the engine loaded or nullptr if the operation failed
//!
nvinfer1::ICudaEngine* loadEngine(std::string const& engine, int32_t DLACore, std::ostream& err);

//!
//! \brief Save an engine into a file
//!
//! \return boolean Return true if the engine was successfully saved
//!
bool saveEngine(nvinfer1::ICudaEngine const& engine, std::string const& fileName, std::ostream& err);

//!
//! \brief Create an engine from model or serialized file, and optionally save engine
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
bool getEngineBuildEnv(ModelOptions const& model, BuildOptions const& build, SystemOptions const& sys, 
    BuildEnvironment& env, std::ostream& err);

//!
//! \brief Measure and report refit runtime.
//!
bool timeRefit(INetworkDefinition const& network, nvinfer1::ICudaEngine& engine, bool multiThreading);

//!
//! \brief Set tensor scales from a calibration table
//!
void setTensorScalesFromCalibration(nvinfer1::INetworkDefinition& network, std::vector<IOFormat> const& inputFormats,
    std::vector<IOFormat> const& outputFormats, std::string const& calibrationFile);

//!
//! \brief Check if safe runtime is loaded.
//!
bool hasSafeRuntime();

//!
//! \brief Create a safe runtime object if the dynamic library is loaded.
//!
nvinfer1::safe::IRuntime* createSafeInferRuntime(nvinfer1::ILogger& logger) noexcept;

//!
//! \brief Check if consistency checker is loaded.
//!
bool hasConsistencyChecker();

//!
//! \brief Create a consistency checker object if the dynamic library is loaded.
//!
nvinfer1::consistency::IConsistencyChecker* createConsistencyChecker(
    nvinfer1::ILogger& logger, IHostMemory const* engine) noexcept;

//!
//! \brief Run consistency check on serialized engine.
//!
bool checkSafeEngine(void const* serializedEngine, int32_t const engineSize);
} // namespace sample

#endif // TRT_SAMPLE_ENGINES_H
