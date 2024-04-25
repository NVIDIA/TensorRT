/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInfer.h"
#include "NvInferConsistency.h"
#include "NvInferSafeRuntime.h"
#include "NvOnnxParser.h"
#include "sampleOptions.h"
#include "sampleUtils.h"
#include "streamReader.h"

namespace sample
{

struct Parser
{
    std::unique_ptr<nvonnxparser::IParser> onnxParser;

    operator bool() const
    {
        return onnxParser != nullptr;
    }
};

//!
//! \brief Helper struct to faciliate engine serialization and deserialization. It does not own the underlying memory.
//!
struct EngineBlob
{
    EngineBlob(void* engineData, size_t engineSize)
        : data(engineData)
        , size(engineSize)
    {
    }
    void* data{};
    size_t size{};
    bool empty() const
    {
        return size == 0;
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
    LazilyDeserializedEngine(bool isSafe, bool versionCompatible, int32_t DLACore, std::string const& tempdir,
        nvinfer1::TempfileControlFlags tempfileControls, std::string const& leanDLLPath)
        : mIsSafe(isSafe)
        , mVersionCompatible(versionCompatible)
        , mDLACore(DLACore)
        , mTempdir(tempdir)
        , mTempfileControls(tempfileControls)
        , mLeanDLLPath(leanDLLPath)
    {
        mFileReader = std::make_unique<samplesCommon::FileStreamReader>();
    }

    //!
    //! \brief Move from another LazilyDeserializedEngine.
    //!
    LazilyDeserializedEngine(LazilyDeserializedEngine&& other) = default;

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
    EngineBlob const getBlob() const
    {
        ASSERT((!mFileReader || !mFileReader->isOpen())
            && "Attempting to access the glob when there is an open file reader!");
        if (!mEngineBlob.empty())
        {
            return EngineBlob{const_cast<void*>(static_cast<void const*>(mEngineBlob.data())), mEngineBlob.size()};
        }
        if (mEngineBlobHostMemory.get() != nullptr && mEngineBlobHostMemory->size() > 0)
        {
            return EngineBlob{mEngineBlobHostMemory->data(), mEngineBlobHostMemory->size()};
        }
        ASSERT(false && "Attempting to access an empty engine!");
        return EngineBlob{nullptr, 0};
    }

    //!
    //! \brief Set the underlying blob storing the serialized engine without duplicating IHostMemory.
    //!
    void setBlob(std::unique_ptr<nvinfer1::IHostMemory>& data)
    {
        ASSERT(data.get() && data->size() > 0);
        mEngineBlobHostMemory = std::move(data);
        mEngine.reset();
        mSafeEngine.reset();
    }

    //!
    //! \brief Set the underlying blob storing the serialized engine without duplicating vector memory.
    //!
    void setBlob(std::vector<uint8_t>&& engineBlob)
    {
        mEngineBlob = std::move(engineBlob);
        mEngine.reset();
        mSafeEngine.reset();
    }

    //!
    //! \brief Release the underlying blob without deleting the deserialized engine.
    //!
    void releaseBlob()
    {
        mEngineBlob.clear();
        mEngineBlobHostMemory.reset();
    }

    //!
    //! \brief Get the file stream reader used for deserialization
    //!
    samplesCommon::FileStreamReader& getFileReader()
    {
        ASSERT(mFileReader);
        return *mFileReader;
    }

    //!
    //! \brief Get if safe mode is enabled.
    //!
    bool isSafe()
    {
        return mIsSafe;
    }

    void setDynamicPlugins(std::vector<std::string> const& dynamicPlugins)
    {
        mDynamicPlugins = dynamicPlugins;
    }

private:
    bool mIsSafe{false};
    bool mVersionCompatible{false};
    int32_t mDLACore{-1};
    std::vector<uint8_t> mEngineBlob;
    std::unique_ptr<samplesCommon::FileStreamReader> mFileReader;

    // Directly use the host memory of a serialized engine instead of duplicating the engine in CPU memory.
    std::unique_ptr<nvinfer1::IHostMemory> mEngineBlobHostMemory;

    std::string mTempdir{};
    nvinfer1::TempfileControlFlags mTempfileControls{getTempfileControlDefaults()};
    std::string mLeanDLLPath{};
    std::vector<std::string> mDynamicPlugins;

    //! \name Owned TensorRT objects
    //! Per TensorRT object lifetime requirements as outlined in the developer guide,
    //! the runtime must remain live while any engines created by the runtime are live.
    //! DO NOT ADJUST the declaration order here: runtime -> (engine|safeEngine).
    //! Destruction occurs in reverse declaration order: (engine|safeEngine) -> runtime.
    //!@{

    //! The runtime used to track parent of mRuntime if one exists.
    //! Needed to load mRuntime if lean.so is supplied through file system path.
    std::unique_ptr<nvinfer1::IRuntime> mParentRuntime{};

    //! The runtime that is used to deserialize the engine.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{};

    //! If mIsSafe is false, this points to the deserialized std engine
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{};

    //! If mIsSafe is true, this points to the deserialized safe engine
    std::unique_ptr<nvinfer1::safe::ICudaEngine> mSafeEngine{};

    //!@}
};

struct BuildEnvironment
{
    BuildEnvironment() = delete;
    BuildEnvironment(BuildEnvironment const& other) = delete;
    BuildEnvironment(BuildEnvironment&& other) = delete;
    BuildEnvironment(bool isSafe, bool versionCompatible, int32_t DLACore, std::string const& tempdir,
        nvinfer1::TempfileControlFlags tempfileControls, std::string const& leanDLLPath = "")
        : engine(isSafe, versionCompatible, DLACore, tempdir, tempfileControls, leanDLLPath)
    {
    }

    //! \name Owned TensorRT objects
    //! Per TensorRT object lifetime requirements as outlined in the developer guide,
    //! factory objects must remain live while the objects created by those factories
    //! are live (with the exception of builder -> engine).
    //! DO NOT ADJUST the declaration order here: builder -> network -> parser.
    //! Destruction occurs in reverse declaration order: parser -> network -> builder.
    //!@{

    //! The builder used to build the engine.
    std::unique_ptr<nvinfer1::IBuilder> builder;

    //! The network used by the builder.
    std::unique_ptr<nvinfer1::INetworkDefinition> network;

    //! The parser used to specify the network.
    Parser parser;

    //! The engine.
    LazilyDeserializedEngine engine;
    //!@}
};

//!
//! \brief Set up network and config
//!
//! \return boolean Return true if network and config were successfully set
//!
bool setupNetworkAndConfig(const BuildOptions& build, const SystemOptions& sys, nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, std::ostream& err,
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
bool getEngineBuildEnv(
    ModelOptions const& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err);

//!
//! \brief Create a serialized network
//!
//! \return Pointer to a host memory for a serialized network
//!
nvinfer1::IHostMemory* networkToSerialized(const BuildOptions& build, const SystemOptions& sys,
    nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition& network, std::ostream& err);

//!
//! \brief Tranfer model to a serialized network
//!
//! \return Pointer to a host memory for a serialized network
//!
nvinfer1::IHostMemory* modelToSerialized(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

//!
//! \brief Serialize network and save it into a file
//!
//! \return boolean Return true if the network was successfully serialized and saved
//!
bool serializeAndSave(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err);

bool timeRefit(const nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool multiThreading);

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
    nvinfer1::ILogger& logger, nvinfer1::IHostMemory const* engine) noexcept;

//!
//! \brief Run consistency check on serialized engine.
//!
bool checkSafeEngine(void const* serializedEngine, int32_t const engineSize);

bool loadStreamingEngineToBuildEnv(std::string const& engine, BuildEnvironment& env, std::ostream& err);

bool loadEngineToBuildEnv(std::string const& engine, bool enableConsistency, BuildEnvironment& env, std::ostream& err);
} // namespace sample

#endif // TRT_SAMPLE_ENGINES_H
