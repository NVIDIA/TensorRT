/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInfer.h"
#if TRT_BUILD_ONNX_PARSER
#include "NvOnnxParser.h"
#endif
#include "sampleEntrypoints.h"
#include "sampleOptions.h"
#include "sampleUtils.h"
#include "streamReader.h"
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

namespace sample
{

//! \brief Callback invoked after standard builder configuration, before engine build.
//! Custom tools can use this to apply additional builder configuration on top of trtexec's.
using PostConfigCallback = std::function<void(
    nvinfer1::IBuilder&, nvinfer1::IBuilderConfig&, BuildOptions const&, SystemOptions const&)>;

#if TRT_BUILD_ONNX_PARSER
struct Parser
{
    std::unique_ptr<nvonnxparser::IParser> onnxParser;

    operator bool() const
    {
        return onnxParser != nullptr;
    }
};
#endif // TRT_BUILD_ONNX_PARSER

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
        // Only one of these is relevant for any given trtexec call.
        // Enabled using  --asyncFileReader flag.
        mAsyncFileReader = std::make_unique<samplesCommon::AsyncStreamReader>();
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

    //! \overload nvinfer1::ICudaEngine* get();
    [[nodiscard]] nvinfer1::ICudaEngine* operator->()
    {
        return this->get();
    }

    //!
    //! \brief Get the pointer to the ICudaEngine and release the ownership.
    //!
    nvinfer1::ICudaEngine* release();

    //!
    //! \brief Check Safe DLA engine built with kDLA_STANDALONE should not be run via TRT
    //!
    bool checkDLASafe();

    //!
    //! \brief Get the underlying blob storing serialized engine.
    //!
    EngineBlob const getBlob() const
    {
        ASSERT(!(mAsyncFileReader && mAsyncFileReader->isOpen())
            && "Attempting to access the glob when there is an open async file reader!");
        if (!mEngineBlob.empty())
        {
            // 'EngineBlob' is a non-owning view over a byte buffer. We intentionally avoid copying the data here.
            // 'EngineBlob' stores a non-const pointer (`void*`) for legacy reasons, but callers must treat the buffer
            // as read-only (e.g. writing to disk / passing to deserialize APIs).
            return EngineBlob{static_cast<void*>(const_cast<uint8_t*>(mEngineBlob.data())), mEngineBlob.size()};
        }
        if (mEngineBlobHostMemory != nullptr && mEngineBlobHostMemory->size() > 0)
        {
            return EngineBlob{mEngineBlobHostMemory->data(), mEngineBlobHostMemory->size()};
        }
        ASSERT(false && "Attempting to access an empty engine!");
        return EngineBlob{nullptr, 0};
    }

    //!
    //! \brief Get the underlying blob storing serialized engine if present, otherwise return an empty blob.
    //!
    //! Unlike getBlob(), this function does NOT assert if the blob is empty. This is useful for optional artifacts
    //! such as the checker blob generated via `trtexec --dumpCheckerBlob`.
    //!
    EngineBlob const getBlobOrEmpty() const
    {
        ASSERT(!(mAsyncFileReader && mAsyncFileReader->isOpen())
            && "Attempting to access the glob when there is an open async file reader!");
        if (!mEngineBlob.empty())
        {
            // NOTE: `EngineBlob` is a non-owning view over a byte buffer. We intentionally avoid copying the data here.
            // `EngineBlob` stores a non-const pointer (`void*`) for legacy reasons, but callers must treat the buffer
            // as read-only (e.g. writing to disk / passing to deserialize APIs).
            return EngineBlob{static_cast<void*>(const_cast<uint8_t*>(mEngineBlob.data())), mEngineBlob.size()};
        }
        if (mEngineBlobHostMemory != nullptr)
        {
            return EngineBlob{mEngineBlobHostMemory->data(), mEngineBlobHostMemory->size()};
        }
        return EngineBlob{nullptr, 0};
    }

    //!
    //! \brief Check whether the underlying blob is present, even if it is empty.
    //!
    [[nodiscard]] bool hasBlob() const
    {
        return !mEngineBlob.empty() || mEngineBlobHostMemory != nullptr;
    }

    //!
    //! \brief Set the underlying blob storing the serialized engine without duplicating IHostMemory.
    //!
    void setBlob(std::unique_ptr<nvinfer1::IHostMemory> data)
    {
        ASSERT(data.get() && data->size() > 0);
        mEngineBlobHostMemory = std::move(data);
        mEngine.reset();
    }

    //!
    //! \brief Set the underlying blob without duplicating IHostMemory, allowing an empty blob.
    //!
    void setBlobOrEmpty(std::unique_ptr<nvinfer1::IHostMemory> data)
    {
        ASSERT(data.get() != nullptr);
        mEngineBlobHostMemory = std::move(data);
        mEngine.reset();
    }

    //!
    //! \brief Set the underlying blob storing the serialized engine without duplicating vector memory.
    //!
    void setBlob(std::vector<uint8_t>&& engineBlob)
    {
        mEngineBlob = std::move(engineBlob);
        mEngine.reset();
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
    samplesCommon::AsyncStreamReader& getAsyncFileReader()
    {
        ASSERT(mAsyncFileReader);
        return *mAsyncFileReader;
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

#if !TRT_WINML
    //! \brief Set the DLA workspace allocation strategy used during deserialization.
    void setDLAWorkspaceAllocationStrategy(nvinfer1::DLAWorkspaceAllocationStrategy strategy)
    {
        mDLAWorkspaceAllocationStrategy = strategy;
    }
#endif // !TRT_WINML

#if TRT_WINML
    //! \brief Request that the runtime defer GPU weight allocation during deserialization.
    void setDeferredWeightsLoading(bool defer)
    {
        mDeferredWeightsLoading = defer;
    }
#endif // TRT_WINML

private:
    bool mIsSafe{false};
    bool mVersionCompatible{false};
    int32_t mDLACore{-1};
#if !TRT_WINML
    nvinfer1::DLAWorkspaceAllocationStrategy mDLAWorkspaceAllocationStrategy{
        nvinfer1::DLAWorkspaceAllocationStrategy::kDEFAULT};
#endif // !TRT_WINML
    std::vector<uint8_t> mEngineBlob;
    std::unique_ptr<samplesCommon::AsyncStreamReader> mAsyncFileReader;

    // Directly use the host memory of a serialized engine instead of duplicating the engine in CPU memory.
    std::unique_ptr<nvinfer1::IHostMemory> mEngineBlobHostMemory;

    std::string mTempdir{};
    nvinfer1::TempfileControlFlags mTempfileControls{getTempfileControlDefaults()};
    std::string mLeanDLLPath{};
    std::vector<std::string> mDynamicPlugins;
#if TRT_WINML
    bool mDeferredWeightsLoading{false};
#endif // TRT_WINML

    //! \name Owned TensorRT objects
    //! Per TensorRT object lifetime requirements as outlined in the developer guide,
    //! the runtime must remain live while any engines created by the runtime are live.
    //! DO NOT ADJUST the declaration order here: runtime -> (engine).
    //! Destruction occurs in reverse declaration order: (engine) -> runtime.
    //!@{

    //! The runtime used to track parent of mRuntime if one exists.
    //! Needed to load mRuntime if lean.so is supplied through file system path.
    std::unique_ptr<nvinfer1::IRuntime> mParentRuntime{};

    //! The runtime that is used to deserialize the engine.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{};

    //! If mIsSafe is false, this points to the deserialized std engine
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{};

    //!@}
};

struct BuildEnvironment
{
    BuildEnvironment() = delete;
    BuildEnvironment(BuildEnvironment const& other) = delete;
    BuildEnvironment(BuildEnvironment&& other) = delete;
    BuildEnvironment(bool isSafe, bool versionCompatible, int32_t DLACore, std::string const& tempdir,
        nvinfer1::TempfileControlFlags tempfileControls, std::string const& leanDLLPath = "",
        std::string const& cmdline = "")
        : engine(isSafe, versionCompatible, DLACore, tempdir, tempfileControls, leanDLLPath)
        , checkerBlob(false, false, -1, "", tempfileControls, "")
#if ENABLE_UNIFIED_BUILDER
        , companionSo(false, false, -1, "", tempfileControls, "")
#endif // ENABLE_UNIFIED_BUILDER
        , cmdline(cmdline)
    {
    }

    //! \name Owned TensorRT objects
    //! Per TensorRT object lifetime requirements as outlined in the developer guide,
    //! factory objects must remain live while the objects created by those factories
    //! are live (with the exception of builder -> engine).
    //! DO NOT ADJUST the declaration order here: builder -> builder config -> network -> parser.
    //! Destruction occurs in reverse declaration order: parser -> network -> builder config -> builder.
    //!@{

    //! The builder used to build the engine.
    std::unique_ptr<nvinfer1::IBuilder> builder;

    // Builder config used to build the engine.
    std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig;

    //! The network used by the builder.
    std::unique_ptr<nvinfer1::INetworkDefinition> network;

#if TRT_BUILD_ONNX_PARSER
    //! The parser used to specify the network.
    Parser parser;
#endif // TRT_BUILD_ONNX_PARSER

    //! The engine.
    LazilyDeserializedEngine engine;

    //! The checker blob: generated kernel sources for the kernel checker, and the per-kernel metadata
    //! the reference checker replays.
    LazilyDeserializedEngine checkerBlob;

#if ENABLE_UNIFIED_BUILDER
    //! The companion library holding the safe engine's generated host code. Loading the engine needs it, so
    //! it is saved beside the engine and handed back to the runtime at load.
    LazilyDeserializedEngine companionSo;
#endif // ENABLE_UNIFIED_BUILDER

    //! Path to the engine's companion library on disk, std::nullopt when the engine needs none. The
    //! runtime loads the library by path, so it has to exist as a file before inference.
    std::optional<std::string> companionSoPath;

    //! The command line string.
    std::string cmdline;
    //!@}
};

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
    ModelOptions const& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err, PostConfigCallback const& postConfigHook = nullptr);

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

#if TRT_BUILD_ONNX_PARSER
//!
//! \brief Refit an engine using the weights from the specified ONNX model.
//!
//! \return boolean Return true if the engine was successfully refit from the model.
//!
bool refitFromOnnx(nvinfer1::ICudaEngine& engine, std::string onnxModelFile, bool multiThreading);
#endif // TRT_BUILD_ONNX_PARSER

//!
//! \brief Refit an engine using the weights from the INetworkDefintiion and report the amount of time it took.
//!
//! \return boolean Return true if the engine was successfully refit from the INetworkDefinition.
//!
bool timeRefit(nvinfer1::INetworkDefinition const& network, nvinfer1::ICudaEngine& engine, bool multiThreading);

//! \brief Check if safe runtime is loaded.
[[nodiscard]] bool hasSafeRuntime();

//!
//! \brief Run consistency check on serialized engine.
//!
[[nodiscard]] bool checkSafeEngine(void const* serializedEngine, int64_t const engineSize,
    char const* const* pluginBuildLibs, int64_t const nbPluginBuildLibs);

//!
//! \brief Run the per-kernel reference check on a serialized safe engine.
//!
//! Confirms that the per-kernel metadata a reference check is derived from describes this engine, and
//! that the remote target described by \p remoteConfig can serve one. Mirrors checkSafeEngine.
//!
//! \param serializedEngine The serialized safe engine.
//! \param engineSize Size of \p serializedEngine in bytes.
//! \param checkerBlob The archive the build wrote beside the engine, which carries the metadata. May be
//!        empty: an engine of nothing but static library kernels needs none, and the checker reports the
//!        omission for any engine that does.
//! \param remoteConfig The remote target connection token (from --remoteConfig).
//!
[[nodiscard]] bool referenceCheckEngine(void const* serializedEngine, int64_t const engineSize,
    EngineBlob const& checkerBlob, std::string const& remoteConfig);

bool loadStreamingEngineToBuildEnv(std::string const& engine, BuildEnvironment& env, std::ostream& err);

#if TRT_WINML
//! \brief Re-read the engine file from disk and call ICudaEngine::loadWeights() to complete a
//! deferred deserialization. Used by trtexec when --deferWeightsLoading + --loadEngine are
//! combined and inference is requested.
[[nodiscard]] bool loadDeferredWeightsFromEngineFile(
    nvinfer1::ICudaEngine& engine, std::string const& filepath, std::ostream& err);
#endif // TRT_WINML

bool loadEngineToBuildEnv(std::string const& engine, BuildEnvironment& env, std::ostream& err, SystemOptions const& sys,
    bool const enableConsistency);

//!
//! \brief Load the checker blob that carries the reference-check metadata into \p env.
//!
//! Reads the path given by --loadCheckerBlob. Omitting the flag is not an error: only the checker can
//! tell an engine that never needed a blob from an engine whose blob was not supplied.
//!
//! \param build The build options, for the --loadCheckerBlob path.
//! \param env   The environment to load into.
//! \param err   Stream for diagnostics.
//! \return false if --loadCheckerBlob names a file that could not be read.
//!
[[nodiscard]] bool loadCheckerBlobToBuildEnv(BuildOptions const& build, BuildEnvironment& env, std::ostream& err);
} // namespace sample

#endif // TRT_SAMPLE_ENGINES_H
