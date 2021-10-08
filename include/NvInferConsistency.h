/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef NV_INFER_CONSISTENCY_H
#define NV_INFER_CONSISTENCY_H

#include "NvInferConsistencyImpl.h"
#include "NvInferRuntimeCommon.h"

//!
//! \file NvInferConsistency.h
//!

namespace nvinfer1
{

namespace consistency
{

//!
//! \class IConsistencyChecker
//!
//! \brief Validates a serialized engine blob.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConsistencyChecker
{
public:
    //!
    //! \brief Check that a blob that was input to createConsistencyChecker method represents a valid engine.
    //
    //! \return true if the original blob encoded an engine that belongs to valid engine domain with
    //! target capability EngineCapability::kSAFETY, false otherwise.
    //!
    bool validate() const noexcept
    {
        return mImpl->validate();
    }

    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    virtual ~IConsistencyChecker() = default;

protected:
    apiv::VConsistencyChecker* mImpl;
    IConsistencyChecker() = default;
    IConsistencyChecker(const IConsistencyChecker& other) = delete;
    IConsistencyChecker& operator=(const IConsistencyChecker& other) = delete;
    IConsistencyChecker(IConsistencyChecker&& other) = delete;
    IConsistencyChecker& operator=(IConsistencyChecker&& other) = delete;
};

//!
//! \class IPluginChecker
//!
//! \brief Consistency Checker plugin class for user implemented Plugins.
//!
//! Plugins are a mechanism for applications to implement custom layers. It provides a
//! mechanism to register Consistency plugins and
//! look up the Plugin Registry during validate.
//!
//! Supported IPlugin inferfaces are limited to IPluginV2IOExt only.
//!
class IPluginChecker : public IPluginCreator
{
public:
    //!
    //! \brief Called during IConsistencyChecker::validate. Allows users to provide
    //! custom validation of serialized Plugin data. Returns boolean that indicates
    //! whether or not the Plugin passed validation.
    //!
    //! \param name The plugin name
    //! \param serialData The memory that holds the plugin serialized data.
    //! \param serialLength The size of the plugin serialized data.
    //! \param in The input tensors attributes.
    //! \param nbInputs The number of input tensors.
    //! \param out The output tensors attributes.
    //! \param nbOutputs The number of output tensors.
    //! \param workspaceSize The size of workspace provided during enqueue.
    //!
    virtual bool validate(const char* name, const void* serialData, size_t serialLength, const PluginTensorDesc* in,
        size_t nbInputs, const PluginTensorDesc* out, size_t nbOutputs, int64_t workspaceSize) const noexcept
        = 0;

    IPluginChecker() = default;
    virtual ~IPluginChecker() override = default;

protected:
    IPluginChecker(IPluginChecker const&) = default;
    IPluginChecker(IPluginChecker&&) = default;
    IPluginChecker& operator=(IPluginChecker const&) & = default;
    IPluginChecker& operator=(IPluginChecker&&) & = default;
};

} // namespace consistency

} // namespace nvinfer1

extern "C" TENSORRTAPI void* createConsistencyChecker_INTERNAL(void* logger, const void* blob, size_t size,
    int32_t version); //!< Internal C entry point for creating IConsistencyChecker.

namespace nvinfer1
{

namespace consistency
{
//!
//! \brief Create an instance of an IConsistencyChecker class.
//!
//! ILogger is the logging class for the consistency checker.
//!
//! anonymous namespace avoids linkage surprises when linking objects built with different versions of this header.
//!
namespace // anonymous
{

inline IConsistencyChecker* createConsistencyChecker(ILogger& logger, const void* blob, size_t size)
{
    return static_cast<IConsistencyChecker*>(
        createConsistencyChecker_INTERNAL(&logger, blob, size, NV_TENSORRT_VERSION));
}

} // namespace

} // namespace consistency

} // namespace nvinfer1

#endif // NV_INFER_CONSISTENCY_H
