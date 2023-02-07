/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    IConsistencyChecker(IConsistencyChecker const& other) = delete;
    IConsistencyChecker& operator=(IConsistencyChecker const& other) = delete;
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
    virtual bool validate(char const* name, void const* serialData, size_t serialLength, PluginTensorDesc const* in,
        size_t nbInputs, PluginTensorDesc const* out, size_t nbOutputs, int64_t workspaceSize) const noexcept = 0;

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

extern "C" TENSORRTAPI void* createConsistencyChecker_INTERNAL(void* logger, void const* blob, size_t size,
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

inline IConsistencyChecker* createConsistencyChecker(ILogger& logger, void const* blob, size_t size)
{
    return static_cast<IConsistencyChecker*>(
        createConsistencyChecker_INTERNAL(&logger, blob, size, NV_TENSORRT_VERSION));
}

} // namespace

} // namespace consistency

} // namespace nvinfer1

#endif // NV_INFER_CONSISTENCY_H
