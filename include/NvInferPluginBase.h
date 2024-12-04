/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_INFER_PLUGIN_BASE_H
#define NV_INFER_PLUGIN_BASE_H

#if !defined(NV_INFER_INTERNAL_INCLUDE)
static_assert(false, "Do not directly include this file. Include NvInferRuntime.h or NvInferPluginUtils.h");
#endif

#define NV_INFER_INTERNAL_INCLUDE 1
#include "NvInferRuntimeBase.h"
#undef NV_INFER_INTERNAL_INCLUDE
namespace nvinfer1
{

//!
//! \enum PluginFieldType
//!
//! \brief The possible field types for custom layer.
//!
enum class PluginFieldType : int32_t
{
    //! FP16 field type.
    kFLOAT16 = 0,
    //! FP32 field type.
    kFLOAT32 = 1,
    //! FP64 field type.
    kFLOAT64 = 2,
    //! INT8 field type.
    kINT8 = 3,
    //! INT16 field type.
    kINT16 = 4,
    //! INT32 field type.
    kINT32 = 5,
    //! char field type.
    kCHAR = 6,
    //! nvinfer1::Dims field type.
    kDIMS = 7,
    //! Unknown field type.
    kUNKNOWN = 8,
    //! BF16 field type.
    kBF16 = 9,
    //! INT64 field type.
    kINT64 = 10,
    //! FP8 field type.
    kFP8 = 11,
    //! INT4 field type.
    kINT4 = 12,
};

//!
//! \class PluginField
//!
//! \brief Structure containing plugin attribute field names and associated data
//! This information can be parsed to decode necessary plugin metadata
//!
//!
class PluginField
{
public:
    //! Plugin field attribute name
    AsciiChar const* name;
    //! Plugin field attribute data
    void const* data;
    //! Plugin field attribute type
    PluginFieldType type;
    //! Number of data entries in the Plugin attribute
    int32_t length;

    PluginField(AsciiChar const* const name_ = nullptr, void const* const data_ = nullptr,
        PluginFieldType const type_ = PluginFieldType::kUNKNOWN, int32_t const length_ = 0) noexcept
        : name(name_)
        , data(data_)
        , type(type_)
        , length(length_)
    {
    }
};

//!
//! \struct PluginFieldCollection
//!
//! \brief Plugin field collection struct.
//!
struct PluginFieldCollection
{
    //! Number of PluginField entries.
    int32_t nbFields{};
    //! Pointer to PluginField entries.
    PluginField const* fields{};
};

//!
//! \enum TensorRTPhase
//!
//! \brief Indicates a phase of operation of TensorRT
//!
enum class TensorRTPhase : int32_t
{
    //! Build phase of TensorRT
    kBUILD = 0,
    //! Execution phase of TensorRT
    kRUNTIME = 1
};

//!
//! \enum PluginCapabilityType
//!
//! \brief Enumerates the different capability types a IPluginV3 object may have
//!
enum class PluginCapabilityType : int32_t
{
    //! Core capability. Every IPluginV3 object must have this.
    kCORE = 0,
    //! Build capability. IPluginV3 objects provided to TensorRT build phase must have this.
    kBUILD = 1,
    //! Runtime capability. IPluginV3 objects provided to TensorRT build and execution phases must have this.
    kRUNTIME = 2
};

namespace v_1_0
{
class IPluginCapability : public IVersionedInterface
{
};

class IPluginResource : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IPluginResource", 1, 0};
    }
    //!
    //! \brief Free the underlying resource
    //!
    //! This will only be called for IPluginResource objects that were produced from IPluginResource::clone()
    //!
    //! The IPluginResource object on which release() is called must still be in a clone-able state
    //! after release() returns
    //!
    //! \return 0 for success, else non-zero
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No; this method is not required to be thread-safe
    //!
    virtual int32_t release() noexcept = 0;

    //!
    //! \brief Clone the resource object
    //!
    //! \note Resource initialization (if any) may be skipped for non-cloned objects since only clones will be
    //! registered by TensorRT
    //!
    //! \return Pointer to cloned object. nullptr if there was an issue.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual IPluginResource* clone() noexcept = 0;

    ~IPluginResource() noexcept override = default;

    IPluginResource() = default;
    IPluginResource(IPluginResource const&) = default;
    IPluginResource(IPluginResource&&) = default;
    IPluginResource& operator=(IPluginResource const&) & = default;
    IPluginResource& operator=(IPluginResource&&) & = default;
}; // class IPluginResource

class IPluginCreatorInterface : public IVersionedInterface
{
public:
    ~IPluginCreatorInterface() noexcept override = default;

protected:
    IPluginCreatorInterface() = default;
    IPluginCreatorInterface(IPluginCreatorInterface const&) = default;
    IPluginCreatorInterface(IPluginCreatorInterface&&) = default;
    IPluginCreatorInterface& operator=(IPluginCreatorInterface const&) & = default;
    IPluginCreatorInterface& operator=(IPluginCreatorInterface&&) & = default;
};

class IPluginV3 : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN", 1, 0};
    }

    //! \brief Return a pointer to plugin object implementing the specified PluginCapabilityType.
    //!
    //! \note IPluginV3 objects added for the build phase (through addPluginV3()) must return valid objects for
    //! PluginCapabilityType::kCORE, PluginCapabilityType::kBUILD and PluginCapabilityType::kRUNTIME.
    //!
    //! \note IPluginV3 objects added for the runtime phase must return valid objects for
    //! PluginCapabilityType::kCORE and PluginCapabilityType::kRUNTIME.
    //!
    //! \see TensorRTPhase
    //! \see IPluginCreatorV3One::createPlugin()
    //!
    virtual IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
    //! these parameters. The cloned object must be in a fully initialized state.
    //!
    //! \note The cloned object must return valid objects through getCapabilityInterface() for at least the same
    //! PluginCapabilityTypes as the original object.
    //!
    //! \return A cloned plugin object in an initialized state with the same parameters as the current object.
    //!         nullptr must be returned if the cloning fails.
    //!
    virtual IPluginV3* clone() noexcept = 0;
};
} // namespace v_1_0

//!
//! \class IPluginResource
//!
//! \brief Interface for plugins to define custom resources that could be shared through the plugin registry
//!
//! \see IPluginRegistry::acquirePluginResource
//! \see IPluginRegistry::releasePluginResource
//!
using IPluginResource = v_1_0::IPluginResource;

//!
//! \class IPluginCreatorInterface
//!
//! \brief Base class for all plugin creator versions.
//!
//! \see IPluginCreator and IPluginRegistry
//!
using IPluginCreatorInterface = v_1_0::IPluginCreatorInterface;

//!
//! \class IPluginV3
//!
//! \brief Plugin class for the V3 generation of user-implemented layers.
//!
//! IPluginV3 acts as a wrapper around the plugin capability interfaces that define the actual behavior of the plugin.
//!
//! \see IPluginCapability
//! \see IPluginCreatorV3One
//! \see IPluginRegistry
//!
using IPluginV3 = v_1_0::IPluginV3;

//!
//! \class IPluginCapability
//!
//! \brief Base class for plugin capability interfaces
//!
//!  IPluginCapability represents a split in TensorRT V3 plugins to sub-objects that expose different types of
//!  capabilites a plugin may have, as opposed to a single interface which defines all capabilities and behaviors of a
//!  plugin.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see PluginCapabilityType
//!
using IPluginCapability = v_1_0::IPluginCapability;
} // namespace nvinfer1

#endif /* NV_INFER_PLUGIN_BASE_H */
