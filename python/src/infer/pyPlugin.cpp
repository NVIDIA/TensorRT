/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file contains all bindings related to plugins.
#include "ForwardDeclarations.h"
#include "impl/NvInferPythonPlugin.h"
#include "infer/pyPluginDoc.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if EXPORT_ALL_BINDINGS
#include "NvInferPlugin.h"
#endif

#define PLUGIN_API_CATCH(func)                                                                                         \
    catch (std::exception const& e)                                                                                    \
    {                                                                                                                  \
        std::cerr << "[ERROR] Exception caught in " << (func) << "(): " << e.what() << std::endl;                      \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        std::cerr << "[ERROR] Exception caught in " << (func) << "()" << std::endl;                                    \
    }

#define PLUGIN_API_CATCH_CAST(func, returnType)                                                                        \
    catch (py::cast_error const& e)                                                                                    \
    {                                                                                                                  \
        std::cerr << "[ERROR] Return value of " << (func) << "() could not be interpreted as " << (returnType)         \
                  << std::endl;                                                                                        \
    }

namespace tensorrt
{
using namespace nvinfer1;
#if EXPORT_ALL_BINDINGS
using namespace nvinfer1::plugin;
#endif

constexpr PluginFieldCollection EMPTY_PLUGIN_FIELD_COLLECTION{0, nullptr};
constexpr uint32_t kTHREE_BYTE_SHIFT{24U};
constexpr uint32_t kBYTE_MASK{0xFFU};

inline PluginVersion getPluginVersion(int32_t const version)
{
    return static_cast<PluginVersion>(version >> kTHREE_BYTE_SHIFT & kBYTE_MASK);
}

class PyIDimensionExprImpl : public IDimensionExpr
{
public:
    using IDimensionExpr::IDimensionExpr;
    ~PyIDimensionExprImpl() override = default;
};

class PyIExprBuilderImpl : public IExprBuilder
{
public:
    using IExprBuilder::IExprBuilder;
    ~PyIExprBuilderImpl() override = default;
};

class PyIPluginV2DynamicExt : public IPluginV2DynamicExt
{
public:
    ~PyIPluginV2DynamicExt() override = default;
};

std::map<IPluginV2*, py::handle> pyObjVec;

class PyIPluginV2DynamicExtImpl : public PyIPluginV2DynamicExt
{
public:
    using PyIPluginV2DynamicExt::PyIPluginV2DynamicExt;
    PyIPluginV2DynamicExtImpl() = default;
    PyIPluginV2DynamicExtImpl(const PyIPluginV2DynamicExt& a){};

    int32_t getNbOutputs() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNbOutputsInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "num_outputs not initialized");
            }
            return mNbOutputs;
        }
        PLUGIN_API_CATCH("num_outputs")
        return -1;
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySupportsFormatCombination
                = utils::getOverride(static_cast<PyIPluginV2DynamicExt*>(this), "supports_format_combination");
            if (!pySupportsFormatCombination)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for supports_format_combination()");
            }

            std::vector<PluginTensorDesc> inOutVector;
            for (int32_t idx = 0; idx < nbInputs + nbOutputs; ++idx)
            {
                inOutVector.push_back(*(inOut + idx));
            }

            py::object pyResult = pySupportsFormatCombination(pos, inOutVector, nbInputs);

            try
            {
                auto result = pyResult.cast<bool>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("supports_format_combination", "bool")
            return false;
        }
        PLUGIN_API_CATCH("supports_format_combination")
        return false;
    }

    int32_t initialize() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyInitialize = py::get_override(static_cast<PyIPluginV2DynamicExt*>(this), "initialize");

            if (!pyInitialize)
            {
                // if no implementation is provided, default to empty initialize()
                return 0;
            }

            try
            {
                py::object pyResult = pyInitialize();
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from initialize() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("initialize")
        return -1;
    }

    void terminate() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyTerminate = py::get_override(static_cast<PyIPluginV2DynamicExt*>(this), "terminate");

            // if no implementation is provided for terminate(), it is defaulted to `pass`
            if (pyTerminate)
            {
                pyTerminate();
            }
        }
        PLUGIN_API_CATCH("terminate")
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyEnqueue = utils::getOverride(static_cast<PyIPluginV2DynamicExt*>(this), "enqueue");
            if (!pyEnqueue)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for enqueue()");
            }

            std::vector<PluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inVector.push_back(*(inputDesc + idx));
            }
            std::vector<PluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outVector.push_back(*(outputDesc + idx));
            }

            std::vector<intptr_t> inPtrs;
            for (int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inPtrs.push_back(reinterpret_cast<intptr_t>(inputs[idx]));
            }
            std::vector<intptr_t> outPtrs;
            for (int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outPtrs.push_back(reinterpret_cast<intptr_t>(outputs[idx]));
            }

            intptr_t workspacePtr = reinterpret_cast<intptr_t>(workspace);
            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);

            try
            {
                pyEnqueue(inVector, outVector, inPtrs, outPtrs, workspacePtr, cudaStreamPtr);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from enqueue() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("enqueue")
        return -1;
    }

    size_t getSerializationSize() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetSerializationSize
                = py::get_override(static_cast<PyIPluginV2DynamicExt const*>(this), "get_serialization_size");

            if (!pyGetSerializationSize)
            {
                // if no implementation is provided for get_serialization_size(), default to len(serialize())
                py::gil_scoped_acquire gil{};
                py::function pySerialize
                    = utils::getOverride(static_cast<PyIPluginV2DynamicExt const*>(this), "serialize");
                if (!pySerialize)
                {
                    utils::throwPyError(PyExc_RuntimeError, "no implementation provided for serialize()");
                }

                py::object pyResult = pySerialize();

                try
                {
                    std::string pyResultString = pyResult.cast<std::string>();
                    return pyResultString.size();
                }
                PLUGIN_API_CATCH_CAST("serialize", "std::string")
                return 0;
            }

            py::object pyResult = pyGetSerializationSize();

            try
            {
                auto result = pyResult.cast<size_t>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_serialization_size", "size_t")
            return 0;
        }
        PLUGIN_API_CATCH("get_serialization_size")
        return 0;
    }

    void serialize(void* buffer) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySerialize = utils::getOverride(static_cast<PyIPluginV2DynamicExt const*>(this), "serialize");
            if (!pySerialize)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for serialize()");
            }

            py::object pyResult = pySerialize();

            try
            {
                std::string pyResultString = pyResult.cast<std::string>();
                std::memcpy(buffer, pyResultString.data(), getSerializationSize());
            }
            PLUGIN_API_CATCH_CAST("serialize", "std::string")
        }
        PLUGIN_API_CATCH("serialize")
    }

    char const* getPluginType() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginTypeInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_type not initialized");
            }
            return mPluginType.c_str();
        }
        PLUGIN_API_CATCH("plugin_type")
        return nullptr;
    }

    char const* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginVersionInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyClone = utils::getOverride(static_cast<const PyIPluginV2DynamicExt*>(this), "clone");
            if (!pyClone)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for clone()");
            }

            py::handle handle = pyClone().release();

            try
            {
                auto result = handle.cast<PyIPluginV2DynamicExt*>();
                pyObjVec[result] = handle;
                return result;
            }
            PLUGIN_API_CATCH_CAST("clone", "nvinfer1::IPluginV2DynamicExt")
            return nullptr;
        }
        PLUGIN_API_CATCH("clone")
        return nullptr;
    }

    void destroy() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyDestroy = py::get_override(static_cast<const PyIPluginV2DynamicExt*>(this), "destroy");

            if (pyDestroy)
            {
                pyDestroy();
            }

            // Remove reference to the Python plugin object so that it could be garbage-collected
            pyObjVec[this].dec_ref();
        }
        PLUGIN_API_CATCH("destroy")
    }

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        // setPluginNamespace() is not passed through to the Python side
        std::string libNamespaceStr{libNamespace};
        mNamespace = std::move(libNamespaceStr);
        mIsNamespaceInitialized = true;
    }

    const char* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            // getPluginNamespace() is not passed through to the Python side
            if (!mIsNamespaceInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mNamespace.c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputDataType
                = utils::getOverride(static_cast<PyIPluginV2DynamicExt const*>(this), "get_output_datatype");
            if (!pyGetOutputDataType)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_datatype()");
            }

            std::vector<DataType> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputTypes + idx));
            }

            py::object pyResult = pyGetOutputDataType(index, inVector);

            try
            {
                auto result = pyResult.cast<DataType>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_output_datatype", "nvinfer1::DataType")
            return DataType{};
        }
        PLUGIN_API_CATCH("get_output_datatype")
        return DataType{};
    }

    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputDimensions
                = utils::getOverride(static_cast<PyIPluginV2DynamicExt*>(this), "get_output_dimensions");
            if (!pyGetOutputDimensions)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_dimensions()");
            }

            std::vector<DimsExprs> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            py::object pyResult = pyGetOutputDimensions(outputIndex, inVector, &exprBuilder);

            try
            {
                auto result = pyResult.cast<DimsExprs>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_output_dimensions", "nvinfer1::DimsExprs")
            return DimsExprs{};
        }
        PLUGIN_API_CATCH("get_output_dimensions")
        return DimsExprs{};
    }

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        try
        {
            mNbInputs = nbInputs;

            py::gil_scoped_acquire gil{};

            py::function pyConfigurePlugin
                = utils::getOverride(static_cast<PyIPluginV2DynamicExt*>(this), "configure_plugin");
            if (!pyConfigurePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for configure_plugin()");
            }

            std::vector<DynamicPluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(in + idx));
            }

            std::vector<DynamicPluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(out + idx));
            }

            pyConfigurePlugin(inVector, outVector);
        }
        PLUGIN_API_CATCH("configure_plugin")
    }

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetWorkspaceSize
                = py::get_override(static_cast<PyIPluginV2DynamicExt const*>(this), "get_workspace_size");

            if (!pyGetWorkspaceSize)
            {
                // if no implementation is provided for get_workspace_size(), default to zero workspace size required
                return 0;
            }

            std::vector<PluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            std::vector<PluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(outputs + idx));
            }

            py::object pyResult = pyGetWorkspaceSize(inVector, outVector);

            try
            {
                auto result = pyResult.cast<size_t>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_workspace_size", "size_t")
            return 0;
        }
        PLUGIN_API_CATCH("get_workspace_size")
        return 0;
    }

    void setNbOutputs(int32_t nbOutputs)
    {
        mNbOutputs = nbOutputs;
        mIsNbOutputsInitialized = true;
    }

    void setPluginType(std::string pluginType)
    {
        mPluginType = std::move(pluginType);
        mIsPluginTypeInitialized = true;
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
        mIsPluginVersionInitialized = true;
    }

private:
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_DYNAMICEXT_PYTHON) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

    int32_t mNbInputs{};
    int32_t mNbOutputs{};
    std::string mNamespace;
    std::string mPluginType;
    std::string mPluginVersion;

    bool mIsNbOutputsInitialized{false};
    bool mIsNamespaceInitialized{false};
    bool mIsPluginTypeInitialized{false};
    bool mIsPluginVersionInitialized{false};
};

class IPluginCreatorImpl : public IPluginCreator
{
public:
    IPluginCreatorImpl() = default;

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    AsciiChar const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNameInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "name not initialized");
            }
            return mName.c_str();
        }
        PLUGIN_API_CATCH("name")
        return nullptr;
    }

    const char* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginVersionInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsFCInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "field_names not initialized");
            }
            return &mFC;
        }
        PLUGIN_API_CATCH("field_names")
        return nullptr;
    }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyCreatePlugin = utils::getOverride(static_cast<IPluginCreator*>(this), "create_plugin");
            if (!pyCreatePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for create_plugin()");
            }

            std::string nameString{name};

            py::handle handle = pyCreatePlugin(nameString, fc).release();
            try
            {
                auto result = handle.cast<IPluginV2*>();
                pyObjVec[result] = handle;
                return result;
            }
            PLUGIN_API_CATCH_CAST("create_plugin", "IPluginV2*")
            return nullptr;
        }
        PLUGIN_API_CATCH("create_plugin")
        return nullptr;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyDeserializePlugin
                = utils::getOverride(static_cast<IPluginCreator*>(this), "deserialize_plugin");
            if (!pyDeserializePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for deserialize_plugin()");
            }

            std::string nameString{name};

            py::handle handle
                = pyDeserializePlugin(nameString, py::bytes(static_cast<const char*>(serialData), serialLength))
                      .release();
            try
            {
                auto result = handle.cast<IPluginV2*>();
                pyObjVec[result] = handle;
                return result;
            }
            PLUGIN_API_CATCH_CAST("deserialize_plugin", "IPluginV2*")
            return nullptr;
        }
        PLUGIN_API_CATCH("deserialize_plugin")
        return nullptr;
    }

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        std::string libNamespaceStr{libNamespace};
        mNamespace = std::move(libNamespaceStr);
        mIsNamespaceInitialized = true;
    }

    const char* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNamespaceInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mNamespace.c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    void setFieldNames(PluginFieldCollection fc)
    {
        mFC = fc;
        mIsFCInitialized = true;
    }

    void setPluginName(std::string name)
    {
        mName = std::move(name);
        mIsNameInitialized = true;
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
        mIsPluginVersionInitialized = true;
    }

private:
    nvinfer1::PluginFieldCollection mFC{};
    std::string mNamespace;
    std::string mName;
    std::string mPluginVersion;

    bool mIsFCInitialized{false};
    bool mIsNamespaceInitialized{false};
    bool mIsNameInitialized{false};
    bool mIsPluginVersionInitialized{false};
};

class PyIPluginV3Impl : public IPluginV3
{
public:
    using IPluginV3::IPluginV3;
    PyIPluginV3Impl() = default;
    PyIPluginV3Impl(const IPluginV3& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetCapabilityInterface
                = utils::getOverride(static_cast<const IPluginV3*>(this), "get_capability_interface");
            if (!pyGetCapabilityInterface)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_capability_interface()");
            }

            auto pyResult = pyGetCapabilityInterface(type).release();

            try
            {
                if (type == PluginCapabilityType::kCORE)
                {
                    try
                    {
                        return pyResult.cast<IPluginV3OneCore*>();
                    }
                    catch (py::cast_error const& e)
                    {
                        try
                        {
                            return pyResult.cast<IPluginV3QuickCore*>();
                        }
                        PLUGIN_API_CATCH_CAST("get_capability_interface", " a valid core capability interface")
                    }
                }
                if (type == PluginCapabilityType::kBUILD)
                {
                    try
                    {
                        return pyResult.cast<IPluginV3OneBuildV2*>();
                    }
                    catch (py::cast_error const& e)
                    {
                        try
                        {
                            return pyResult.cast<IPluginV3OneBuild*>();
                        }
                        catch (py::cast_error const& e)
                        {
                            try
                            {
                                return pyResult.cast<IPluginV3QuickAOTBuild*>();
                            }
                            catch (py::cast_error const& e)
                            {
                                try
                                {
                                    return pyResult.cast<IPluginV3QuickBuild*>();
                                }
                                PLUGIN_API_CATCH_CAST("get_capability_interface", " a valid build capability interface")
                            }
                        }
                    }
                }
                if (type == PluginCapabilityType::kRUNTIME)
                {
                    try
                    {
                        return pyResult.cast<IPluginV3OneRuntime*>();
                    }
                    catch (py::cast_error const& e)
                    {
                        try
                        {
                            return pyResult.cast<IPluginV3QuickRuntime*>();
                        }
                        PLUGIN_API_CATCH_CAST("get_capability_interface", " a valid runtime capability interface")
                    }
                }
            }
            PLUGIN_API_CATCH_CAST("get_capability_interface", "nvinfer1::IPluginCapability")
            return nullptr;
        }
        PLUGIN_API_CATCH("get_capability_interface")
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyClone = utils::getOverride(static_cast<const IPluginV3*>(this), "clone");
            if (!pyClone)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for clone()");
            }

            // Release so that pybind11 does not manage lifetime anymore
            // We will manually decrement ref count in the destructor so that Python could garbage collect
            py::handle handle = pyClone().release();

            try
            {
                return handle.cast<IPluginV3*>();
            }
            PLUGIN_API_CATCH_CAST("clone", "nvinfer1::IPluginV3")
            return nullptr;
        }
        PLUGIN_API_CATCH("clone")
        return nullptr;
    }

    ~PyIPluginV3Impl() override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyDestroy = py::get_override(static_cast<IPluginV3 const*>(this), "destroy");

            if (pyDestroy)
            {
                pyDestroy();
            }

            // If ref_count is > 1 at this point, then this is guaranteed to be a plugin instance
            // which was release()'d (e.g. after being clone()'d) (lifetime managed by TRT)
            // So only dec_ref() for those plugin instances
            // Ref counts for others will be automatically managed by Python
            auto obj = py::cast(this);
            if (obj.ref_count() > 1)
            {
                obj.dec_ref();
            }
        }
        PLUGIN_API_CATCH("destroy")
    }
};

class PyIPluginResourceImpl : public IPluginResource
{
public:
    using IPluginResource::IPluginResource;
    PyIPluginResourceImpl() = default;
    PyIPluginResourceImpl(const IPluginResource& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    int32_t release() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyRelease = utils::getOverride(static_cast<IPluginResource const*>(this), "release");

            if (!pyRelease)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for release()");
            }

            try
            {
                pyRelease();
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from release() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("release")
        return -1;
    }

    IPluginResource* clone() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyClone = utils::getOverride(static_cast<IPluginResource const*>(this), "clone");

            if (!pyClone)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for clone()");
            }

            try
            {
                auto handle = pyClone().release();
                try
                {
                    return handle.cast<IPluginResource*>();
                }
                PLUGIN_API_CATCH_CAST("clone", "nvinfer1::IPluginResource")
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from clone() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("clone")
        return nullptr;
    }

    ~PyIPluginResourceImpl() override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            auto obj = py::cast(this);
            // The cloned resource (which is registered by TRT) may have many references on the Python
            // side, including at call sites of IPluginRegistry.acquire_plugin_resource().
            // But even though IPluginRegistry.release_plugin_resource() internally decrements the ref count,
            // this is not reflected on the Python side. Therefore, when the registered resource is manually
            // deleted by TRT, set the ref count to zero so the object may be properly garbage collected by
            // Python.
            while (obj.ref_count())
            {
                obj.dec_ref();
            }
        }
        PLUGIN_API_CATCH("IPluginResource destruction")
    }
};

template <class T>
class PyIPluginV3OneBuildBaseImpl : public T
{
private:
    T* mBuild{nullptr};

protected:
    PyIPluginV3OneBuildBaseImpl(T* build)
        : mBuild{build}
    {
    }

    PyIPluginV3OneBuildBaseImpl(T const* build)
        : mBuild{const_cast<T*>(build)}
    {
    }

public:
    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    int32_t getNbOutputs() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNbOutputsInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "num_outputs not initialized");
            }
            return mNbOutputs;
        }
        PLUGIN_API_CATCH("num_outputs")
        return -1;
    }

    int32_t getNbTactics() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            try
            {
                py::function pyGetValidTactics = py::get_override(static_cast<T const*>(mBuild), "get_valid_tactics");

                mIsTacticsInitialized = true;

                if (!pyGetValidTactics)
                {
                    // if no implementation is provided for get_valid_tactics(), communicate that no custom tactics are
                    // used by the plugin
                    return 0;
                }

                py::object pyResult = pyGetValidTactics();
                mTactics = pyResult.cast<std::vector<int32_t>>();
                return static_cast<int32_t>(mTactics.size());
            }
            PLUGIN_API_CATCH_CAST("get_valid_tactics", "std::vector<int32_t>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_valid_tactics() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("tactics")
        return -1;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            try
            {
                // getValidTactics() must immediately follow getNbTactics()
                // because it is impossible to call getValidTactics() without knowing the
                // correct number of tactics. So check that mIsTacticsInitialized is true.
                // Otherwise, something has gone wrong.
                if (mIsTacticsInitialized)
                {
                    // Unset to catch any subsequent violations
                    mIsTacticsInitialized = false;
                    if (nbTactics != static_cast<int32_t>(mTactics.size()))
                    {
                        utils::throwPyError(
                            PyExc_RuntimeError, "number of tactics does not match cached number of tactics");
                    }
                    std::copy(mTactics.begin(), mTactics.end(), tactics);
                    return 0;
                }
                else
                {
                    utils::throwPyError(
                        PyExc_RuntimeError, "Internal error. getValidTactics() called before getNbTactics().");
                }
                return -1;
            }
            PLUGIN_API_CATCH_CAST("get_valid_tactics", "std::vector<int32_t>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_valid_tactics() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("tactics")
        return -1;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySupportsFormatCombination
                = utils::getOverride(static_cast<T*>(mBuild), "supports_format_combination");
            if (!pySupportsFormatCombination)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for supports_format_combination()");
            }

            std::vector<DynamicPluginTensorDesc> inOutVector;
            for (int32_t idx = 0; idx < nbInputs + nbOutputs; ++idx)
            {
                inOutVector.push_back(*(inOut + idx));
            }

            py::object pyResult = pySupportsFormatCombination(pos, inOutVector, nbInputs);

            try
            {
                auto result = pyResult.cast<bool>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("supports_format_combination", "bool")
            return false;
        }
        PLUGIN_API_CATCH("supports_format_combination")
        return false;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, const DataType* inputTypes, int32_t nbInputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputDataTypes
                = utils::getOverride(static_cast<T const*>(mBuild), "get_output_data_types");
            if (!pyGetOutputDataTypes)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_data_types()");
            }

            std::vector<DataType> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputTypes + idx));
            }

            try
            {
                py::object pyResult = pyGetOutputDataTypes(inVector);
                auto result = pyResult.cast<std::vector<DataType>>();

                if (static_cast<int32_t>(result.size()) != nbOutputs)
                {
                    utils::throwPyError(PyExc_RuntimeError,
                        "get_output_data_types() returned a list with a different length than num_outputs");
                }

                std::copy(result.begin(), result.end(), outputTypes);
                return 0;
            }
            PLUGIN_API_CATCH_CAST("get_output_data_types", "std::vector<nvinfer1::DataType>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_output_data_types() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("get_output_data_types")
        return -1;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputShapes = utils::getOverride(static_cast<T*>(mBuild), "get_output_shapes");
            if (!pyGetOutputShapes)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_shapes()");
            }

            std::vector<DimsExprs> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            std::vector<DimsExprs> shapeInVector;
            for (int32_t idx = 0; idx < nbShapeInputs; ++idx)
            {
                shapeInVector.push_back(*(shapeInputs + idx));
            }

            py::object pyResult = pyGetOutputShapes(inVector, shapeInVector, &exprBuilder);

            try
            {
                auto result = pyResult.cast<std::vector<DimsExprs>>();
                if (static_cast<int32_t>(result.size()) != nbOutputs)
                {
                    utils::throwPyError(PyExc_RuntimeError,
                        "get_output_shapes() returned a list with a different length than num_outputs");
                }
                std::copy(result.begin(), result.end(), outputs);
                return 0;
            }
            PLUGIN_API_CATCH_CAST("get_output_shapes", "std::vector<nvinfer1::DimsExprs>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_output_shapes() " << e.what() << std::endl;
            }
            return -1;
        }
        PLUGIN_API_CATCH("get_output_shapes")
        return -1;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyConfigurePlugin = utils::getOverride(static_cast<T*>(mBuild), "configure_plugin");

            if (!pyConfigurePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for configure_plugin()");
            }

            std::vector<DynamicPluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(in + idx));
            }

            std::vector<DynamicPluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(out + idx));
            }

            try
            {
                pyConfigurePlugin(inVector, outVector);
                return 0;
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from configure_plugin() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("configure_plugin")
        return -1;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetWorkspaceSize = py::get_override(static_cast<T const*>(mBuild), "get_workspace_size");

            if (!pyGetWorkspaceSize)
            {
                // if no implementation is provided for get_workspace_size(), default to zero workspace size required
                return 0U;
            }

            std::vector<DynamicPluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            std::vector<DynamicPluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(outputs + idx));
            }

            py::object pyResult = pyGetWorkspaceSize(inVector, outVector);

            try
            {
                auto result = pyResult.cast<size_t>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_workspace_size", "size_t")
            return 0U;
        }
        PLUGIN_API_CATCH("get_workspace_size")
        return 0U;
    }

    char const* getTimingCacheID() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsTimingCachedIdInitialized)
            {
                return nullptr;
            }
            return mTimingCachedId.c_str();
        }
        PLUGIN_API_CATCH("timing_cache_id")
        return nullptr;
    }

    char const* getMetadataString() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsMetadataStringInitialized)
            {
                return nullptr;
            }
            return mMetadataString.c_str();
        }
        PLUGIN_API_CATCH("metadata_string")
        return nullptr;
    }

    int32_t getFormatCombinationLimit() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsFormatCombinationLimitInitialized)
            {
                return IPluginV3OneBuild::kDEFAULT_FORMAT_COMBINATION_LIMIT;
            }
            return mFormatCombinationLimit;
        }
        PLUGIN_API_CATCH("format_combination_limit")
        return -1;
    }

    void setNbOutputs(int32_t nbOutputs)
    {
        mNbOutputs = nbOutputs;
        mIsNbOutputsInitialized = true;
    }

    void setFormatCombinationLimit(int32_t formatCombinationLimit)
    {
        mFormatCombinationLimit = formatCombinationLimit;
        mIsFormatCombinationLimitInitialized = true;
    }

    void setTimingCachedId(std::string timingCachedId)
    {
        mTimingCachedId = std::move(timingCachedId);
        mIsTimingCachedIdInitialized = true;
    }

    void setMetadataString(std::string metadataString)
    {
        mMetadataString = std::move(metadataString);
        mIsMetadataStringInitialized = true;
    }

private:
    int32_t mNbOutputs{};
    int32_t mFormatCombinationLimit{};
    std::string mTimingCachedId{};
    std::string mMetadataString{};
    std::vector<int32_t> mTactics;

    bool mIsNbOutputsInitialized{false};
    bool mIsTimingCachedIdInitialized{false};
    bool mIsFormatCombinationLimitInitialized{false};
    bool mIsMetadataStringInitialized{false};
    bool mIsTacticsInitialized{false};
};

class PyIPluginV3OneBuildImpl : public PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuild>
{
public:
    PyIPluginV3OneBuildImpl()
        : PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuild>(this)
    {
    }
    PyIPluginV3OneBuildImpl(IPluginV3OneBuild const& a)
        : PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuild>(this){};
};

class PyIPluginV3OneBuildV2Impl : public PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuildV2>
{
public:
    PyIPluginV3OneBuildV2Impl()
        : PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuildV2>(this)
    {
    }
    PyIPluginV3OneBuildV2Impl(IPluginV3OneBuildV2 const& a)
        : PyIPluginV3OneBuildBaseImpl<IPluginV3OneBuildV2>(this){};

    int32_t getAliasedInput(int32_t outputIndex) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetAliasedInput
                = py::get_override(static_cast<IPluginV3OneBuildV2*>(this), "get_aliased_input");

            if (!pyGetAliasedInput)
            {
                // if no implementation is provided for get_aliased_input(), default to no aliasing
                return -1;
            }

            py::object pyResult = pyGetAliasedInput(outputIndex);

            try
            {
                auto result = pyResult.cast<int32_t>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_aliased_input", "int32_t")
            return -1;
        }
        PLUGIN_API_CATCH("get_aliased_input")
        return -1;
    }
};

class PyIPluginV3QuickCoreImpl : public IPluginV3QuickCore
{
public:
    using IPluginV3QuickCore::IPluginV3QuickCore;
    PyIPluginV3QuickCoreImpl() = default;
    PyIPluginV3QuickCoreImpl(const IPluginV3QuickCore& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    char const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mPluginName.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_name not initialized");
            }
            return mPluginName.value().c_str();
        }
        PLUGIN_API_CATCH("plugin_name")
        return nullptr;
    }

    char const* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mPluginVersion.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.value().c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    char const* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            // getPluginNamespace() is not passed through to the Python side
            if (!mPluginNamespace.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mPluginNamespace.value().c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    void setPluginName(std::string pluginName)
    {
        mPluginName = std::move(pluginName);
    }

    void setPluginNamespace(std::string pluginNamespace)
    {
        mPluginNamespace = std::move(pluginNamespace);
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
    }

private:
    std::optional<std::string> mPluginNamespace;
    std::optional<std::string> mPluginName;
    std::optional<std::string> mPluginVersion;
};

template <class T>
class PyIPluginV3QuickBuildBaseImpl : public T
{
private:
    T* mBuild{nullptr};

protected:
    PyIPluginV3QuickBuildBaseImpl(T* build)
        : mBuild{build}
    {
    }

    PyIPluginV3QuickBuildBaseImpl(T const* build)
        : mBuild{const_cast<T*>(build)}
    {
    }

public:
    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    int32_t getNbOutputs() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            try
            {
                py::function pyNbTactics = py::get_override(static_cast<T const*>(mBuild), "get_num_outputs");

                if (!pyNbTactics)
                {
                    // if no implementation is provided for get_num_outputs(), communicate that no custom tactics are
                    // used by the plugin
                    return 0;
                }

                py::object pyResult = pyNbTactics();
                return pyResult.cast<int32_t>();
                ;
            }
            PLUGIN_API_CATCH_CAST("get_num_outputs", "int32_t")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_num_outputs() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("get_num_outputs")
        return -1;
    }

    int32_t getNbTactics() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            try
            {
                py::function pyGetValidTactics = py::get_override(static_cast<T const*>(mBuild), "get_valid_tactics");

                if (!pyGetValidTactics)
                {
                    // if no implementation is provided for get_valid_tactics(), communicate that no custom tactics are
                    // used by the plugin
                    return 0;
                }

                py::object pyResult = pyGetValidTactics();
                mTactics = pyResult.cast<std::vector<int32_t>>();
                return static_cast<int32_t>(mTactics.value().size());
            }
            PLUGIN_API_CATCH_CAST("get_valid_tactics", "std::vector<int32_t>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_valid_tactics() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("tactics")
        return -1;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            try
            {
                // getValidTactics() must immediately follow getNbTactics()
                // because it is impossible to call getValidTactics() without knowing the
                // correct number of tactics. So check that mTactics.has_value() is true.
                // Otherwise, something has gone wrong.
                if (mTactics.has_value())
                {
                    if (nbTactics != static_cast<int32_t>(mTactics.value().size()))
                    {
                        utils::throwPyError(
                            PyExc_RuntimeError, "number of tactics does not match cached number of tactics");
                    }
                    std::copy(mTactics.value().begin(), mTactics.value().end(), tactics);
                    // Reset to catch any subsequent violations
                    mTactics.reset();
                    return 0;
                }
                else
                {
                    utils::throwPyError(
                        PyExc_RuntimeError, "Internal error. getValidTactics() called before getNbTactics().");
                }
                return -1;
            }
            PLUGIN_API_CATCH_CAST("get_valid_tactics", "std::vector<int32_t>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_valid_tactics() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("tactics")
        return -1;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyConfigurePlugin = utils::getOverride(static_cast<T*>(mBuild), "configure_plugin");

            if (!pyConfigurePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for configure_plugin()");
            }

            std::vector<DynamicPluginTensorDesc> inVector;
            std::vector<DynamicPluginTensorDesc> outVector;
            std::copy_n(in, nbInputs, std::back_inserter(inVector));
            std::copy_n(out, nbOutputs, std::back_inserter(outVector));

            try
            {
                pyConfigurePlugin(inVector, outVector);
                return 0;
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from configure_plugin() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("configure_plugin")
        return -1;
    }

    int32_t getNbSupportedFormatCombinations(
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySupportsFormatCombination
                = utils::getOverride(static_cast<T*>(mBuild), "get_supported_format_combinations");
            if (!pySupportsFormatCombination)
            {
                utils::throwPyError(
                    PyExc_RuntimeError, "no implementation provided for get_supported_format_combinations()");
            }

            std::vector<DynamicPluginTensorDesc> inOutVector;
            std::copy_n(inOut, nbInputs + nbOutputs, std::back_inserter(inOutVector));

            py::object pyResult = pySupportsFormatCombination(inOutVector, nbInputs);
            try
            {
                mSupportedFormatCombinations = pyResult.cast<std::vector<PluginTensorDesc>>();
                if (static_cast<int32_t>(mSupportedFormatCombinations.value().size()) % (nbInputs + nbOutputs) != 0)
                {
                    utils::throwPyError(
                        PyExc_ValueError, "Number of supported format combinations not a multiple of number of IO.");
                }
                return static_cast<int32_t>(mSupportedFormatCombinations.value().size()) / (nbInputs + nbOutputs);
            }
            PLUGIN_API_CATCH_CAST("get_nb_supported_format_combinations", "int32_t")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_supported_format_combinations() " << e.what()
                          << std::endl;
            }
            return -1;
        }
        PLUGIN_API_CATCH("get_nb_supported_format_combinations")
        return -1;
    }

    int32_t getSupportedFormatCombinations(DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs,
        PluginTensorDesc* supportedCombinations, int32_t nbFormatCombinations) noexcept override
    {
        py::gil_scoped_acquire gil{};

        py::function pySupportsFormatCombination
            = utils::getOverride(static_cast<T*>(mBuild), "get_supported_format_combinations");
        if (!pySupportsFormatCombination)
        {
            utils::throwPyError(
                PyExc_RuntimeError, "no implementation provided for get_supported_format_combinations()");
        }

        std::vector<DynamicPluginTensorDesc> inOutVector;
        std::copy_n(inOut, nbInputs + nbOutputs, std::back_inserter(inOutVector));

        py::object pyResult = pySupportsFormatCombination(inOutVector, nbInputs);

        try
        {
            // getSupportedFormatCombinations() must immediately follow getNbSupportedFormatCombinations()
            // because it is impossible to call getSupportedFormatCombinations() without knowing the
            // correct number of tactics. So check that mSupportedFormatCombinations.has_value().
            // Otherwise, something has gone wrong.
            if (mSupportedFormatCombinations.has_value())
            {
                std::copy(mSupportedFormatCombinations.value().begin(), mSupportedFormatCombinations.value().end(),
                    supportedCombinations);
                // Reset to catch any subsequent violations
                mSupportedFormatCombinations.reset();
                return 0;
            }
            else
            {
                utils::throwPyError(PyExc_RuntimeError,
                    "Internal error. getSupportedFormatCombinations() called before "
                    "getNbSupportedFormatCombinations().");
            }
            return -1;
        }
        PLUGIN_API_CATCH_CAST("get_supported_format_combinations", "std::vector<PluginTensorDesc>")
        catch (py::error_already_set& e)
        {
            std::cerr << "[ERROR] Exception thrown from get_supported_format_combinations() " << e.what() << std::endl;
        }
        return -1;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
        int32_t const* inputRanks, int32_t nbInputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputDataTypes
                = utils::getOverride(static_cast<T const*>(mBuild), "get_output_data_types");
            if (!pyGetOutputDataTypes)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_data_types()");
            }

            std::vector<DataType> inVector;
            std::vector<int32_t> ranksVector;
            std::copy_n(inputTypes, nbInputs, std::back_inserter(inVector));
            std::copy_n(inputRanks, nbInputs, std::back_inserter(ranksVector));

            try
            {
                py::object pyResult = pyGetOutputDataTypes(inVector, ranksVector);
                auto result = pyResult.cast<std::vector<DataType>>();

                if (static_cast<int32_t>(result.size()) != nbOutputs)
                {
                    utils::throwPyError(PyExc_RuntimeError,
                        "get_output_data_types() returned a list with a different length than num_outputs");
                }

                std::copy(result.begin(), result.end(), outputTypes);
                return 0;
            }
            PLUGIN_API_CATCH_CAST("get_output_data_types", "std::vector<nvinfer1::DataType>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_output_data_types() " << e.what() << std::endl;
            }
        }
        PLUGIN_API_CATCH("get_output_data_types")
        return -1;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetOutputShapes = utils::getOverride(static_cast<T*>(mBuild), "get_output_shapes");
            if (!pyGetOutputShapes)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_output_shapes()");
            }

            std::vector<DimsExprs> inVector;
            std::vector<DimsExprs> shapeInVector;
            std::copy_n(inputs, nbInputs, std::back_inserter(inVector));
            std::copy_n(shapeInputs, nbShapeInputs, std::back_inserter(shapeInVector));

            py::object pyResult = pyGetOutputShapes(inVector, shapeInVector, &exprBuilder);

            try
            {
                auto result = pyResult.cast<std::vector<DimsExprs>>();
                if (static_cast<int32_t>(result.size()) != nbOutputs)
                {
                    utils::throwPyError(PyExc_RuntimeError,
                        "get_output_shapes() returned a list with a different length than num_outputs");
                }
                std::copy(result.begin(), result.end(), outputs);
                return 0;
            }
            PLUGIN_API_CATCH_CAST("get_output_shapes", "std::vector<nvinfer1::DimsExprs>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_output_shapes() " << e.what() << std::endl;
            }
            return -1;
        }
        PLUGIN_API_CATCH("get_output_shapes")
        return -1;
    }

    int32_t getAliasedInput(int32_t outputIndex) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetAliasedInput = py::get_override(static_cast<T*>(mBuild), "get_aliased_input");

            if (!pyGetAliasedInput)
            {
                // if no implementation is provided for get_aliased_input(), default to no aliasing
                return -1;
            }

            py::object pyResult = pyGetAliasedInput(outputIndex);

            try
            {
                auto result = pyResult.cast<int32_t>();
                return result;
            }
            PLUGIN_API_CATCH_CAST("get_aliased_input", "int32_t")
            return -1;
        }
        PLUGIN_API_CATCH("get_aliased_input")
        return -1;
    }

    char const* getTimingCacheID() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyGetTimingCacheID = py::get_override(static_cast<T*>(mBuild), "get_timing_cache_id");

            if (!pyGetTimingCacheID)
            {
                return nullptr;
            }

            py::object pyResult = pyGetTimingCacheID();

            try
            {
                mTimingCachedId = pyResult.cast<std::string>();
                return mTimingCachedId.c_str();
            }
            PLUGIN_API_CATCH_CAST("get_timing_cache_id", "std::string")
            return nullptr;
        }
        PLUGIN_API_CATCH("get_timing_cache_id")
        return nullptr;
    }

    char const* getMetadataString() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyGetMetadataString = py::get_override(static_cast<T*>(mBuild), "get_metadata_string");

            if (!pyGetMetadataString)
            {
                return nullptr;
            }

            py::object pyResult = pyGetMetadataString();

            try
            {
                mMetadataString = pyResult.cast<std::string>();
                return mMetadataString.c_str();
            }
            PLUGIN_API_CATCH_CAST("get_metadata_string", "std::string")
            return nullptr;
        }
        PLUGIN_API_CATCH("metadata_string")
        return nullptr;
    }

private:
    std::string mTimingCachedId{};
    std::string mMetadataString{};
    std::optional<std::vector<int32_t>> mTactics;
    std::optional<std::vector<PluginTensorDesc>> mSupportedFormatCombinations{};
};

class PyIPluginV3QuickBuildImpl : public PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickBuild>
{
public:
    PyIPluginV3QuickBuildImpl()
        : PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickBuild>(this)
    {
    }

    PyIPluginV3QuickBuildImpl(IPluginV3QuickBuild const& a)
        : PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickBuild>(&a){};
};

class PyIPluginV3QuickAOTBuildImpl : public PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickAOTBuild>
{
public:
    PyIPluginV3QuickAOTBuildImpl()
        : PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickAOTBuild>(this)
    {
    }

    PyIPluginV3QuickAOTBuildImpl(IPluginV3QuickAOTBuild const& a)
        : PyIPluginV3QuickBuildBaseImpl<IPluginV3QuickAOTBuild>(&a){};

    int32_t getKernel(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs,
        const char** kernelName, char** compiledKernel, int32_t* compiledKernelSize) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetKernel = utils::getOverride(static_cast<IPluginV3QuickAOTBuild*>(this), "get_kernel");

            if (!pyGetKernel)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_kernel()");
            }

            std::vector<PluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(in + idx));
            }

            std::vector<PluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(out + idx));
            }

            py::object pyResult = pyGetKernel(inVector, outVector);

            try
            {
                auto result = pyResult.cast<std::tuple<std::string, py::bytes>>();

                mKernelName = std::get<0>(result);
                *kernelName = mKernelName.c_str();
                mCompiledKernel = std::get<1>(result);
                py::buffer_info buffer(py::buffer(mCompiledKernel).request());
                *compiledKernel = static_cast<char*>(buffer.ptr);
                *compiledKernelSize = buffer.size;

                return 0;
            }
            PLUGIN_API_CATCH_CAST("get_kernel", "std::tuple<std::string, std::bytes>")
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from get_kernel() " << e.what() << std::endl;
            }
            return -1;
        }
        PLUGIN_API_CATCH("get_kernel")
        return -1;
    }

    int32_t getLaunchParams(DimsExprs const* inputs, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs, IKernelLaunchParams* launchParams, ISymExprs* extraArgs,
        IExprBuilder& exprBuilder) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetLaunchParams
                = utils::getOverride(static_cast<IPluginV3QuickAOTBuild*>(this), "get_launch_params");
            if (!pyGetLaunchParams)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_launch_params()");
            }

            std::vector<DimsExprs> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            std::vector<DynamicPluginTensorDesc> inOutVector;
            std::copy_n(inOut, nbInputs + nbOutputs, std::back_inserter(inOutVector));

            py::object pyResult
                = pyGetLaunchParams(inVector, inOutVector, nbInputs, launchParams, extraArgs, &exprBuilder);

            return 0;
        }
        PLUGIN_API_CATCH("get_launch_params")
        return -1;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySetTactic = utils::getOverride(static_cast<IPluginV3QuickAOTBuild*>(this), "set_tactic");
            if (!pySetTactic)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for set_tactic()");
            }

            try
            {
                pySetTactic(tactic);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from set_tactic() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("set_tactic")
        return -1;
    }

private:
    py::bytes mCompiledKernel;
    std::string mKernelName;
};

class PyIPluginV3QuickRuntimeImpl : public IPluginV3QuickRuntime
{
public:
    using IPluginV3QuickRuntime::IPluginV3QuickRuntime;
    PyIPluginV3QuickRuntimeImpl() = default;
    PyIPluginV3QuickRuntimeImpl(const IPluginV3QuickRuntime& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, Dims const* inputStrides, Dims const* outputStrides, int32_t nbInputs, int32_t nbOutputs,
        cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyEnqueue = utils::getOverride(static_cast<IPluginV3QuickRuntime*>(this), "enqueue");
            if (!pyEnqueue)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for enqueue()");
            }

            std::vector<PluginTensorDesc> inVector;
            std::vector<PluginTensorDesc> outVector;
            std::copy_n(inputDesc, nbInputs, std::back_inserter(inVector));
            std::copy_n(outputDesc, nbOutputs, std::back_inserter(outVector));

            std::vector<intptr_t> inPtrs;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inPtrs.push_back(reinterpret_cast<intptr_t>(inputs[idx]));
            }
            std::vector<intptr_t> outPtrs;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outPtrs.push_back(reinterpret_cast<intptr_t>(outputs[idx]));
            }

            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);

            std::vector<Dims> inStrides;
            std::vector<Dims> outStrides;
            std::copy_n(inputStrides, nbInputs, std::back_inserter(inStrides));
            std::copy_n(outputStrides, nbOutputs, std::back_inserter(outStrides));

            try
            {
                pyEnqueue(inVector, outVector, inPtrs, outPtrs, inStrides, outStrides, cudaStreamPtr);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from enqueue() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("enqueue")
        return -1;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySetTactic = utils::getOverride(static_cast<IPluginV3QuickRuntime*>(this), "set_tactic");
            if (!pySetTactic)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for set_tactic()");
            }

            try
            {
                pySetTactic(tactic);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from set_tactic() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("set_tactic")
        return -1;
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetFieldsToSerialize
                = utils::getOverride(static_cast<const IPluginV3QuickRuntime*>(this), "get_fields_to_serialize");
            if (!pyGetFieldsToSerialize)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_fields_to_serialize()");
            }

            py::object result = pyGetFieldsToSerialize();

            try
            {
                mFC = result.cast<PluginFieldCollection>();
                return &mFC;
            }
            PLUGIN_API_CATCH_CAST("get_fields_to_serialize", "nvinfer1::PluginFieldCollection")
            return nullptr;
        }
        PLUGIN_API_CATCH("get_fields_to_serialize")
        return nullptr;
    }

    void setPluginType(std::string pluginType)
    {
        mPluginType = std::move(pluginType);
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
    }

private:
    PluginFieldCollection mFC;
    std::optional<std::string> mNamespace;
    std::optional<std::string> mPluginType;
    std::optional<std::string> mPluginVersion;
};

class PyIPluginV3OneRuntimeImpl : public IPluginV3OneRuntime
{
public:
    using IPluginV3OneRuntime::IPluginV3OneRuntime;
    PyIPluginV3OneRuntimeImpl() = default;
    PyIPluginV3OneRuntimeImpl(const IPluginV3OneRuntime& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyEnqueue = utils::getOverride(static_cast<IPluginV3OneRuntime*>(this), "enqueue");
            if (!pyEnqueue)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for enqueue()");
            }

            std::vector<PluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inVector.push_back(*(inputDesc + idx));
            }
            std::vector<PluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outVector.push_back(*(outputDesc + idx));
            }

            std::vector<intptr_t> inPtrs;
            for (int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inPtrs.push_back(reinterpret_cast<intptr_t>(inputs[idx]));
            }
            std::vector<intptr_t> outPtrs;
            for (int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outPtrs.push_back(reinterpret_cast<intptr_t>(outputs[idx]));
            }

            intptr_t workspacePtr = reinterpret_cast<intptr_t>(workspace);
            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);

            try
            {
                pyEnqueue(inVector, outVector, inPtrs, outPtrs, workspacePtr, cudaStreamPtr);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from enqueue() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("enqueue")
        return -1;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pySetTactic = utils::getOverride(static_cast<IPluginV3OneRuntime*>(this), "set_tactic");
            if (!pySetTactic)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for set_tactic()");
            }

            try
            {
                pySetTactic(tactic);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from set_tactic() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("set_tactic")
        return -1;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        try
        {
            mNbInputs = nbInputs;
            mNbOutputs = nbOutputs;

            py::gil_scoped_acquire gil{};

            py::function pyConfigurePlugin
                = utils::getOverride(static_cast<IPluginV3OneRuntime*>(this), "on_shape_change");
            if (!pyConfigurePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for on_shape_change()");
            }

            std::vector<PluginTensorDesc> inVector;
            for (int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(in + idx));
            }

            std::vector<PluginTensorDesc> outVector;
            for (int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(out + idx));
            }

            try
            {
                pyConfigurePlugin(inVector, outVector);
            }
            catch (py::error_already_set& e)
            {
                std::cerr << "[ERROR] Exception thrown from on_shape_change() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("on_shape_change")
        return -1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyAttachToContext
                = utils::getOverride(static_cast<const IPluginV3OneRuntime*>(this), "attach_to_context");
            if (!pyAttachToContext)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for attach_to_context()");
            }

            py::handle handle = pyAttachToContext(context).release();

            try
            {
                return handle.cast<IPluginV3*>();
            }
            PLUGIN_API_CATCH_CAST("attach_to_context", "nvinfer1::IPluginV3")
            return nullptr;
        }
        PLUGIN_API_CATCH("attach_to_context")
        return nullptr;
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetFieldsToSerialize
                = utils::getOverride(static_cast<const IPluginV3OneRuntime*>(this), "get_fields_to_serialize");
            if (!pyGetFieldsToSerialize)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for get_fields_to_serialize()");
            }

            py::object result = pyGetFieldsToSerialize();

            try
            {
                mFC = result.cast<PluginFieldCollection>();
                return &mFC;
            }
            PLUGIN_API_CATCH_CAST("get_fields_to_serialize", "nvinfer1::PluginFieldCollection")
            return nullptr;
        }
        PLUGIN_API_CATCH("get_fields_to_serialize")
        return nullptr;
    }

    void setPluginType(std::string pluginType)
    {
        mPluginType = std::move(pluginType);
        mIsPluginTypeInitialized = true;
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
        mIsPluginVersionInitialized = true;
    }

private:
    int32_t mNbInputs{};
    int32_t mNbOutputs{};
    std::string mNamespace;
    std::string mPluginType;
    std::string mPluginVersion;
    PluginFieldCollection mFC;

    bool mIsPluginTypeInitialized{false};
    bool mIsPluginVersionInitialized{false};
};

class PyIPluginV3OneCoreImpl : public IPluginV3OneCore
{
public:
    using IPluginV3OneCore::IPluginV3OneCore;
    PyIPluginV3OneCoreImpl() = default;
    PyIPluginV3OneCoreImpl(const IPluginV3OneCore& a){};

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    char const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginNameInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_name not initialized");
            }
            return mPluginName.c_str();
        }
        PLUGIN_API_CATCH("plugin_name")
        return nullptr;
    }

    char const* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginVersionInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    const char* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            // getPluginNamespace() is not passed through to the Python side
            if (!mIsPluginNamespaceInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mPluginNamespace.c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    void setPluginName(std::string pluginName)
    {
        mPluginName = std::move(pluginName);
        mIsPluginNameInitialized = true;
    }

    void setPluginNamespace(std::string pluginNamespace)
    {
        mPluginNamespace = std::move(pluginNamespace);
        mIsPluginNamespaceInitialized = true;
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
        mIsPluginVersionInitialized = true;
    }

private:
    std::string mPluginNamespace;
    std::string mPluginName;
    std::string mPluginVersion;

    bool mIsPluginNamespaceInitialized{false};
    bool mIsPluginNameInitialized{false};
    bool mIsPluginVersionInitialized{false};
};

class IPluginCreatorV3OneImpl : public IPluginCreatorV3One
{
public:
    IPluginCreatorV3OneImpl() = default;

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    char const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNameInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "name not initialized");
            }
            return mName.c_str();
        }
        PLUGIN_API_CATCH("name")
        return nullptr;
    }

    const char* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsPluginVersionInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsFCInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "field_names not initialized");
            }
            return &mFC;
        }
        PLUGIN_API_CATCH("field_names")
        return nullptr;
    }

    IPluginV3* createPlugin(const char* name, const PluginFieldCollection* fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyCreatePlugin = utils::getOverride(static_cast<IPluginCreatorV3One*>(this), "create_plugin");
            if (!pyCreatePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for create_plugin()");
            }

            std::string nameString{name};

            py::handle handle = pyCreatePlugin(nameString, fc, phase).release();
            try
            {
                return handle.cast<IPluginV3*>();
            }
            PLUGIN_API_CATCH_CAST("create_plugin", "IPluginV3*")
            return nullptr;
        }
        PLUGIN_API_CATCH("create_plugin")
        return nullptr;
    }

    const char* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mIsNamespaceInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mNamespace.c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    void setFieldNames(PluginFieldCollection fc)
    {
        mFC = fc;
        mIsFCInitialized = true;
    }

    void setPluginName(std::string name)
    {
        mName = std::move(name);
        mIsNameInitialized = true;
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
        mIsPluginVersionInitialized = true;
    }

    void setPluginNamespace(std::string pluginNamespace)
    {
        mNamespace = std::move(pluginNamespace);
        mIsNamespaceInitialized = true;
    }

private:
    nvinfer1::PluginFieldCollection mFC{};
    std::string mNamespace;
    std::string mName;
    std::string mPluginVersion;

    bool mIsFCInitialized{false};
    bool mIsNamespaceInitialized{false};
    bool mIsNameInitialized{false};
    bool mIsPluginVersionInitialized{false};
};

class IPluginCreatorV3QuickImpl : public IPluginCreatorV3Quick
{
public:
    IPluginCreatorV3QuickImpl() = default;

    APILanguage getAPILanguage() const noexcept final
    {
        return APILanguage::kPYTHON;
    }

    char const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mName.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "name not initialized");
            }
            return mName.value().c_str();
        }
        PLUGIN_API_CATCH("name")
        return nullptr;
    }

    char const* getPluginVersion() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mPluginVersion.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_version not initialized");
            }
            return mPluginVersion.value().c_str();
        }
        PLUGIN_API_CATCH("plugin_version")
        return nullptr;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mFC.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "field_names not initialized");
            }
            return &mFC.value();
        }
        PLUGIN_API_CATCH("field_names")
        return nullptr;
    }

    IPluginV3* createPlugin(char const* name, char const* nspace, const PluginFieldCollection* fc, TensorRTPhase phase,
        QuickPluginCreationRequest quickPluginType) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyCreatePlugin
                = utils::getOverride(static_cast<IPluginCreatorV3Quick*>(this), "create_plugin");
            if (!pyCreatePlugin)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for create_plugin()");
            }

            std::string nameString{name};
            std::string namespaceString{nspace};
            py::handle handle = pyCreatePlugin(nameString, namespaceString, fc, phase, quickPluginType).release();
            try
            {
                return handle.cast<IPluginV3*>();
            }
            PLUGIN_API_CATCH_CAST("create_plugin", "IPluginV3*")
            return nullptr;
        }
        PLUGIN_API_CATCH("create_plugin")
        return nullptr;
    }

    char const* getPluginNamespace() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mNamespace.has_value())
            {
                utils::throwPyError(PyExc_AttributeError, "plugin_namespace not initialized");
            }
            return mNamespace.value().c_str();
        }
        PLUGIN_API_CATCH("plugin_namespace")
        return nullptr;
    }

    void setFieldNames(PluginFieldCollection fc)
    {
        mFC = fc;
    }

    void setPluginName(std::string name)
    {
        mName = std::move(name);
    }

    void setPluginVersion(std::string pluginVersion)
    {
        mPluginVersion = std::move(pluginVersion);
    }

    void setPluginNamespace(std::string pluginNamespace)
    {
        mNamespace = std::move(pluginNamespace);
    }

private:
    std::optional<nvinfer1::PluginFieldCollection> mFC;
    std::optional<std::string> mNamespace;
    std::optional<std::string> mName;
    std::optional<std::string> mPluginVersion;
};

class SymExprImpl : public ISymExpr
{
public:
    SymExprImpl() = default;

    PluginArgType getType() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mType.has_value())
            {
                utils::throwPyError(PyExc_RuntimeError, "type not initialized");
            }
            return mType.value();
        }
        PLUGIN_API_CATCH("get_type")
        return PluginArgType{};
    }

    PluginArgDataType getDataType() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mDtype.has_value())
            {
                utils::throwPyError(PyExc_RuntimeError, "data_type not initialized");
            }
            return mDtype.value();
        }
        PLUGIN_API_CATCH("get_data_type")
        return PluginArgDataType{};
    }

    void* getExpr() noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if (!mExpr.has_value())
            {
                utils::throwPyError(PyExc_RuntimeError, "expr not initialized");
            }
            return mExpr.value();
        }
        PLUGIN_API_CATCH("get_expr")
        return nullptr;
    }

    std::optional<PluginArgDataType> mDtype;
    std::optional<PluginArgType> mType;
    std::optional<void*> mExpr;
};

namespace
{
bool isPython(IVersionedInterface const& versionedInterface)
{
    return versionedInterface.getAPILanguage() == APILanguage::kPYTHON;
}
} // namespace

// Long lambda functions should go here rather than being inlined into the bindings (1 liners are OK).
namespace lambdas
{
// For IPluginV2
static const auto IPluginV2_get_output_shape
    = [](IPluginV2& self, int32_t const index, std::vector<Dims> const& inputShapes) {
          return self.getOutputDimensions(index, inputShapes.data(), inputShapes.size());
      };

static const auto IPluginV2_configure_with_format
    = [](IPluginV2& self, std::vector<Dims> const& inputShapes, std::vector<Dims> const& outputShapes, DataType dtype,
          TensorFormat format, int32_t maxBatchSize) {
          return self.configureWithFormat(inputShapes.data(), inputShapes.size(), outputShapes.data(),
              outputShapes.size(), dtype, format, maxBatchSize);
      };

static const auto IPluginV2_serialize = [](IPluginV2& self) {
    size_t size = self.getSerializationSize();
    // Python will own and free the memory returned by this function
    uint8_t* buffer = new uint8_t[size];
    self.serialize(buffer);

#if PYBIND11_VERSION_MAJOR < 2 || PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR < 6
    py::buffer_info info{
        buffer,                                   /* Pointer to buffer */
        sizeof(uint8_t),                          /* Size of one scalar */
        py::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
        1,                                        /* Number of dimensions */
        {size},                                   /* Buffer dimensions */
        {sizeof(uint8_t)}                         /* Strides (in bytes) for each index */
    };
    py::memoryview pyBuffer{info};
#else
    py::memoryview pyBuffer{py::memoryview::from_buffer(buffer, {size}, {sizeof(uint8_t)})};
#endif
    return pyBuffer;
};

// `const vector<const void*>::data()` corresponds to `const void* const*` (pointer to const-pointer to const void)
static const auto IPluginV2_execute_async = [](IPluginV2& self, int32_t batchSize,
                                                const std::vector<const void*>& inputs, std::vector<void*>& outputs,
                                                void* workspace, long stream) {
    return self.enqueue(batchSize, inputs.data(), outputs.data(), workspace, reinterpret_cast<cudaStream_t>(stream));
};

static const auto IPluginV2_set_num_outputs = [](IPluginV2& self, int32_t numOutputs) {
    if (getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
    {
        auto plugin = static_cast<PyIPluginV2DynamicExtImpl*>(&self);
        plugin->setNbOutputs(numOutputs);
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: num_outputs is read-only for C++ plugins");
};

static const auto IPluginV2_set_plugin_type = [](IPluginV2& self, std::string pluginType) {
    if (getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
    {
        auto plugin = reinterpret_cast<PyIPluginV2DynamicExtImpl*>(&self);
        plugin->setPluginType(std::move(pluginType));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: plugin_type is read-only for C++ plugins");
};

static const auto IPluginV2_set_plugin_version = [](IPluginV2& self, std::string pluginVersion) {
    if (getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
    {
        auto plugin = reinterpret_cast<PyIPluginV2DynamicExtImpl*>(&self);
        plugin->setPluginVersion(std::move(pluginVersion));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: plugin_version is read-only for C++ plugins");
};

// For IPluginV2Ext
static const auto get_output_data_type = [](IPluginV2Ext& self, int32_t index, const std::vector<DataType> inputTypes) {
    return self.getOutputDataType(index, inputTypes.data(), inputTypes.size());
};

// For IPluginV2Ext - makes copy of a vector<bool> as a bool[].
static std::unique_ptr<bool[]> makeBoolArray(std::vector<bool> const& v)
{
    int32_t const n{static_cast<int32_t>(v.size())};
    std::unique_ptr<bool[]> result(n > 0 ? new bool[n] : nullptr);
    std::copy_n(v.begin(), n, result.get());
    return result;
}

static const auto configure_plugin
    = [](IPluginV2Ext& self, std::vector<Dims> const& inputShapes, std::vector<Dims> const& outputShapes,
          std::vector<DataType> const& inputTypes, std::vector<DataType> const& outputTypes,
          std::vector<bool> const& inputIsBroadcasted, std::vector<bool> const& outputIsBroadcasted,
          TensorFormat format, int32_t maxBatchSize) {
          auto inputBroadcast = makeBoolArray(inputIsBroadcasted);
          auto outputBroadcast = makeBoolArray(outputIsBroadcasted);
          return self.configurePlugin(inputShapes.data(), inputShapes.size(), outputShapes.data(), outputShapes.size(),
              inputTypes.data(), outputTypes.data(), inputBroadcast.get(), outputBroadcast.get(), format, maxBatchSize);
      };

static const auto attach_to_context = [](IPluginV2Ext& self, void* cudnn, void* cublas, void* allocator) {
    self.attachToContext(
        static_cast<cudnnContext*>(cudnn), static_cast<cublasContext*>(cublas), static_cast<IGpuAllocator*>(allocator));
};

// For PluginField
static const auto plugin_field_default_constructor
    = [](const FallbackString& name) { return new PluginField{name.c_str()}; };

static const auto plugin_field_constructor
    = [](const FallbackString& name, py::buffer& data, nvinfer1::PluginFieldType type) {
          py::buffer_info info = data.request();
          // PluginField length is number of entries. type gives information about the size of each entry.
          return new PluginField{name.c_str(), info.ptr, type, static_cast<int32_t>(info.size)};
      };

// For PluginFieldCollection
static const auto plugin_field_collection_constructor = [](std::vector<PluginField> const& fields) {
    return new PluginFieldCollection{static_cast<int32_t>(fields.size()), fields.data()};
};

// For IPluginRegistry. We do an allocation here, but python takes ownership.
static const auto get_plugin_creator_list = [](IPluginRegistry& self) {
    int32_t numCreators{0};
    IPluginCreator* const* ptr = self.getPluginCreatorList(&numCreators);
    // This is NOT a memory leak - python will free when done.
    return new std::vector<IPluginCreator*>(ptr, ptr + numCreators);
};

std::vector<py::object>* getCreatorsUtil(
    std::function<IPluginCreatorInterface* const*(int32_t* const)> getCreatorsFunc, std::string const& funcName)
{
    int32_t numCreators{0};
    IPluginCreatorInterface* const* ptr = getCreatorsFunc(&numCreators);
    // Python will free when done.
    auto vec = std::make_unique<std::vector<py::object>>(numCreators);
    try
    {
        std::generate(vec->begin(), vec->end(), [&ptr, i = 0]() mutable -> py::object {
            if (std::strcmp(ptr[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V1") == 0)
            {
                return py::cast(static_cast<IPluginCreator const*>(ptr[i++]));
            }
            if (std::strcmp(ptr[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V3ONE") == 0)
            {
                return py::cast(static_cast<IPluginCreatorV3One const*>(ptr[i++]));
            }
            if (std::strcmp(ptr[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V3QUICK") == 0)
            {
                return py::cast(static_cast<IPluginCreatorV3Quick const*>(ptr[i++]));
            }
            utils::throwPyError(PyExc_RuntimeError, "Unknown plugin creator type");
            return py::none{};
        });
        return vec.release();
    }
    catch (std::exception const& e)
    {
        std::cerr << "[ERROR] Exception caught in " << funcName << "(): " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[ERROR] Exception caught in " << funcName << "()" << std::endl;
    }
    return nullptr;
}

static const auto get_all_creators = [](IPluginRegistry& self) -> std::vector<py::object>* {
    return getCreatorsUtil(
        std::bind(&IPluginRegistry::getAllCreators, &self, std::placeholders::_1), "get_all_creators");
};

static const auto get_all_creators_recursive = [](IPluginRegistry& self) -> std::vector<py::object>* {
    return getCreatorsUtil(std::bind(&IPluginRegistry::getAllCreatorsRecursive, &self, std::placeholders::_1),
        "get_all_creators_recursive");
};

static const auto get_capability_interface = [](IPluginV3& self, PluginCapabilityType type) -> py::object {
    IPluginCapability* capability_interface = self.getCapabilityInterface(type);

    if (capability_interface == nullptr)
    {
        return py::none{};
    }
    else
    {
        try
        {
            if (type == PluginCapabilityType::kCORE)
            {
                try
                {
                    return py::cast(static_cast<IPluginV3OneCore*>(capability_interface));
                }
                catch (py::cast_error const& e)
                {
                    return py::cast(static_cast<IPluginV3QuickCore*>(capability_interface));
                }
            }
            if (type == PluginCapabilityType::kBUILD)
            {
                try
                {
                    return py::cast(static_cast<IPluginV3OneBuildV2*>(capability_interface));
                }
                catch (py::cast_error const& e)
                {
                    try
                    {
                        return py::cast(static_cast<IPluginV3OneBuild*>(capability_interface));
                    }
                    catch (py::cast_error const& e)
                    {
                        try
                        {
                            return py::cast(static_cast<IPluginV3QuickAOTBuild*>(capability_interface));
                        }
                        catch (py::cast_error const& e)
                        {
                            try
                            {
                                return py::cast(static_cast<IPluginV3QuickBuild*>(capability_interface));
                            }
                            PLUGIN_API_CATCH_CAST("get_capability_interface", " a valid build capability interface")
                        }
                    }
                }
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                try
                {
                    return py::cast(static_cast<IPluginV3OneRuntime*>(capability_interface));
                }
                catch (py::cast_error const& e)
                {
                    try
                    {
                        return py::cast(static_cast<IPluginV3QuickRuntime*>(capability_interface));
                    }
                    PLUGIN_API_CATCH_CAST("get_capability_interface", " a valid runtime capability interface")
                }
            }
        }
        PLUGIN_API_CATCH_CAST("get_capability_interface", "nvinfer1::IPluginCapability")
    }

    utils::throwPyError(PyExc_RuntimeError, "Unknown plugin capability type");
    return py::none{};
};

static const auto get_creator = [](IPluginRegistry& self, char const* pluginType, char const* pluginVersion,
                                    char const* pluginNamespace) -> py::object {
    IPluginCreatorInterface* creator = self.getCreator(pluginType, pluginVersion, pluginNamespace);
    if (creator == nullptr)
    {
        return py::none{};
    }
    else
    {
        if (std::strcmp(creator->getInterfaceInfo().kind, "PLUGIN CREATOR_V1") == 0)
        {
            return py::cast(static_cast<IPluginCreator*>(creator));
        }
        if (std::strcmp(creator->getInterfaceInfo().kind, "PLUGIN CREATOR_V3ONE") == 0)
        {
            return py::cast(static_cast<IPluginCreatorV3One*>(creator));
        }
        if (std::strcmp(creator->getInterfaceInfo().kind, "PLUGIN CREATOR_V3QUICK") == 0)
        {
            return py::cast(static_cast<IPluginCreatorV3Quick*>(creator));
        }
        utils::throwPyError(PyExc_RuntimeError, "Unknown plugin creator type");
        return py::none{};
    }
};

// For IPluginCreator
static const auto creator_create_plugin
    = [](IPluginCreator& self, std::string const& name, PluginFieldCollection const* fc) {
          return self.createPlugin(name.c_str(), fc);
      };

static const auto creator_create_plugin_v3
    = [](IPluginCreatorV3One& self, std::string const& name, PluginFieldCollection const* fc, TensorRTPhase phase) {
          return self.createPlugin(name.c_str(), fc, phase);
      };

static const auto creator_create_plugin_v3_quick
    = [](IPluginCreatorV3Quick& self, std::string const& name, std::string const& nspace,
          PluginFieldCollection const* fc, TensorRTPhase phase, QuickPluginCreationRequest quickPluginType) {
          return self.createPlugin(name.c_str(), nspace.c_str(), fc, phase, quickPluginType);
      };

static const auto deserialize_plugin = [](IPluginCreator& self, std::string const& name, py::buffer& serializedPlugin) {
    py::buffer_info info = serializedPlugin.request();
    return self.deserializePlugin(name.c_str(), info.ptr, info.size * info.itemsize);
};

static const auto IPluginV3_get_format_combination_limit = [](IPluginV3OneBuild& self, int32_t formatCombinationLimit) {
    if (isPython(self))
    {
        auto plugin = static_cast<PyIPluginV3OneBuildImpl*>(&self);
        plugin->setFormatCombinationLimit(formatCombinationLimit);
        return;
    }
    utils::throwPyError(
        PyExc_AttributeError, "Can't set attribute: format_combination_limit is read-only for C++ plugins");
};

static const auto IPluginV3_get_metadata_string = [](IPluginV3OneBuild& self, std::string metadataString) {
    if (isPython(self))
    {
        auto plugin = static_cast<PyIPluginV3OneBuildImpl*>(&self);
        plugin->setMetadataString(std::move(metadataString));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: metadata_string is read-only for C++ plugins");
};

static const auto IPluginV3_get_timing_cache_id = [](IPluginV3OneBuild& self, std::string timingCacheId) {
    if (isPython(self))
    {
        auto plugin = static_cast<PyIPluginV3OneBuildImpl*>(&self);
        plugin->setTimingCachedId(std::move(timingCacheId));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: timing_cache_id is read-only for C++ plugins");
};

// For base Dims class
static const auto dimsexprs_vector_constructor = [](std::vector<IDimensionExpr const*> const& in) {
    // This is required, because otherwise MAX_DIMS will not be resolved at compile time.
    int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
    PY_ASSERT_VALUE_ERROR(in.size() <= static_cast<size_t>(maxDims),
        "Input length " + std::to_string(in.size()) + ". Max expected length is " + std::to_string(maxDims));

    // Create the Dims object.
    DimsExprs* self = new DimsExprs{};
    self->nbDims = in.size();
    for (int32_t i = 0; static_cast<size_t>(i) < in.size(); ++i)
        self->d[i] = in[i];
    return self;
};

static const auto dimsexprs_len_constructor = [](int32_t const size) {
    // This is required, because otherwise MAX_DIMS will not be resolved at compile time.
    int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
    PY_ASSERT_VALUE_ERROR(size <= maxDims,
        "Input length " + std::to_string(size) + ". Max expected length is " + std::to_string(maxDims));

    // Create the Dims object.
    DimsExprs* self = new DimsExprs{};
    self->nbDims = size;
    return self;
};

static const auto dimsexprs_len = [](DimsExprs const& self) { return self.nbDims; };

// TODO: Add slicing support?
static const auto dimsexprs_getter = [](DimsExprs const& self, int32_t const pyIndex) -> IDimensionExpr const* {
    // Without these bounds checks, horrible infinite looping will occur.
    int32_t const index{(pyIndex < 0) ? static_cast<int32_t>(self.nbDims) + pyIndex : pyIndex};
    PY_ASSERT_INDEX_ERROR(index >= 0 && index < self.nbDims);
    return self.d[index];
};

static const auto dimsexprs_setter = [](DimsExprs& self, int32_t const pyIndex, IDimensionExpr const* item) {
    int32_t const index{(pyIndex < 0) ? static_cast<int32_t>(self.nbDims) + pyIndex : pyIndex};
    PY_ASSERT_INDEX_ERROR(index >= 0 && index < self.nbDims);
    self.d[index] = item;
};

// IPluginV3 lambdas

static const auto IPluginV3_set_num_outputs = [](IPluginV3OneBuild& self, int32_t numOutputs) {
    if (isPython(self))
    {
        auto plugin = static_cast<PyIPluginV3OneBuildImpl*>(&self);
        plugin->setNbOutputs(numOutputs);
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: num_outputs is read-only for C++ plugins");
};

static const auto ISymExpr_set_type = [](ISymExpr& self, PluginArgType mType) {
    auto expr = static_cast<SymExprImpl*>(&self);
    expr->mType = mType;
    return;
};

static const auto ISymExpr_set_data_type = [](ISymExpr& self, PluginArgDataType mDtype) {
    auto expr = static_cast<SymExprImpl*>(&self);
    expr->mDtype = mDtype;
    return;
};

static const auto ISymExpr_set_expr = [](ISymExpr& self, void* mExpr) {
    auto expr = static_cast<SymExprImpl*>(&self);
    expr->mExpr = mExpr;
    return;
};

} // namespace lambdas

namespace helpers
{
template <typename T>
inline PluginFieldCollection const* getFieldNames(T& self)
{
    PluginFieldCollection const* fieldCollection = self.getFieldNames();
    if (!fieldCollection)
    {
        return &EMPTY_PLUGIN_FIELD_COLLECTION;
    }
    return fieldCollection;
}

template <typename T, typename U>
inline void setPluginName(T& self, std::string name)
{
    if (isPython(self))
    {
        auto object = static_cast<U*>(&self);
        object->setPluginName(std::move(name));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: read-only for C++ plugins");
}

template <typename T, typename U>
inline void setPluginVersion(T& self, std::string pluginVersion)
{
    if (isPython(self))
    {
        auto object = static_cast<U*>(&self);
        object->setPluginVersion(std::move(pluginVersion));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: read-only for C++ plugins");
}

template <typename T, typename U>
inline void setPluginNamespace(T& self, std::string namespace_)
{
    if (isPython(self))
    {
        auto object = static_cast<U*>(&self);
        object->setPluginNamespace(std::move(namespace_));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: read-only for C++ plugins");
}

template <typename T, typename U>
inline void setPluginCreatorFieldNames(T& self, PluginFieldCollection pfc)
{
    if (isPython(self))
    {
        auto pluginCreator = static_cast<U*>(&self);
        pluginCreator->setFieldNames(pfc);
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: read-only for C++ plugins");
}

} // namespace helpers

// NOTE: Fake bindings are provided solely to document the API in the C++ -> Python direction for these methods
// These bindings will never be called.
namespace pluginDoc
{

py::bytes serialize(PyIPluginV2DynamicExt& self)
{
    return py::bytes();
}

DataType getOutputDataType(PyIPluginV2DynamicExt& self, int32_t index, std::vector<DataType> const& inputTypes)
{
    return DataType{};
}

PyIPluginV2DynamicExt* deserializePlugin(
    PyIPluginV2DynamicExt& self, std::string const& name, py::bytes const& serializedPlugin)
{
    return nullptr;
}

DimsExprs getOutputDimensions(
    PyIPluginV2DynamicExt& self, int32_t outputIndex, std::vector<DimsExprs> const& inputs, IExprBuilder& exprBuilder)
{
    return DimsExprs{};
}

void configurePlugin(PyIPluginV2DynamicExt& self, std::vector<DynamicPluginTensorDesc> const& in,
    std::vector<DynamicPluginTensorDesc> const& out)
{
}

size_t getWorkspaceSize(PyIPluginV2DynamicExt& self, std::vector<PluginTensorDesc> const& inputDesc,
    std::vector<PluginTensorDesc> const& outputDesc)
{
    return 0U;
}

bool supportsFormatCombination(
    PyIPluginV2DynamicExt& self, int32_t pos, std::vector<PluginTensorDesc> const& inOut, int32_t nbInputs)
{
    return false;
}

void enqueue(PyIPluginV2DynamicExt& self, std::vector<PluginTensorDesc> const& inputDesc,
    std::vector<PluginTensorDesc> const& outputDesc, const std::vector<intptr_t>& inputs,
    std::vector<intptr_t>& outputs, intptr_t workspace, long stream)
{
}

int32_t initialize(PyIPluginV2DynamicExt& self)
{
    return -1;
}

void terminate(PyIPluginV2DynamicExt& self) {}

void destroy(PyIPluginV2DynamicExt& self) {}

PyIPluginV2DynamicExt* clone(PyIPluginV2DynamicExt& self)
{
    return nullptr;
}

size_t getSerializationSize(PyIPluginV2DynamicExt& self)
{
    return 0U;
}

std::vector<DataType> getOutputDataTypes(IPluginV3& self, std::vector<DataType> const& inputTypes)
{
    return {};
}

std::vector<DimsExprs> getOutputShapes(IPluginV3& self, std::vector<DimsExprs> const& inputs,
    std::vector<DimsExprs> const& shapeInputs, IExprBuilder& exprBuilder)
{
    return {};
}

void configurePluginV3(
    IPluginV3& self, std::vector<DynamicPluginTensorDesc> const& in, std::vector<DynamicPluginTensorDesc> const& out)
{
}

void onShapeChange(IPluginV3& self, std::vector<PluginTensorDesc> const& in, std::vector<PluginTensorDesc> const& out)
{
}

size_t getWorkspaceSizeV3(IPluginV3& self, std::vector<DynamicPluginTensorDesc> const& inputDesc,
    std::vector<DynamicPluginTensorDesc> const& outputDesc)
{
    return 0U;
}

bool supportsFormatCombinationV3(
    IPluginV3& self, int32_t pos, std::vector<DynamicPluginTensorDesc> const& inOut, int32_t nbInputs)
{
    return false;
}

void enqueueV3(IPluginV3& self, std::vector<PluginTensorDesc> const& inputDesc,
    std::vector<PluginTensorDesc> const& outputDesc, const std::vector<intptr_t>& inputs,
    std::vector<intptr_t>& outputs, intptr_t workspace, long stream)
{
}

void destroyV3(IPluginV3& self) {}

IPluginV3* cloneV3(IPluginV3& self)
{
    return nullptr;
}

IPluginV3* attachToContext(IPluginV3& self, IPluginResourceContext& context)
{
    return nullptr;
}

PluginFieldCollection* getFieldsToSerialize(IPluginV3& self)
{
    return nullptr;
}

std::vector<int32_t> getValidTactics(IPluginV3& self)
{
    return {};
}

void setTactic(IPluginV3& self, int32_t tactic) {}

void release(IPluginResource& self) {}

IPluginResource* clonePluginResource(IPluginResource& self)
{
    return nullptr;
}

int32_t getAliasedInput(int32_t outputIndex)
{
    return -1;
}

} // namespace pluginDoc


void bindPlugin(py::module& m)
{
    py::class_<IDimensionExpr, PyIDimensionExprImpl, std::unique_ptr<IDimensionExpr, py::nodelete>>(
        m, "IDimensionExpr", IDimensionExprDoc::descr, py::module_local())
        .def("is_constant", &IDimensionExpr::isConstant, IDimensionExprDoc::is_constant)
        .def("get_constant_value", &IDimensionExpr::getConstantValue, IDimensionExprDoc::get_constant_value)
        .def("is_size_tensor", &IDimensionExpr::isSizeTensor, IDimensionExprDoc::is_size_tensor);

    py::class_<DimsExprs>(m, "DimsExprs", DimsExprsDoc::descr, py::module_local())
        .def(py::init<>())
        // Allows for construction from python lists and tuples.
        .def(py::init(lambdas::dimsexprs_vector_constructor))
        // Allows for construction with a specified number of dims.
        .def(py::init(lambdas::dimsexprs_len_constructor))
        // These functions allow us to use DimsExprs like an iterable.
        .def("__len__", lambdas::dimsexprs_len)
        .def("__getitem__", lambdas::dimsexprs_getter)
        .def("__setitem__", lambdas::dimsexprs_setter);

    py::enum_<DimensionOperation>(
        m, "DimensionOperation", py::arithmetic{}, DimensionOperationDoc::descr, py::module_local())
        .value("SUM", DimensionOperation::kSUM)
        .value("PROD", DimensionOperation::kPROD)
        .value("MAX", DimensionOperation::kMAX)
        .value("MIN", DimensionOperation::kMIN)
        .value("SUB", DimensionOperation::kSUB)
        .value("EQUAL", DimensionOperation::kEQUAL)
        .value("LESS", DimensionOperation::kLESS)
        .value("FLOOR_DIV", DimensionOperation::kFLOOR_DIV)
        .value("CEIL_DIV", DimensionOperation::kCEIL_DIV);

    py::class_<IExprBuilder, PyIExprBuilderImpl, std::unique_ptr<IExprBuilder, py::nodelete>>(
        m, "IExprBuilder", IExprBuilderDoc::descr, py::module_local())
        .def(py::init<>())
        .def(
            "constant", &IExprBuilder::constant, py::return_value_policy::reference_internal, IExprBuilderDoc::constant)
        .def("operation", &IExprBuilder::operation, py::return_value_policy::reference_internal,
            IExprBuilderDoc::operation)
        .def("declare_size_tensor", &IExprBuilder::declareSizeTensor, py::return_value_policy::reference_internal,
            IExprBuilderDoc::declare_size_tensor);

    py::class_<PluginTensorDesc>(m, "PluginTensorDesc", PluginTensorDescDoc::descr, py::module_local())
        .def(py::init<>())
        .def_readwrite("dims", &PluginTensorDesc::dims)
        .def_readwrite("type", &PluginTensorDesc::type)
        .def_readwrite("format", &PluginTensorDesc::format)
        .def_readwrite("scale", &PluginTensorDesc::scale);

    py::class_<DynamicPluginTensorDesc>(
        m, "DynamicPluginTensorDesc", DynamicPluginTensorDescDoc::descr, py::module_local())
        .def(py::init<>())
        .def_readwrite("desc", &DynamicPluginTensorDesc::desc)
        .def_readwrite("min", &DynamicPluginTensorDesc::min)
        .def_readwrite("opt", &DynamicPluginTensorDesc::opt)
        .def_readwrite("max", &DynamicPluginTensorDesc::max);

    py::enum_<PluginFieldType>(m, "PluginFieldType", PluginFieldTypeDoc::descr, py::module_local())
        .value("FLOAT16", PluginFieldType::kFLOAT16)
        .value("FLOAT32", PluginFieldType::kFLOAT32)
        .value("FLOAT64", PluginFieldType::kFLOAT64)
        .value("INT8", PluginFieldType::kINT8)
        .value("INT16", PluginFieldType::kINT16)
        .value("INT32", PluginFieldType::kINT32)
        .value("CHAR", PluginFieldType::kCHAR)
        .value("DIMS", PluginFieldType::kDIMS)
        .value("UNKNOWN", PluginFieldType::kUNKNOWN)
        .value("BF16", PluginFieldType::kBF16)
        .value("INT64", PluginFieldType::kINT64)
        .value("FP8", PluginFieldType::kFP8)
        .value("INT4", PluginFieldType::kINT4)
        .value("FP4", PluginFieldType::kFP4);

    py::class_<PluginField>(m, "PluginField", PluginFieldDoc::descr, py::module_local())
        .def(py::init(lambdas::plugin_field_default_constructor), "name"_a = "", py::keep_alive<1, 2>{})
        .def(py::init(lambdas::plugin_field_constructor), "name"_a, "data"_a,
            "type"_a = nvinfer1::PluginFieldType::kUNKNOWN, py::keep_alive<1, 2>{}, py::keep_alive<1, 3>{})
        .def_property(
            "name", [](PluginField& self) { return self.name; },
            py::cpp_function(
                [](PluginField& self, FallbackString& name) { self.name = name.c_str(); }, py::keep_alive<1, 2>{}))
        .def_property(
            "data",
            [](PluginField& self) {
                switch (self.type)
                {
                case PluginFieldType::kINT32: return py::array(self.length, static_cast<int32_t const*>(self.data));
                case PluginFieldType::kUNKNOWN:
                case PluginFieldType::kINT8: return py::array(self.length, static_cast<int8_t const*>(self.data));
                case PluginFieldType::kINT16: return py::array(self.length, static_cast<int16_t const*>(self.data));
                case PluginFieldType::kFLOAT32: return py::array(self.length, static_cast<float const*>(self.data));
                case PluginFieldType::kFLOAT64: return py::array(self.length, static_cast<double const*>(self.data));
                case PluginFieldType::kINT64: return py::array(self.length, static_cast<int64_t const*>(self.data));
                case PluginFieldType::kCHAR: return py::array(self.length, static_cast<char const*>(self.data));
                case PluginFieldType::kINT4:
                case PluginFieldType::kFLOAT16:
                case PluginFieldType::kBF16:
                case PluginFieldType::kDIMS:
                case PluginFieldType::kFP8:
                case PluginFieldType::kFP4:
                    utils::throwPyError(
                        PyExc_AttributeError, "No known conversion for returning data from PluginField");
                    break;
                default: return py::array();
                }
                // should not reach this line
                return py::array();
            },
            py::cpp_function(
                [](PluginField& self, py::buffer& buffer) {
                    py::buffer_info info = buffer.request();
                    self.data = info.ptr;
                },
                py::keep_alive<1, 2>{}))
        .def_readwrite("type", &PluginField::type)
        .def_readwrite("size", &PluginField::length);

    // PluginFieldCollection behaves like an iterable, and can be constructed from iterables.
    py::class_<PluginFieldCollection>(m, "PluginFieldCollection_", PluginFieldCollectionDoc::descr, py::module_local())
        .def(py::init<>(lambdas::plugin_field_collection_constructor), py::keep_alive<1, 2>{})
        .def("__len__", [](PluginFieldCollection& self) { return self.nbFields; })
        .def("__getitem__", [](PluginFieldCollection& self, int32_t const index) {
            PY_ASSERT_INDEX_ERROR(index < self.nbFields);
            return self.fields[index];
        });

    // Creating a trt.PluginFieldCollection in Python will actually construct a vector,
    // which can then be converted to an actual C++ PluginFieldCollection.
    py::implicitly_convertible<std::vector<nvinfer1::PluginField>, PluginFieldCollection>();

    py::class_<IPluginV2>(m, "IPluginV2", IPluginV2Doc::descr, py::module_local())
        .def_property("num_outputs", &IPluginV2::getNbOutputs, lambdas::IPluginV2_set_num_outputs)
        .def_property_readonly("tensorrt_version", &IPluginV2::getTensorRTVersion)
        .def_property("plugin_type", &IPluginV2::getPluginType,
            py::cpp_function(lambdas::IPluginV2_set_plugin_type, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginV2::getPluginVersion,
            py::cpp_function(lambdas::IPluginV2_set_plugin_version, py::keep_alive<1, 2>{}))
        .def_property("plugin_namespace", &IPluginV2::getPluginNamespace,
            py::cpp_function(&IPluginV2::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("get_output_shape", lambdas::IPluginV2_get_output_shape, "index"_a, "input_shapes"_a,
            IPluginV2Doc::get_output_shape)
        .def("supports_format", &IPluginV2::supportsFormat, "dtype"_a, "format"_a, IPluginV2Doc::supports_format)
        .def("configure_with_format", lambdas::IPluginV2_configure_with_format, "input_shapes"_a, "output_shapes"_a,
            "dtype"_a, "format"_a, "max_batch_size"_a, IPluginV2Doc::configure_with_format)
        .def("initialize", &IPluginV2::initialize, IPluginV2Doc::initialize)
        .def("terminate", &IPluginV2::terminate, IPluginV2Doc::terminate)
        .def("get_workspace_size", &IPluginV2::getWorkspaceSize, "max_batch_size"_a, IPluginV2Doc::get_workspace_size)
        .def("execute_async", lambdas::IPluginV2_execute_async, "batch_size"_a, "inputs"_a, "outputs"_a, "workspace"_a,
            "stream_handle"_a, IPluginV2Doc::execute_async)
        .def_property_readonly("serialization_size", &IPluginV2::getSerializationSize)
        .def(
            "serialize", lambdas::IPluginV2_serialize, IPluginV2Doc::serialize, py::return_value_policy::take_ownership)
        .def("destroy", &IPluginV2::destroy, IPluginV2Doc::destroy)
        .def("clone", &IPluginV2::clone, IPluginV2Doc::clone);

    py::class_<IPluginV2Ext, IPluginV2>(m, "IPluginV2Ext", IPluginV2ExtDoc::descr, py::module_local())
        .def("get_output_data_type", lambdas::get_output_data_type, "index"_a, "input_types"_a,
            IPluginV2ExtDoc::get_output_data_type)
        .def("configure_plugin", lambdas::configure_plugin, "input_shapes"_a, "output_shapes"_a, "input_types"_a,
            "output_types"_a, "input_is_broadcasted"_a, "output_is_broacasted"_a, "format"_a, "max_batch_size"_a,
            IPluginV2ExtDoc::configure_plugin)
        .def("attach_to_context", lambdas::attach_to_context, "cudnn"_a, "cublas"_a, "allocator"_a,
            IPluginV2ExtDoc::attach_to_context)
        .def("detach_from_context", &IPluginV2Ext::detachFromContext, IPluginV2ExtDoc::detach_from_context)
        .def("clone", &IPluginV2Ext::clone, IPluginV2ExtDoc::clone);
    ;

    py::class_<IPluginV2DynamicExt, IPluginV2, std::unique_ptr<IPluginV2DynamicExt, py::nodelete>>(
        m, "IPluginV2DynamicExtBase", py::module_local());

    py::class_<PyIPluginV2DynamicExt, IPluginV2DynamicExt, IPluginV2, PyIPluginV2DynamicExtImpl,
        std::unique_ptr<PyIPluginV2DynamicExt>>(
        m, "IPluginV2DynamicExt", IPluginV2DynamicExtDoc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const PyIPluginV2DynamicExt&>())
        .def_property_readonly_static(
            "FORMAT_COMBINATION_LIMIT", [](py::object) { return IPluginV2DynamicExt::kFORMAT_COMBINATION_LIMIT; })
        // The following defs are only for documenting the API for Python-based plugins
        .def("initialize", &pluginDoc::initialize, IPluginV2DynamicExtDoc::initialize)
        .def("terminate", &pluginDoc::terminate, IPluginV2DynamicExtDoc::terminate)
        .def("serialize", &pluginDoc::serialize, IPluginV2DynamicExtDoc::serialize)
        .def("get_output_datatype", &pluginDoc::getOutputDataType, "index"_a, "input_types"_a,
            IPluginV2DynamicExtDoc::get_output_data_type)
        .def("destroy", &pluginDoc::destroy, IPluginV2DynamicExtDoc::destroy)
        .def("get_serialization_size", &pluginDoc::getSerializationSize, IPluginV2DynamicExtDoc::get_serialization_size)
        .def("get_output_dimensions", &pluginDoc::getOutputDimensions, "output_index"_a, "inputs"_a, "expr_builder"_a,
            IPluginV2DynamicExtDoc::get_output_dimensions)
        .def("get_workspace_size", &pluginDoc::getWorkspaceSize, "in"_a, "out"_a,
            IPluginV2DynamicExtDoc::get_workspace_size)
        .def("configure_plugin", &pluginDoc::configurePlugin, "pos"_a, "in_out"_a,
            IPluginV2DynamicExtDoc::configure_plugin)
        .def("supports_format_combination", &pluginDoc::supportsFormatCombination, "pos"_a, "in_out"_a, "num_inputs"_a,
            IPluginV2DynamicExtDoc::supports_format_combination)
        .def("enqueue", &pluginDoc::enqueue, "input_desc"_a, "output_desc"_a, "inputs"_a, "outputs"_a, "workspace"_a,
            "stream"_a, IPluginV2DynamicExtDoc::enqueue)
        .def("clone", &pluginDoc::clone, IPluginV2DynamicExtDoc::clone);

    py::class_<IPluginCapability, IVersionedInterface, std::unique_ptr<IPluginCapability>>(
        m, "IPluginCapability", IPluginV3Doc::iplugincapability_descr, py::module_local());

    py::class_<IPluginV3, IVersionedInterface, PyIPluginV3Impl, std::unique_ptr<IPluginV3>>(
        m, "IPluginV3", IPluginV3Doc::ipluginv3_descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3&>())
        .def("get_capability_interface", lambdas::get_capability_interface, "type"_a,
            py::return_value_policy::reference_internal, IPluginV3Doc::get_capability_interface)
        // The following defs are only for documenting API for Python-based plugins
        .def("clone", &pluginDoc::cloneV3, IPluginV3Doc::clone)
        .def("destroy", &pluginDoc::destroyV3, IPluginV3Doc::destroy);

    py::class_<IPluginV3OneCore, IPluginCapability, IVersionedInterface, PyIPluginV3OneCoreImpl,
        std::unique_ptr<IPluginV3OneCore>>(
        m, "IPluginV3OneCore", IPluginV3Doc::ipluginv3onecore_descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3OneCore&>())
        .def_property("plugin_name", &IPluginV3OneCore::getPluginName,
            py::cpp_function(&helpers::setPluginName<IPluginV3OneCore, PyIPluginV3OneCoreImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginV3OneCore::getPluginVersion,
            py::cpp_function(
                &helpers::setPluginVersion<IPluginV3OneCore, PyIPluginV3OneCoreImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_namespace", &IPluginV3OneCore::getPluginNamespace,
            py::cpp_function(
                &helpers::setPluginNamespace<IPluginV3OneCore, PyIPluginV3OneCoreImpl>, py::keep_alive<1, 2>{}));

    py::class_<IPluginV3OneBuild, IPluginCapability, IVersionedInterface, PyIPluginV3OneBuildImpl,
        std::unique_ptr<IPluginV3OneBuild>>(
        m, "IPluginV3OneBuild", IPluginV3Doc::ipluginv3onebuild_descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3OneBuild&>())
        .def_property_readonly_static("DEFAULT_FORMAT_COMBINATION_LIMIT",
            [](py::object) { return IPluginV3OneBuild::kDEFAULT_FORMAT_COMBINATION_LIMIT; })
        .def_property("num_outputs", &IPluginV3OneBuild::getNbOutputs, lambdas::IPluginV3_set_num_outputs)
        .def_property("format_combination_limit", &IPluginV3OneBuild::getFormatCombinationLimit,
            lambdas::IPluginV3_get_format_combination_limit)
        .def_property("metadata_string", &IPluginV3OneBuild::getMetadataString,
            py::cpp_function(lambdas::IPluginV3_get_metadata_string, py::keep_alive<1, 2>{}))
        .def_property("timing_cache_id", &IPluginV3OneBuild::getTimingCacheID,
            py::cpp_function(lambdas::IPluginV3_get_timing_cache_id, py::keep_alive<1, 2>{}))
        // The following defs are only for documenting the API for Python-based plugins
        .def("get_output_data_types", &pluginDoc::getOutputDataTypes, "input_types"_a,
            IPluginV3Doc::get_output_data_types)
        .def("get_output_shapes", &pluginDoc::getOutputShapes, "inputs"_a, "shape_inputs"_a, "expr_builder"_a,
            IPluginV3Doc::get_output_shapes)
        .def("get_workspace_size", &pluginDoc::getWorkspaceSizeV3, "in"_a, "out"_a, IPluginV3Doc::get_workspace_size)
        .def("configure_plugin", &pluginDoc::configurePluginV3, "in"_a, "out"_a, IPluginV3Doc::configure_plugin)
        .def("supports_format_combination", &pluginDoc::supportsFormatCombinationV3, "pos"_a, "in_out"_a,
            "num_inputs"_a, IPluginV3Doc::supports_format_combination)
        .def("get_valid_tactics", &pluginDoc::getValidTactics, IPluginV3Doc::get_valid_tactics);

    py::class_<IPluginV3OneBuildV2, IPluginV3OneBuild, IPluginCapability, IVersionedInterface,
        PyIPluginV3OneBuildV2Impl, std::unique_ptr<IPluginV3OneBuildV2>>(
        m, "IPluginV3OneBuildV2", IPluginV3Doc::ipluginv3onebuildv2_descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3OneBuildV2&>())
        // The following defs are only for documenting the API for Python-based plugins
        .def("get_aliased_input", &pluginDoc::getAliasedInput, IPluginV3Doc::get_valid_tactics);

    py::class_<IPluginV3QuickCore, IPluginCapability, IVersionedInterface, PyIPluginV3QuickCoreImpl,
        std::unique_ptr<IPluginV3QuickCore>>(m, "IPluginV3QuickCore", py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3QuickCore&>())
        .def_property("plugin_name", &IPluginV3QuickCore::getPluginName,
            py::cpp_function(
                &helpers::setPluginName<IPluginV3QuickCore, PyIPluginV3QuickCoreImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginV3QuickCore::getPluginVersion,
            py::cpp_function(
                &helpers::setPluginVersion<IPluginV3QuickCore, PyIPluginV3QuickCoreImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_namespace", &IPluginV3QuickCore::getPluginNamespace,
            py::cpp_function(
                &helpers::setPluginNamespace<IPluginV3QuickCore, PyIPluginV3QuickCoreImpl>, py::keep_alive<1, 2>{}));

    py::class_<IPluginV3QuickBuild, IPluginCapability, IVersionedInterface, PyIPluginV3QuickBuildImpl,
        std::unique_ptr<IPluginV3QuickBuild>>(m, "IPluginV3QuickBuild", py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3QuickBuild&>());

    py::class_<IPluginV3QuickAOTBuild, IPluginV3QuickBuild, IPluginCapability, IVersionedInterface,
        PyIPluginV3QuickAOTBuildImpl, std::unique_ptr<IPluginV3QuickAOTBuild>>(
        m, "IPluginV3QuickAOTBuild", py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3QuickAOTBuild&>());

    py::class_<IPluginV3QuickRuntime, IPluginCapability, IVersionedInterface, PyIPluginV3QuickRuntimeImpl,
        std::unique_ptr<IPluginV3QuickRuntime>>(m, "IPluginV3QuickRuntime", py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3QuickRuntime&>());

    py::class_<IPluginV3OneRuntime, IPluginCapability, IVersionedInterface, PyIPluginV3OneRuntimeImpl,
        std::unique_ptr<IPluginV3OneRuntime>>(
        m, "IPluginV3OneRuntime", IPluginV3Doc::ipluginv3oneruntime_descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const IPluginV3OneRuntime&>())
        // The following defs are only for documenting the API for Python-based plugins
        .def("on_shape_change", &pluginDoc::onShapeChange, "in"_a, "out"_a, IPluginV3Doc::on_shape_change)
        .def("set_tactic", &pluginDoc::setTactic, "tactic"_a, IPluginV3Doc::set_tactic)
        .def("get_fields_to_serialize", &pluginDoc::getFieldsToSerialize, IPluginV3Doc::get_fields_to_serialize)
        .def("enqueue", &pluginDoc::enqueueV3, "input_desc"_a, "output_desc"_a, "inputs"_a, "outputs"_a, "workspace"_a,
            "stream"_a, IPluginV3Doc::enqueue)
        .def("attach_to_context", &pluginDoc::attachToContext, "resource_context"_a, IPluginV3Doc::attach_to_context);

    py::class_<IPluginCreatorInterface, IVersionedInterface>(
        m, "IPluginCreatorInterface", IPluginCreatorInterfaceDoc::descr, py::module_local());

    py::class_<IPluginCreator, IPluginCreatorImpl, IPluginCreatorInterface, IVersionedInterface>(
        m, "IPluginCreator", IPluginCreatorDoc::descr, py::module_local())
        .def(py::init<>())
        .def_property("name", &IPluginCreator::getPluginName,
            py::cpp_function(&helpers::setPluginName<IPluginCreator, IPluginCreatorImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginCreator::getPluginVersion,
            py::cpp_function(&helpers::setPluginVersion<IPluginCreator, IPluginCreatorImpl>, py::keep_alive<1, 2>{}))
        .def_property("field_names", &helpers::getFieldNames<IPluginCreator>,
            py::cpp_function(
                &helpers::setPluginCreatorFieldNames<IPluginCreator, IPluginCreatorImpl>, py::keep_alive<1, 2>{}),
            py::return_value_policy::reference_internal)
        .def_property("plugin_namespace", &IPluginCreator::getPluginNamespace,
            py::cpp_function(&IPluginCreator::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("create_plugin", lambdas::creator_create_plugin, "name"_a, "field_collection"_a,
            IPluginCreatorDoc::create_plugin)
        .def("deserialize_plugin", lambdas::deserialize_plugin, "name"_a, "serialized_plugin"_a,
            IPluginCreatorDoc::deserialize_plugin)
        .def("deserialize_plugin", &pluginDoc::deserializePlugin, "name"_a, "serialized_plugin"_a,
            IPluginCreatorDoc::deserialize_plugin_python) // Should never be used. For documenting C++ -> Python API
                                                          // only.
        ;

    py::class_<IPluginCreatorV3One, IPluginCreatorV3OneImpl, IPluginCreatorInterface, IVersionedInterface>(
        m, "IPluginCreatorV3One", IPluginCreatorV3OneDoc::descr, py::module_local())
        .def(py::init<>())
        .def_property("name", &IPluginCreatorV3One::getPluginName,
            py::cpp_function(
                &helpers::setPluginName<IPluginCreatorV3One, IPluginCreatorV3OneImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginCreatorV3One::getPluginVersion,
            py::cpp_function(
                &helpers::setPluginVersion<IPluginCreatorV3One, IPluginCreatorV3OneImpl>, py::keep_alive<1, 2>{}))
        .def_property("field_names", &helpers::getFieldNames<IPluginCreatorV3One>,
            py::cpp_function(&helpers::setPluginCreatorFieldNames<IPluginCreatorV3One, IPluginCreatorV3OneImpl>,
                py::keep_alive<1, 2>{}),
            py::return_value_policy::reference_internal)
        .def_property("plugin_namespace", &IPluginCreatorV3One::getPluginNamespace,
            py::cpp_function(
                &helpers::setPluginNamespace<IPluginCreatorV3One, IPluginCreatorV3OneImpl>, py::keep_alive<1, 2>{}))
        .def("create_plugin", lambdas::creator_create_plugin_v3, "name"_a, "field_collection"_a, "phase"_a,
            IPluginCreatorV3OneDoc::create_plugin);

    py::class_<IPluginCreatorV3Quick, IPluginCreatorV3QuickImpl, IPluginCreatorInterface, IVersionedInterface>(
        m, "IPluginCreatorV3Quick", py::module_local())
        .def(py::init<>())
        .def_property("name", &IPluginCreatorV3Quick::getPluginName,
            py::cpp_function(
                &helpers::setPluginName<IPluginCreatorV3Quick, IPluginCreatorV3QuickImpl>, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginCreatorV3Quick::getPluginVersion,
            py::cpp_function(
                &helpers::setPluginVersion<IPluginCreatorV3Quick, IPluginCreatorV3QuickImpl>, py::keep_alive<1, 2>{}))
        .def_property("field_names", &helpers::getFieldNames<IPluginCreatorV3Quick>,
            py::cpp_function(&helpers::setPluginCreatorFieldNames<IPluginCreatorV3Quick, IPluginCreatorV3QuickImpl>,
                py::keep_alive<1, 2>{}),
            py::return_value_policy::reference_internal)
        .def_property("plugin_namespace", &IPluginCreatorV3Quick::getPluginNamespace,
            py::cpp_function(
                &helpers::setPluginNamespace<IPluginCreatorV3Quick, IPluginCreatorV3QuickImpl>, py::keep_alive<1, 2>{}))
        .def("create_plugin", lambdas::creator_create_plugin_v3_quick, "name"_a, "namespace"_a, "field_collection"_a,
            "phase"_a, "quickPluginType"_a);

    py::class_<IPluginResourceContext, std::unique_ptr<IPluginResourceContext, py::nodelete>>(
        m, "IPluginResourceContext", IPluginResourceContextDoc::descr, py::module_local())
        // return_value_policy::reference_internal is default for the following
        .def_property_readonly("error_recorder", &IPluginResourceContext::getErrorRecorder)
        .def_property_readonly("gpu_allocator", &IPluginResourceContext::getGpuAllocator);

    py::class_<IPluginResource, IVersionedInterface, PyIPluginResourceImpl,
        std::unique_ptr<IPluginResource, py::nodelete>>(
        m, "IPluginResource", IPluginResourceDoc::descr, py::module_local())
        .def(py::init<>())
        // return_value_policy::reference_internal is default for the following
        .def("release", &pluginDoc::release, IPluginResourceDoc::release)
        .def("clone", &pluginDoc::clonePluginResource, IPluginResourceDoc::clone);

    py::class_<IPluginRegistry, std::unique_ptr<IPluginRegistry, py::nodelete>>(
        m, "IPluginRegistry", IPluginRegistryDoc::descr, py::module_local())
        .def_property_readonly("plugin_creator_list", lambdas::get_plugin_creator_list)
        .def_property_readonly("all_creators", lambdas::get_all_creators)
        .def_property_readonly("all_creators_recursive", lambdas::get_all_creators_recursive)
        .def("register_creator",
            py::overload_cast<IPluginCreator&, AsciiChar const* const>(&IPluginRegistry::registerCreator), "creator"_a,
            "plugin_namespace"_a = "", py::keep_alive<1, 2>{}, IPluginRegistryDoc::register_creator_iplugincreator)
        .def("deregister_creator", py::overload_cast<IPluginCreator const&>(&IPluginRegistry::deregisterCreator),
            "creator"_a, IPluginRegistryDoc::deregister_creator_iplugincreator)
        .def("get_plugin_creator", &IPluginRegistry::getPluginCreator, "type"_a, "version"_a, "plugin_namespace"_a = "",
            py::return_value_policy::reference_internal, IPluginRegistryDoc::get_plugin_creator)
        .def_property("error_recorder", &IPluginRegistry::getErrorRecorder,
            py::cpp_function(&IPluginRegistry::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property("parent_search_enabled", &IPluginRegistry::isParentSearchEnabled,
            py::cpp_function(&IPluginRegistry::setParentSearchEnabled, py::keep_alive<1, 2>{}))
        .def("load_library", &IPluginRegistry::loadLibrary, "plugin_path"_a,
            py::return_value_policy::reference_internal, IPluginRegistryDoc::load_library)
        .def("deregister_library", &IPluginRegistry::deregisterLibrary, "handle"_a,
            IPluginRegistryDoc::deregister_library)
        .def("register_creator",
            py::overload_cast<IPluginCreatorInterface&, AsciiChar const* const>(&IPluginRegistry::registerCreator),
            "creator"_a, "plugin_namespace"_a = "", py::keep_alive<1, 2>{}, IPluginRegistryDoc::register_creator)
        .def("deregister_creator",
            py::overload_cast<IPluginCreatorInterface const&>(&IPluginRegistry::deregisterCreator), "creator"_a,
            IPluginRegistryDoc::deregister_creator)
        .def("get_creator", lambdas::get_creator, "type"_a, "version"_a, "plugin_namespace"_a = "",
            py::return_value_policy::reference_internal, IPluginRegistryDoc::get_creator)
        .def("acquire_plugin_resource", &IPluginRegistry::acquirePluginResource, "key"_a, "resource"_a,
            py::return_value_policy::reference_internal, IPluginRegistryDoc::acquire_plugin_resource)
        .def("release_plugin_resource", &IPluginRegistry::releasePluginResource, "key"_a,
            IPluginRegistryDoc::release_plugin_resource);

    py::enum_<PluginCreatorVersion>(m, "PluginCreatorVersion", PluginCreatorVersionDoc::descr, py::module_local())
        .value("V1", PluginCreatorVersion::kV1)
        .value("V1_PYTHON", PluginCreatorVersion::kV1_PYTHON);

    m.add_object("_plugin_registry", py::none());

    m.def(
        "get_plugin_registry",
        [m]() {
            if (m.attr("_plugin_registry").is_none())
            {
                m.attr("_plugin_registry") = py::cast(getPluginRegistry());
            }
            return m.attr("_plugin_registry");
        },
        py::return_value_policy::reference, FreeFunctionsDoc::get_plugin_registry);

    py::enum_<PluginCapabilityType>(
        m, "PluginCapabilityType", py::arithmetic{}, PluginCapabilityTypeDoc::descr, py::module_local())
        .value("CORE", PluginCapabilityType::kCORE)
        .value("BUILD", PluginCapabilityType::kBUILD)
        .value("RUNTIME", PluginCapabilityType::kRUNTIME);

    py::enum_<TensorRTPhase>(m, "TensorRTPhase", py::arithmetic{}, TensorRTPhaseDoc::descr, py::module_local())
        .value("BUILD", TensorRTPhase::kBUILD)
        .value("RUNTIME", TensorRTPhase::kRUNTIME);

    py::enum_<QuickPluginCreationRequest>(m, "QuickPluginCreationRequest", py::arithmetic{}, py::module_local())
        .value("UNKNOWN", QuickPluginCreationRequest::kUNKNOWN)
        .value("PREFER_JIT", QuickPluginCreationRequest::kPREFER_JIT)
        .value("PREFER_AOT", QuickPluginCreationRequest::kPREFER_AOT)
        .value("STRICT_JIT", QuickPluginCreationRequest::kSTRICT_JIT)
        .value("STRICT_AOT", QuickPluginCreationRequest::kSTRICT_AOT);

    py::class_<IKernelLaunchParams, std::unique_ptr<IKernelLaunchParams, py::nodelete>>(
        m, "KernelLaunchParams", py::module_local())
        .def_property("grid_x", &IKernelLaunchParams::getGridX, &IKernelLaunchParams::setGridX)
        .def_property("grid_y", &IKernelLaunchParams::getGridY, &IKernelLaunchParams::setGridY)
        .def_property("grid_z", &IKernelLaunchParams::getGridZ, &IKernelLaunchParams::setGridZ)
        .def_property("block_x", &IKernelLaunchParams::getBlockX, &IKernelLaunchParams::setBlockX)
        .def_property("block_y", &IKernelLaunchParams::getBlockY, &IKernelLaunchParams::setBlockY)
        .def_property("block_z", &IKernelLaunchParams::getBlockZ, &IKernelLaunchParams::setBlockZ)
        .def_property("shared_mem", &IKernelLaunchParams::getSharedMem, &IKernelLaunchParams::setSharedMem);
    py::enum_<PluginArgType>(m, "PluginArgType", py::arithmetic{}, py::module_local())
        .value("INT", PluginArgType::kINT);

    py::enum_<PluginArgDataType>(m, "PluginArgDataType", py::arithmetic{}, py::module_local())
        .value("INT8", PluginArgDataType::kINT8)
        .value("INT16", PluginArgDataType::kINT16)
        .value("INT32", PluginArgDataType::kINT32);

    py::class_<ISymExpr, SymExprImpl, std::unique_ptr<ISymExpr, py::nodelete>>(m, "ISymExpr", py::module_local())
        .def(py::init<>())
        .def_property("type", nullptr, lambdas::ISymExpr_set_type)
        .def_property("dtype", nullptr, lambdas::ISymExpr_set_data_type)
        .def_property("expr", nullptr, lambdas::ISymExpr_set_expr);

    py::class_<ISymExprs, std::unique_ptr<ISymExprs, py::nodelete>>(m, "ISymExprs", py::module_local())
        .def_property("nbSymExprs", &ISymExprs::getNbSymExprs, &ISymExprs::setNbSymExprs)
        .def("__len__", &ISymExprs::getNbSymExprs)
        .def("__getitem__", &ISymExprs::getSymExpr)
        .def("__setitem__", &ISymExprs::setSymExpr);
#if EXPORT_ALL_BINDINGS
    m.def("get_builder_plugin_registry", &getBuilderPluginRegistry, py::return_value_policy::reference,
        FreeFunctionsDoc::get_builder_plugin_registry);
    m.def("init_libnvinfer_plugins", &initLibNvInferPlugins, "logger"_a, "namespace"_a,
        FreeFunctionsDoc::init_libnvinfer_plugins);
#endif

} // Plugin
} // namespace tensorrt
