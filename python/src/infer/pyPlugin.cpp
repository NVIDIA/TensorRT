/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "infer/pyPluginDoc.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

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
    catch (const py::cast_error& e)                                                                                    \
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

inline PluginCreatorVersion getPluginCreatorVersion(int32_t const version)
{
    return static_cast<PluginCreatorVersion>(version >> kTHREE_BYTE_SHIFT & kBYTE_MASK);
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
    PyIPluginV2DynamicExtImpl(const PyIPluginV2DynamicExt& a) {};

    int32_t getNbOutputs() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if(!mIsNbOutputsInitialized)
            {
                utils::throwPyError(PyExc_AttributeError, "num_outputs not initialized");
            }
            return mNbOutputs;
        }
        PLUGIN_API_CATCH("num_outputs")
        return -1;
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
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
            for(int32_t idx = 0; idx < nbInputs + nbOutputs; ++idx)
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

            try{
                py::object pyResult = pyInitialize();
            }
            catch (py::error_already_set &e)
            {
                std::cerr << "[ERROR] Exception thrown from initialize() " << e.what() << std::endl;
                return -1;
            }
            return 0;
        }
        PLUGIN_API_CATCH("initialize")
        return -1;
    }

    void terminate() noexcept override {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyTerminate = py::get_override(static_cast<PyIPluginV2DynamicExt*>(this), "terminate");

            // if no implementation is provided for terminate(), it is defaulted to `pass`
            if(pyTerminate)
            {
                pyTerminate();
            }
        }
        PLUGIN_API_CATCH("terminate")
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            
            py::function pyEnqueue
                = utils::getOverride(static_cast<PyIPluginV2DynamicExt*>(this), "enqueue");
            if (!pyEnqueue)
            {
                utils::throwPyError(PyExc_RuntimeError, "no implementation provided for enqueue()");
            }

            std::vector<PluginTensorDesc> inVector;
            for(int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inVector.push_back(*(inputDesc + idx));
            }
            std::vector<PluginTensorDesc> outVector;
            for(int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outVector.push_back(*(outputDesc + idx));
            }

            std::vector<intptr_t> inPtrs;
            for(int32_t idx = 0; idx < mNbInputs; ++idx)
            {
                inPtrs.push_back(reinterpret_cast<intptr_t>(inputs[idx]));
                
            }
            std::vector<intptr_t> outPtrs;
            for(int32_t idx = 0; idx < mNbOutputs; ++idx)
            {
                outPtrs.push_back(reinterpret_cast<intptr_t>(outputs[idx]));
                
            }

            intptr_t workspacePtr = reinterpret_cast<intptr_t>(workspace);
            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);

            try{
                pyEnqueue(inVector, outVector, inPtrs, outPtrs, workspacePtr, cudaStreamPtr);
            }
            catch (py::error_already_set &e)
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

            py::function pyGetSerializationSize = py::get_override(static_cast<PyIPluginV2DynamicExt const*>(this), "get_serialization_size");
            
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
            if(!mIsPluginTypeInitialized)
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
            if(!mIsPluginVersionInitialized)
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
            
            py::function pyClone
                = utils::getOverride(static_cast<const PyIPluginV2DynamicExt*>(this), "clone");
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
            if(!mIsNamespaceInitialized)
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
            for(int32_t idx = 0; idx < nbInputs; ++idx)
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


    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
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
            for(int32_t idx = 0; idx < nbInputs; ++idx)
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

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
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
            for(int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(in + idx));
            }

            std::vector<DynamicPluginTensorDesc> outVector;
            for(int32_t idx = 0; idx < nbOutputs; ++idx)
            {
                outVector.push_back(*(out + idx));
            }

            pyConfigurePlugin(inVector, outVector);
        }
        PLUGIN_API_CATCH("configure_plugin")
    }

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetWorkspaceSize = py::get_override(static_cast<PyIPluginV2DynamicExt const*>(this), "get_workspace_size");

            if (!pyGetWorkspaceSize)
            {
                // if no implementation is provided for get_workspace_size(), default to zero workspace size required
                return 0;
            }

            std::vector<PluginTensorDesc> inVector;
            for(int32_t idx = 0; idx < nbInputs; ++idx)
            {
                inVector.push_back(*(inputs + idx));
            }

            std::vector<PluginTensorDesc> outVector;
            for(int32_t idx = 0; idx < nbOutputs; ++idx)
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

    AsciiChar const* getPluginName() const noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            if(!mIsNameInitialized)
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
            if(!mIsPluginVersionInitialized)
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
            if(!mIsFCInitialized)
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
            
            py::function pyCreatePlugin
                = utils::getOverride(static_cast<IPluginCreator*>(this), "create_plugin");
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

    IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override
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

            py::handle handle = pyDeserializePlugin(nameString, py::bytes(static_cast<const char*>(serialData), serialLength)).release();
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
            if(!mIsNamespaceInitialized)
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

    void setName(std::string name)
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
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginCreatorVersion::kV1_PYTHON) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

    nvinfer1::PluginFieldCollection mFC;
    std::string mNamespace;
    std::string mName;
    std::string mPluginVersion;

    bool mIsFCInitialized{false};
    bool mIsNamespaceInitialized{false};
    bool mIsNameInitialized{false};
    bool mIsPluginVersionInitialized{false};
};


// Long lambda functions should go here rather than being inlined into the bindings (1 liners are OK).
namespace lambdas
{
// For IPluginV2
static const auto IPluginV2_get_output_shape = [](IPluginV2& self, int32_t const index, std::vector<Dims> const& inputShapes) {
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
static const auto IPluginV2_execute_async = [](IPluginV2& self, int32_t batchSize, const std::vector<const void*>& inputs,
                                                std::vector<void*>& outputs, void* workspace, long stream) {
    return self.enqueue(batchSize, inputs.data(), outputs.data(), workspace, reinterpret_cast<cudaStream_t>(stream));
};

static const auto IPluginV2_set_num_outputs = [](IPluginV2& self, int32_t numOutputs) {
    if(getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
    {
        auto plugin = static_cast<PyIPluginV2DynamicExtImpl*>(&self);
        plugin->setNbOutputs(numOutputs);
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: num_outputs is read-only for C++ plugins");
};

static const auto IPluginV2_set_plugin_type = [](IPluginV2& self, std::string pluginType) {
    if(getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
    {
        auto plugin = reinterpret_cast<PyIPluginV2DynamicExtImpl*>(&self);
        plugin->setPluginType(std::move(pluginType));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute: plugin_type is read-only for C++ plugins");
};

static const auto IPluginV2_set_plugin_version = [](IPluginV2& self, std::string pluginVersion) {
    if(getPluginVersion(self.getTensorRTVersion()) == PluginVersion::kV2_DYNAMICEXT_PYTHON)
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
    return std::move(result);
}

static const auto configure_plugin
    = [](IPluginV2Ext& self, std::vector<Dims> const& inputShapes, std::vector<Dims> const& outputShapes,
          std::vector<DataType> const& inputTypes, std::vector<DataType> const& outputTypes,
          std::vector<bool> const& inputIsBroadcasted, std::vector<bool> const& outputIsBroadcasted, TensorFormat format,
          int32_t maxBatchSize) {
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

// For IPluginCreator
static const auto creator_create_plugin
    = [](IPluginCreator& self, std::string const& name, PluginFieldCollection const* fc) {
          return self.createPlugin(name.c_str(), fc);
      };

static const auto get_field_names = [](IPluginCreator& self) -> const PluginFieldCollection* {
    const PluginFieldCollection* fieldCollection = self.getFieldNames();
    if (!fieldCollection)
    {
        return &EMPTY_PLUGIN_FIELD_COLLECTION;
    }
    return fieldCollection;
};

static const auto deserialize_plugin = [](IPluginCreator& self, std::string const& name, py::buffer& serializedPlugin) {
    py::buffer_info info = serializedPlugin.request();
    return self.deserializePlugin(name.c_str(), info.ptr, info.size * info.itemsize);
};

static const auto IPluginCreator_set_field_names = [](IPluginCreator& self, PluginFieldCollection pfc) {
    if(getPluginCreatorVersion(self.getTensorRTVersion()) == PluginCreatorVersion::kV1_PYTHON)
    {
        auto pluginCreator = static_cast<IPluginCreatorImpl*>(&self);
        pluginCreator->setFieldNames(pfc);
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute");
};

static const auto IPluginCreator_set_name = [](IPluginCreator& self, std::string name) {
    if(getPluginCreatorVersion(self.getTensorRTVersion()) == PluginCreatorVersion::kV1_PYTHON)
    {
        auto pluginCreator = static_cast<IPluginCreatorImpl*>(&self);
        pluginCreator->setName(std::move(name));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute");
};

static const auto IPluginCreator_set_plugin_version = [](IPluginCreator& self, std::string pluginVersion) {
    if(getPluginCreatorVersion(self.getTensorRTVersion()) == PluginCreatorVersion::kV1_PYTHON)
    {
        auto pluginCreator = static_cast<IPluginCreatorImpl*>(&self);
        pluginCreator->setPluginVersion(std::move(pluginVersion));
        return;
    }
    utils::throwPyError(PyExc_AttributeError, "Can't set attribute");
};

// For base Dims class
static const auto dimsexprs_vector_constructor = [](std::vector<IDimensionExpr const*> const& in) {
    // This is required, because otherwise MAX_DIMS will not be resolved at compile time.
    int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
    PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
            "Input length " + std::to_string(in.size()) + ". Max expected length is " + std::to_string(maxDims));

    // Create the Dims object.
    DimsExprs* self = new DimsExprs{};
    self->nbDims = in.size();
    for (int32_t i = 0; i < in.size(); ++i)
        self->d[i] = in[i];
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

} // namespace lambdas

// NOTE: Fake bindings are provided solely to document the API in the C++ -> Python direction for these methods
// These bindings will never be called.

py::bytes docSerialize(PyIPluginV2DynamicExt& self)
{
    return py::bytes();
}

DataType docGetOutputDataType(PyIPluginV2DynamicExt& self, int32_t index, std::vector<DataType> const& inputTypes)
{
    return DataType{};
};

PyIPluginV2DynamicExt* docDeserializePlugin(
    PyIPluginV2DynamicExt& self, std::string const& name, py::bytes const& serializedPlugin)
{
    return nullptr;
}

DimsExprs docGetOutputDimensions(
    PyIPluginV2DynamicExt& self, int32_t outputIndex, std::vector<DimsExprs> const& inputs, IExprBuilder& exprBuilder)
{
    return DimsExprs{};
}

void docConfigurePlugin(PyIPluginV2DynamicExt& self, std::vector<DynamicPluginTensorDesc> const& in,
    std::vector<DynamicPluginTensorDesc> const& out)
{
}

size_t docGetWorkspaceSize(PyIPluginV2DynamicExt& self, std::vector<PluginTensorDesc> const& inputDesc,
    std::vector<PluginTensorDesc> const& outputDesc)
{
    return 0U;
}

bool docSupportsFormatCombination(
    PyIPluginV2DynamicExt& self, int32_t pos, std::vector<PluginTensorDesc> const& inOut, int32_t nbInputs)
{
    return false;
}

void docEnqueue(PyIPluginV2DynamicExt& self, std::vector<PluginTensorDesc> const& inputDesc,
    std::vector<PluginTensorDesc> const& outputDesc, const std::vector<intptr_t>& inputs,
    std::vector<intptr_t>& outputs, intptr_t workspace, long stream)
{
}

int32_t docInitialize(PyIPluginV2DynamicExt& self)
{
    return -1;
}

void docTerminate(PyIPluginV2DynamicExt& self) {}

void docDestroy(PyIPluginV2DynamicExt& self) {}

PyIPluginV2DynamicExt* docClone(PyIPluginV2DynamicExt& self)
{
    return nullptr;
}

size_t docGetSerializationSize(PyIPluginV2DynamicExt& self)
{
    return 0U;
}

void bindPlugin(py::module& m)
{
    py::class_<IDimensionExpr, PyIDimensionExprImpl, std::unique_ptr<IDimensionExpr, py::nodelete>>(m, "IDimensionExpr", IDimensionExprDoc::descr, py::module_local())
        .def("isConstant", &IDimensionExpr::isConstant, IDimensionExprDoc::is_constant)
        .def("getConstantValue", &IDimensionExpr::getConstantValue, IDimensionExprDoc::get_constant_value);

    py::class_<DimsExprs>(m, "DimsExprs", DimsExprsDoc::descr, py::module_local())
        .def(py::init<>())
        // Allows for construction from python lists and tuples.
        .def(py::init(lambdas::dimsexprs_vector_constructor))
        // These functions allow us to use DimsExprs like an iterable.
        .def("__len__", lambdas::dimsexprs_len)
        .def("__getitem__", lambdas::dimsexprs_getter)
        .def("__setitem__", lambdas::dimsexprs_setter);

    py::class_<IExprBuilder, PyIExprBuilderImpl, std::unique_ptr<IExprBuilder, py::nodelete>>(m, "IExprBuilder", IExprBuilderDoc::descr, py::module_local())
        .def(py::init<>())
        .def("constant", &IExprBuilder::constant, "value"_a, IExprBuilderDoc::constant, py::return_value_policy::reference_internal)
        .def("operation", &IExprBuilder::operation, "op"_a, "first"_a, "second"_a, IExprBuilderDoc::operation, py::return_value_policy::reference_internal);

    py::class_<PluginTensorDesc>(m, "PluginTensorDesc", PluginTensorDescDoc::descr, py::module_local())
        .def(py::init<>())
        .def_readwrite("dims", &PluginTensorDesc::dims)
        .def_readwrite("type", &PluginTensorDesc::type)
        .def_readwrite("format", &PluginTensorDesc::format)
        .def_readwrite("scale", &PluginTensorDesc::scale);

    py::class_<DynamicPluginTensorDesc>(m, "DynamicPluginTensorDesc", DynamicPluginTensorDescDoc::descr, py::module_local())
        .def(py::init<>())
        .def_readwrite("desc", &DynamicPluginTensorDesc::desc)
        .def_readwrite("min", &DynamicPluginTensorDesc::min)
        .def_readwrite("max", &DynamicPluginTensorDesc::max);

    py::class_<IPluginV2>(m, "IPluginV2", IPluginV2Doc::descr, py::module_local())
        .def_property("num_outputs", &IPluginV2::getNbOutputs, lambdas::IPluginV2_set_num_outputs)
        .def_property_readonly("tensorrt_version", &IPluginV2::getTensorRTVersion)
        .def_property("plugin_type", &IPluginV2::getPluginType, py::cpp_function(lambdas::IPluginV2_set_plugin_type, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginV2::getPluginVersion, py::cpp_function(lambdas::IPluginV2_set_plugin_version, py::keep_alive<1, 2>{}))
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

    py::class_<IPluginV2DynamicExt, IPluginV2, std::unique_ptr<IPluginV2DynamicExt, py::nodelete>>(m, "IPluginV2DynamicExtBase", py::module_local());

    py::class_<PyIPluginV2DynamicExt, IPluginV2DynamicExt, IPluginV2, PyIPluginV2DynamicExtImpl, std::unique_ptr<PyIPluginV2DynamicExt>>(m, "IPluginV2DynamicExt", IPluginV2DynamicExtDoc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<const PyIPluginV2DynamicExt&>())
        // The following defs are only for documenting the API for Python-based plugins
        .def("initialize", &docInitialize, IPluginV2DynamicExtDoc::initialize)
        .def("terminate", &docTerminate, IPluginV2DynamicExtDoc::terminate)
        .def("serialize", &docSerialize, IPluginV2DynamicExtDoc::serialize)
        .def("get_output_datatype", &docGetOutputDataType, "index"_a, "input_types"_a, IPluginV2DynamicExtDoc::get_output_data_type)
        .def("destroy", &docDestroy, IPluginV2DynamicExtDoc::destroy)
        .def("get_serialization_size", &docGetSerializationSize, IPluginV2DynamicExtDoc::get_serialization_size)
        .def("get_output_dimensions", &docGetOutputDimensions, "output_index"_a, "inputs"_a, "expr_builder"_a, IPluginV2DynamicExtDoc::get_output_dimensions)
        .def("get_workspace_size", &docGetWorkspaceSize, "in"_a, "out"_a, IPluginV2DynamicExtDoc::get_workspace_size)
        .def("configure_plugin", &docConfigurePlugin, "pos"_a, "in_out"_a, IPluginV2DynamicExtDoc::configure_plugin)
        .def("supports_format_combination", &docSupportsFormatCombination, "pos"_a, "in_out"_a, "num_inputs"_a, IPluginV2DynamicExtDoc::supports_format_combination)
        .def("enqueue", &docEnqueue, "input_desc"_a, "output_desc"_a, "inputs"_a, "outputs"_a, "workspace"_a, "stream"_a, IPluginV2DynamicExtDoc::enqueue)
        .def("clone", &docClone, IPluginV2DynamicExtDoc::clone);

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
        .value("FP8", PluginFieldType::kFP8);

    py::class_<PluginField>(m, "PluginField", PluginFieldDoc::descr, py::module_local())
        .def(py::init(lambdas::plugin_field_default_constructor), "name"_a = "", py::keep_alive<1, 2>{})
        .def(py::init(lambdas::plugin_field_constructor), "name"_a, "data"_a,
            "type"_a = nvinfer1::PluginFieldType::kUNKNOWN, py::keep_alive<1, 2>{}, py::keep_alive<1, 3>{})
        .def_property(
            "name", [](PluginField& self) { return self.name; },
            py::cpp_function(
                [](PluginField& self, FallbackString& name) { self.name = name.c_str(); }, py::keep_alive<1, 2>{}))
        .def_property(
            "data", [](PluginField& self) {
                switch (self.type)
                {
                case PluginFieldType::kINT32:
                    return py::array(self.length, static_cast<int32_t const*>(self.data)); 
                    break;
                case PluginFieldType::kINT8:
                    return py::array(self.length, static_cast<int8_t const*>(self.data)); 
                    break;
                case PluginFieldType::kINT16:
                    return py::array(self.length, static_cast<int16_t const*>(self.data)); 
                    break;
                case PluginFieldType::kFLOAT16:
                    // TODO: Figure out how to handle float16 correctly here
                    return py::array(self.length, static_cast<float const*>(self.data)); 
                    break;
                case PluginFieldType::kFLOAT32:
                    return py::array(self.length, static_cast<float const*>(self.data)); 
                    break;
                case PluginFieldType::kFLOAT64:
                    return py::array(self.length, static_cast<double const*>(self.data)); 
                    break;
                case PluginFieldType::kCHAR:
                    return py::array(self.length, static_cast<char const*>(self.data)); 
                    break;
                default:
                    assert(false && "No known conversion for returning data from PluginField");
                    break;
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

    py::class_<IPluginCreator, IPluginCreatorImpl>(m, "IPluginCreator", IPluginCreatorDoc::descr, py::module_local())
        .def(py::init<>())
        .def_property_readonly("tensorrt_version", &IPluginCreator::getTensorRTVersion)
        .def_property("name", &IPluginCreator::getPluginName, py::cpp_function(lambdas::IPluginCreator_set_name, py::keep_alive<1, 2>{}))
        .def_property("plugin_version", &IPluginCreator::getPluginVersion, py::cpp_function(lambdas::IPluginCreator_set_plugin_version, py::keep_alive<1, 2>{}))
        .def_property("field_names", lambdas::get_field_names, py::cpp_function(lambdas::IPluginCreator_set_field_names, py::keep_alive<1, 2>{}), py::return_value_policy::reference_internal)
        .def_property("plugin_namespace", &IPluginCreator::getPluginNamespace,
            py::cpp_function(&IPluginCreator::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("create_plugin", lambdas::creator_create_plugin, "name"_a, "field_collection"_a,
            IPluginCreatorDoc::create_plugin)
        .def("deserialize_plugin", lambdas::deserialize_plugin, "name"_a, "serialized_plugin"_a,
            IPluginCreatorDoc::deserialize_plugin)
        .def("deserialize_plugin", &docDeserializePlugin, "name"_a, "serialized_plugin"_a,
            IPluginCreatorDoc::deserialize_plugin_python)  // Should never be used. For documenting C++ -> Python API only.
        ; 

    py::class_<IPluginRegistry, std::unique_ptr<IPluginRegistry, py::nodelete>>(
        m, "IPluginRegistry", IPluginRegistryDoc::descr, py::module_local())
        .def_property_readonly("plugin_creator_list", lambdas::get_plugin_creator_list)
        .def("register_creator", &IPluginRegistry::registerCreator, "creator"_a, "plugin_namespace"_a = "",
            py::keep_alive<1, 2>{}, IPluginRegistryDoc::register_creator)
        .def("deregister_creator", &IPluginRegistry::deregisterCreator, "creator"_a,
            IPluginRegistryDoc::deregister_creator)
        .def("get_plugin_creator", &IPluginRegistry::getPluginCreator, "type"_a, "version"_a, "plugin_namespace"_a = "",
            py::return_value_policy::reference_internal, IPluginRegistryDoc::get_plugin_creator)
        .def_property("error_recorder", &IPluginRegistry::getErrorRecorder,
            py::cpp_function(&IPluginRegistry::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property("parent_search_enabled", &IPluginRegistry::isParentSearchEnabled,
            py::cpp_function(&IPluginRegistry::setParentSearchEnabled, py::keep_alive<1, 2>{}))
        .def("load_library", &IPluginRegistry::loadLibrary, "plugin_path"_a,
            py::return_value_policy::reference_internal, IPluginRegistryDoc::load_library)
        .def("deregister_library", &IPluginRegistry::deregisterLibrary, "handle"_a,
            IPluginRegistryDoc::deregister_library);

    py::enum_<PluginCreatorVersion>(m, "PluginCreatorVersion", PluginCreatorVersionDoc::descr, py::module_local())
        .value("V1", PluginCreatorVersion::kV1)
        .value("V1_PYTHON", PluginCreatorVersion::kV1_PYTHON);

    m.def("get_plugin_registry", &getPluginRegistry, py::return_value_policy::reference,
        FreeFunctionsDoc::get_plugin_registry);

    py::enum_<DimensionOperation>(m, "DimensionOperation", py::arithmetic{}, DimensionOperationDoc::descr, py::module_local())
        .value("SUM", DimensionOperation::kSUM)
        .value("PROD", DimensionOperation::kPROD)
        .value("MAX", DimensionOperation::kMAX)
        .value("MIN", DimensionOperation::kMIN)
        .value("SUB", DimensionOperation::kSUB)
        .value("EQUAL", DimensionOperation::kEQUAL)
        .value("LESS", DimensionOperation::kLESS)
        .value("FLOOR_DIV", DimensionOperation::kFLOOR_DIV)
        .value("CEIL_DIV", DimensionOperation::kCEIL_DIV);

#if EXPORT_ALL_BINDINGS
    m.def("get_builder_plugin_registry", &getBuilderPluginRegistry, py::return_value_policy::reference,
        FreeFunctionsDoc::get_builder_plugin_registry);
    m.def("init_libnvinfer_plugins", &initLibNvInferPlugins, "logger"_a, "namespace"_a,
        FreeFunctionsDoc::init_libnvinfer_plugins);
#endif

} // Plugin
} // namespace tensorrt
