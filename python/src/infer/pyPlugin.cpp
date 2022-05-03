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

// This file contains all bindings related to plugins.
#include "ForwardDeclarations.h"
#include "infer/pyPluginDoc.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>

namespace tensorrt
{
using namespace nvinfer1;
using namespace nvinfer1::plugin;

constexpr PluginFieldCollection EMPTY_PLUGIN_FIELD_COLLECTION{0, nullptr};

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

} // namespace lambdas

void bindPlugin(py::module& m)
{
    py::class_<IPluginV2>(m, "IPluginV2", IPluginV2Doc::descr)
        .def_property_readonly("num_outputs", &IPluginV2::getNbOutputs)
        .def_property_readonly("tensorrt_version", &IPluginV2::getTensorRTVersion)
        .def_property_readonly("plugin_type", &IPluginV2::getPluginType)
        .def_property_readonly("plugin_version", &IPluginV2::getPluginVersion)
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

    py::class_<IPluginV2Ext, IPluginV2>(m, "IPluginV2Ext", IPluginV2ExtDoc::descr)
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

    py::enum_<PluginFieldType>(m, "PluginFieldType", PluginFieldTypeDoc::descr)
        .value("FLOAT16", PluginFieldType::kFLOAT16)
        .value("FLOAT32", PluginFieldType::kFLOAT32)
        .value("FLOAT64", PluginFieldType::kFLOAT64)
        .value("INT8", PluginFieldType::kINT8)
        .value("INT16", PluginFieldType::kINT16)
        .value("INT32", PluginFieldType::kINT32)
        .value("CHAR", PluginFieldType::kCHAR)
        .value("DIMS", PluginFieldType::kDIMS)
        .value("UNKNOWN", PluginFieldType::kUNKNOWN);

    py::class_<PluginField>(m, "PluginField", PluginFieldDoc::descr)
        .def(py::init(lambdas::plugin_field_default_constructor), "name"_a = "", py::keep_alive<1, 2>{})
        .def(py::init(lambdas::plugin_field_constructor), "name"_a, "data"_a,
            "type"_a = nvinfer1::PluginFieldType::kUNKNOWN, py::keep_alive<1, 2>{}, py::keep_alive<1, 3>{})
        .def_property("name", [](PluginField& self) { return self.name; },
            py::cpp_function(
                [](PluginField& self, FallbackString& name) { self.name = name.c_str(); }, py::keep_alive<1, 2>{}))
        .def_property("data", [](PluginField& self) { return self.data; },
            py::cpp_function(
                [](PluginField& self, py::buffer& buffer) {
                    py::buffer_info info = buffer.request();
                    self.data = info.ptr;
                },
                py::keep_alive<1, 2>{}))
        .def_readwrite("type", &PluginField::type)
        .def_readwrite("size", &PluginField::length);

    // PluginFieldCollection behaves like an iterable, and can be constructed from iterables.
    py::class_<PluginFieldCollection>(m, "PluginFieldCollection_", PluginFieldCollectionDoc::descr)
        .def(py::init<>(lambdas::plugin_field_collection_constructor), py::keep_alive<1, 2>{})
        .def("__len__", [](PluginFieldCollection& self) { return self.nbFields; })
        .def("__getitem__", [](PluginFieldCollection& self, int32_t const index) {
            PY_ASSERT_INDEX_ERROR(index < self.nbFields);
            return self.fields[index];
        });

    // Creating a trt.PluginFieldCollection in Python will actually construct a vector,
    // which can then be converted to an actual C++ PluginFieldCollection.
    py::implicitly_convertible<std::vector<nvinfer1::PluginField>, PluginFieldCollection>();

    py::class_<IPluginCreator>(m, "IPluginCreator", IPluginCreatorDoc::descr)
        .def_property_readonly("tensorrt_version", &IPluginCreator::getTensorRTVersion)
        .def_property_readonly("name", &IPluginCreator::getPluginName)
        .def_property_readonly("plugin_version", &IPluginCreator::getPluginVersion)
        .def_property_readonly("field_names", lambdas::get_field_names, py::return_value_policy::reference_internal)
        .def_property("plugin_namespace", &IPluginCreator::getPluginNamespace,
            py::cpp_function(&IPluginCreator::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("create_plugin", lambdas::creator_create_plugin, "name"_a, "field_collection"_a,
            IPluginCreatorDoc::create_plugin)
        .def("deserialize_plugin", lambdas::deserialize_plugin, "name"_a, "serialized_plugin"_a,
            IPluginCreatorDoc::deserialize_plugin);

    py::class_<IPluginRegistry, std::unique_ptr<IPluginRegistry, py::nodelete>>(
        m, "IPluginRegistry", IPluginRegistryDoc::descr)
        .def_property_readonly("plugin_creator_list", lambdas::get_plugin_creator_list)
        .def("register_creator", &IPluginRegistry::registerCreator, "creator"_a, "plugin_namespace"_a = "",
            py::keep_alive<1, 2>{}, IPluginRegistryDoc::register_creator)
        .def("deregister_creator", &IPluginRegistry::deregisterCreator, "creator"_a,
            IPluginRegistryDoc::deregister_creator)
        .def("get_plugin_creator", &IPluginRegistry::getPluginCreator, "type"_a, "version"_a, "plugin_namespace"_a = "",
            py::return_value_policy::reference_internal, IPluginRegistryDoc::get_plugin_creator)
        .def_property("error_recorder", &IPluginRegistry::getErrorRecorder,
            py::cpp_function(&IPluginRegistry::setErrorRecorder, py::keep_alive<1, 2>{}));

    m.def("get_plugin_registry", &getPluginRegistry, py::return_value_policy::reference,
        FreeFunctionsDoc::get_plugin_registry);
    m.def("get_builder_plugin_registry", &getBuilderPluginRegistry, py::return_value_policy::reference,
        FreeFunctionsDoc::get_builder_plugin_registry);
    m.def("init_libnvinfer_plugins", &initLibNvInferPlugins, "logger"_a, "namespace"_a,
        FreeFunctionsDoc::init_libnvinfer_plugins);

} // Plugin
} // namespace tensorrt
