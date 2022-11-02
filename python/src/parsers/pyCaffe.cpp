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

// Implementation of PyBind11 Binding Code for CaffeParser
#include "ForwardDeclarations.h"
#include "parsers/pyCaffeDoc.h"
#include "utils.h"
#include <pybind11/numpy.h>

namespace tensorrt
{
using namespace nvcaffeparser1;

namespace lambdas
{
static const auto parse_binary_proto = [](ICaffeParser& self, const std::string& filename) {
    using VoidFunc = void (*)(void*);

    // Type-erasure allows us to properly destroy the IBinaryProtoBlob in the bindings.
    nvcaffeparser1::IBinaryProtoBlob* proto = self.parseBinaryProto(filename.c_str());
    VoidFunc freeFunc = [](void* p) { static_cast<IBinaryProtoBlob*>(p)->destroy(); };
    py::capsule freeBlob{static_cast<void*>(proto), freeFunc};

    // Explicitly check for narrowing dimensions
    auto const volume = utils::volume(proto->getDimensions());
    PY_ASSERT_RUNTIME_ERROR(static_cast<decltype(volume)>(static_cast<int32_t>(volume)) == volume,
        "Volume exceeds the representable limit.");

    // By specifying the py::capsule as a parent here, we tie the lifetime of the data buffer to this array.
    // When this array is eventually destroyed on the Python side, the capsule parent will free(protoPtr).
    return py::array{utils::nptype(proto->getDataType()), static_cast<int32_t>(volume), proto->getData(), freeBlob};
};

static const auto parse_buffer = [](ICaffeParser& self, py::buffer& deploy, py::buffer& model,
                                     nvinfer1::INetworkDefinition& network, nvinfer1::DataType dtype) {
    py::buffer_info deploy_info = deploy.request();
    py::buffer_info model_info = model.request();
    return self.parseBuffers(static_cast<const uint8_t*>(deploy_info.ptr), deploy_info.size * deploy_info.itemsize,
        static_cast<const uint8_t*>(model_info.ptr), model_info.size * model_info.itemsize, network, dtype);
};

// For IPluginFactoryV2
static const auto PluginV2_create_plugin
    = [](IPluginFactoryV2& self, const std::string& layerName, const std::vector<nvinfer1::Weights>& weights) {
          return self.createPlugin(layerName.c_str(), weights.data(), weights.size());
      };
} // namespace lambdas

void bindCaffe(py::module& m)
{
    py::class_<IBlobNameToTensor, std::unique_ptr<IBlobNameToTensor, py::nodelete>>(
        m, "IBlobNameToTensor", IBlobNameToTensorDoc::descr)
        .def("find", &IBlobNameToTensor::find, "name"_a, IBlobNameToTensorDoc::find);

    py::class_<IPluginFactoryV2>(m, "ICaffePluginFactoryV2", ICaffePluginFactoryV2Doc::descr)
        .def("is_plugin_v2", &IPluginFactoryV2::isPluginV2, "layer_name"_a, ICaffePluginFactoryV2Doc::is_plugin_v2)
        .def("create_plugin", lambdas::PluginV2_create_plugin, "layer_name"_a, "weights"_a, py::keep_alive<1, 3>{},
            ICaffePluginFactoryV2Doc::create_plugin);

    py::class_<ICaffeParser, std::unique_ptr<ICaffeParser, py::nodelete>>(m, "CaffeParser", ICaffeParserDoc::descr)
        .def(py::init(&nvcaffeparser1::createCaffeParser))
        .def_property("protobuf_buffer_size", nullptr, &ICaffeParser::setProtobufBufferSize)
        .def_property(
            "plugin_factory_v2", nullptr, py::cpp_function(&ICaffeParser::setPluginFactoryV2, py::keep_alive<1, 2>{}))
        .def_property(
            "plugin_namespace", nullptr, py::cpp_function(&ICaffeParser::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("parse", &ICaffeParser::parse, "deploy"_a, "model"_a, "network"_a, "dtype"_a, ICaffeParserDoc::parse,
            py::keep_alive<4, 1>{})
        .def("parse_buffer", lambdas::parse_buffer, "deploy_buffer"_a, "model_buffer"_a, "network"_a, "dtype"_a,
            ICaffeParserDoc::parse_buffer, py::keep_alive<4, 1>{})
        .def("parse_binary_proto", lambdas::parse_binary_proto, "filename"_a, ICaffeParserDoc::parse_binary_proto)
        .def_property("error_recorder", &ICaffeParser::getErrorRecorder,
            py::cpp_function(&ICaffeParser::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def("__del__", &utils::doNothingDel<ICaffeParser>);

    m.def("shutdown_protobuf_library", &nvcaffeparser1::shutdownProtobufLibrary,
        FreeFunctionsDoc::shutdown_protobuf_library);
}
} // namespace tensorrt
