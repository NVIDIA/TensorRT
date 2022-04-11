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

// Implementation of PyBind11 Binding Code for UffParser
#include "ForwardDeclarations.h"
#include "parsers/pyUffDoc.h"
#include "utils.h"

namespace tensorrt
{
using namespace nvuffparser;

namespace lambdas
{
static const auto uff_parse_buffer = [](IUffParser& self, py::buffer& buffer, nvinfer1::INetworkDefinition& network,
                                         nvinfer1::DataType weightsType = nvinfer1::DataType::kFLOAT) {
    py::buffer_info info = buffer.request();
    return self.parseBuffer(static_cast<const char*>(info.ptr), info.size * info.itemsize, network, weightsType);
};
} // namespace lambdas

void bindUff(py::module& m)
{
    py::enum_<UffInputOrder>(m, "UffInputOrder", UffInputOrderDoc::descr)
        .value("NCHW", UffInputOrder::kNCHW)
        .value("NHWC", UffInputOrder::kNHWC)
        .value("NC", UffInputOrder::kNC);

    py::enum_<FieldType>(m, "FieldType", FieldTypeDoc::descr)
        .value("FLOAT", FieldType::kFLOAT)
        .value("INT32", FieldType::kINT32)
        .value("CHAR", FieldType::kCHAR)
        .value("DIMS", FieldType::kDIMS)
        .value("DATATYPE", FieldType::kDATATYPE)
        .value("UNKNOWN", FieldType::kUNKNOWN);

    py::class_<FieldMap>(m, "FieldMap", FieldMapDoc::descr)
        .def(py::init<const char*, const void*, const FieldType, int>(), "name"_a, "data"_a, "type"_a, "length"_a = 1)
        .def_readwrite("name", &FieldMap::name)
        .def_readwrite("data", &FieldMap::data)
        .def_readwrite("type", &FieldMap::type)
        .def_readwrite("length", &FieldMap::length);

    py::class_<FieldCollection>(m, "FieldCollection", FieldCollectionDoc::descr)
        .def_readwrite("num_fields", &FieldCollection::nbFields)
        .def_readwrite("fields", &FieldCollection::fields);

    py::class_<IUffParser, std::unique_ptr<IUffParser, py::nodelete>>(m, "UffParser", UffParserDoc::descr)
        .def(py::init(&createUffParser))
        .def_property_readonly("uff_required_version_major", &IUffParser::getUffRequiredVersionMajor)
        .def_property_readonly("uff_required_version_minor", &IUffParser::getUffRequiredVersionMinor)
        .def_property_readonly("uff_required_version_patch", &IUffParser::getUffRequiredVersionPatch)
        .def_property(
            "plugin_namespace", nullptr, py::cpp_function(&IUffParser::setPluginNamespace, py::keep_alive<1, 2>{}))
        .def("register_input", &IUffParser::registerInput, "name"_a, "shape"_a, "order"_a = UffInputOrder::kNCHW,
            UffParserDoc::register_input)
        .def("register_output", &IUffParser::registerOutput, "name"_a, UffParserDoc::register_output)
        .def("parse", &IUffParser::parse, "file"_a, "network"_a, "weights_type"_a = nvinfer1::DataType::kFLOAT,
            UffParserDoc::parse, py::keep_alive<3, 1>{})
        .def("parse_buffer", lambdas::uff_parse_buffer, "buffer"_a, "network"_a,
            "weights_type"_a = nvinfer1::DataType::kFLOAT, UffParserDoc::parse_buffer, py::keep_alive<3, 1>{})
        .def_property("error_recorder", &IUffParser::getErrorRecorder,
            py::cpp_function(&IUffParser::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def("__del__", &utils::doNothingDel<IUffParser>);
}
} // namespace tensorrt
