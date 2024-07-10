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

// Implementation of PyBind11 Binding Code for OnnxParser
#include "ForwardDeclarations.h"
#include "onnx/NvOnnxParser.h"
#include "onnx/errorHelpers.hpp"
#include "parsers/pyOnnxDoc.h"
#include "utils.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <vector>

using namespace nvonnxparser;

namespace tensorrt
{
// Long lambda functions should go here rather than being inlined into the bindings (1 liners are OK).
namespace lambdas
{
static const auto parse = [](IParser& self, py::buffer const& model, char const* path = nullptr) {
    py::buffer_info info = model.request();
    return self.parse(info.ptr, info.size * info.itemsize, path);
};

static const auto parse_with_weight_descriptors = [](IParser& self, py::buffer const& model) {
    py::buffer_info info = model.request();
    return self.parseWithWeightDescriptors(info.ptr, info.size * info.itemsize);
};

static const auto parseFromFile
    = [](IParser& self, std::string const& model) { return self.parseFromFile(model.c_str(), 0); };

static const auto supportsModel = [](IParser& self, py::buffer const& model, char const* path = nullptr) {
    py::buffer_info info = model.request();
    SubGraphCollection_t subgraphs;
    bool const supported = self.supportsModel(info.ptr, info.size * info.itemsize, subgraphs, path);
    return std::make_pair(supported, subgraphs);
};

static const auto supportsModelV2 = [](IParser& self, py::buffer const& model, char const* path = nullptr) {
    py::buffer_info info = model.request();
    return self.supportsModelV2(info.ptr, info.size * info.itemsize, path);
};

static const auto isSubgraphSupported
    = [](IParser& self, int64_t const index) { return self.isSubgraphSupported(index); };

static const auto getSubgraphNodes = [](IParser& self, int64_t const index) {
    py::list py_nodes;
    int64_t nb_nodes = 0;
    int64_t* nodes = self.getSubgraphNodes(index, nb_nodes);
    for (int64_t i = 0; i < nb_nodes; i++)
    {
        py_nodes.append(nodes[i]);
    }
    return py_nodes;
};

static const auto get_used_vc_plugin_libraries = [](IParser& self) {
    std::vector<std::string> vcPluginLibs;
    int64_t nbPluginLibs;
    auto libCArray = self.getUsedVCPluginLibraries(nbPluginLibs);
    if (nbPluginLibs < 0)
    {
        utils::throwPyError(PyExc_RuntimeError, "Internal error");
    }
    vcPluginLibs.reserve(nbPluginLibs);
    for (int64_t i = 0; i < nbPluginLibs; ++i)
    {
        vcPluginLibs.emplace_back(std::string{libCArray[i]});
    }
    return vcPluginLibs;
};

static const auto get_local_function_stack = [](IParserError& self) {
    std::vector<std::string> localFunctionStack;
    int32_t localFunctionStackSize = self.localFunctionStackSize();
    if (localFunctionStackSize > 0)
    {
        auto localFunctionStackCArray = self.localFunctionStack();
        localFunctionStack.reserve(localFunctionStackSize);
        for (int32_t i = 0; i < localFunctionStackSize; ++i)
        {
            localFunctionStack.emplace_back(std::string{localFunctionStackCArray[i]});
        }
    }
    return localFunctionStack;
};

static const auto refitFromBytes = [](IParserRefitter& self, py::buffer const& model, char const* path = nullptr) {
    py::buffer_info info = model.request();
    return self.refitFromBytes(info.ptr, info.size * info.itemsize, path);
};

static const auto refitFromFile
    = [](IParserRefitter& self, std::string const& model) { return self.refitFromFile(model.c_str()); };

} // namespace lambdas

void bindOnnx(py::module& m)
{
    py::bind_vector<std::vector<size_t>>(m, "NodeIndices");
    py::bind_vector<SubGraphCollection_t>(m, "SubGraphCollection");

    py::class_<IParser>(m, "OnnxParser", OnnxParserDoc::descr, py::module_local())
        .def(py::init(&nvonnxparser::createParser), "network"_a, "logger"_a, OnnxParserDoc::init,
            py::keep_alive<1, 3>{}, py::keep_alive<2, 1>{})
        .def("parse", lambdas::parse, "model"_a, "path"_a = nullptr, OnnxParserDoc::parse,
            py::call_guard<py::gil_scoped_release>{})
        .def("parse_with_weight_descriptors", lambdas::parse_with_weight_descriptors, "model"_a,
            OnnxParserDoc::parse_with_weight_descriptors, py::call_guard<py::gil_scoped_release>{})
        .def("parse_from_file", lambdas::parseFromFile, "model"_a, OnnxParserDoc::parse_from_file,
            py::call_guard<py::gil_scoped_release>{})
        .def("supports_operator", &IParser::supportsOperator, "op_name"_a, OnnxParserDoc::supports_operator)
        .def("supports_model", lambdas::supportsModel, "model"_a, "path"_a = nullptr, OnnxParserDoc::supports_model)
        .def("supports_model_v2", lambdas::supportsModelV2, "model"_a, "path"_a = nullptr,
            OnnxParserDoc::supports_model_v2)
        .def_property_readonly("num_subgraphs", &IParser::getNbSubgraphs)
        .def("is_subgraph_supported", lambdas::isSubgraphSupported, "index"_a, OnnxParserDoc::is_subgraph_supported)
        .def("get_subgraph_nodes", lambdas::getSubgraphNodes, "index"_a, OnnxParserDoc::get_subgraph_nodes)
        .def_property_readonly("num_errors", &IParser::getNbErrors)
        .def("get_error", &IParser::getError, "index"_a, OnnxParserDoc::get_error)
        .def("clear_errors", &IParser::clearErrors, OnnxParserDoc::clear_errors)
        .def_property("flags", &IParser::getFlags, &IParser::setFlags)
        .def("clear_flag", &IParser::clearFlag, "flag"_a, OnnxParserDoc::clear_flag)
        .def("set_flag", &IParser::setFlag, "flag"_a, OnnxParserDoc::set_flag)
        .def("get_flag", &IParser::getFlag, "flag"_a, OnnxParserDoc::get_flag)
        .def("get_layer_output_tensor", &IParser::getLayerOutputTensor, "name"_a, "i"_a,
            OnnxParserDoc::get_layer_output_tensor)
        .def("get_used_vc_plugin_libraries", lambdas::get_used_vc_plugin_libraries,
            OnnxParserDoc::get_used_vc_plugin_libraries)
        .def("__del__", &utils::doNothingDel<IParser>);

    py::enum_<OnnxParserFlag>(m, "OnnxParserFlag", OnnxParserFlagDoc::descr, py::module_local())
        .value("NATIVE_INSTANCENORM", OnnxParserFlag::kNATIVE_INSTANCENORM, OnnxParserFlagDoc::NATIVE_INSTANCENORM);

    py::enum_<ErrorCode>(m, "ErrorCode", ErrorCodeDoc::descr, py::module_local())
        .value("SUCCESS", ErrorCode::kSUCCESS)
        .value("INTERNAL_ERROR", ErrorCode::kINTERNAL_ERROR)
        .value("MEM_ALLOC_FAILED", ErrorCode::kMEM_ALLOC_FAILED)
        .value("MODEL_DESERIALIZE_FAILED", ErrorCode::kMODEL_DESERIALIZE_FAILED)
        .value("INVALID_VALUE", ErrorCode::kINVALID_VALUE)
        .value("INVALID_GRAPH", ErrorCode::kINVALID_GRAPH)
        .value("INVALID_NODE", ErrorCode::kINVALID_NODE)
        .value("UNSUPPORTED_GRAPH", ErrorCode::kUNSUPPORTED_GRAPH)
        .value("UNSUPPORTED_NODE", ErrorCode::kUNSUPPORTED_NODE)
        .value("UNSUPPORTED_NODE_ATTR", ErrorCode::kUNSUPPORTED_NODE_ATTR)
        .value("UNSUPPORTED_NODE_INPUT", ErrorCode::kUNSUPPORTED_NODE_INPUT)
        .value("UNSUPPORTED_NODE_DATATYPE", ErrorCode::kUNSUPPORTED_NODE_DATATYPE)
        .value("UNSUPPORTED_NODE_DYNAMIC", ErrorCode::kUNSUPPORTED_NODE_DYNAMIC)
        .value("UNSUPPORTED_NODE_SHAPE", ErrorCode::kUNSUPPORTED_NODE_SHAPE)
        .value("REFIT_FAILED", ErrorCode::kREFIT_FAILED)
        .def("__str__", &onnx2trt::errorCodeStr)
        .def("__repr__", &onnx2trt::errorCodeStr);

    py::class_<IParserError, std::unique_ptr<IParserError, py::nodelete>>(m, "ParserError", py::module_local())
        .def("code", &IParserError::code, ParserErrorDoc::code)
        .def("desc", &IParserError::desc, ParserErrorDoc::desc)
        .def("file", &IParserError::file, ParserErrorDoc::file)
        .def("line", &IParserError::line, ParserErrorDoc::line)
        .def("func", &IParserError::func, ParserErrorDoc::func)
        .def("node", &IParserError::node, ParserErrorDoc::node)
        .def("node_name", &IParserError::nodeName, ParserErrorDoc::node_name)
        .def("node_operator", &IParserError::nodeOperator, ParserErrorDoc::node_operator)
        .def("local_function_stack", lambdas::get_local_function_stack, ParserErrorDoc::local_function_stack)
        .def("local_function_stack_size", &IParserError::localFunctionStackSize,
            ParserErrorDoc::local_function_stack_size)
        .def("__str__", &onnx2trt::parserErrorStr)
        .def("__repr__", &onnx2trt::parserErrorStr);

    py::class_<IParserRefitter>(m, "OnnxParserRefitter", OnnxParserRefitterDoc::descr, py::module_local())
        .def(py::init(&nvonnxparser::createParserRefitter), "refitter"_a, "logger"_a, OnnxParserRefitterDoc::init,
            py::keep_alive<1, 3>{}, py::keep_alive<2, 1>{})
        .def("refit_from_bytes", lambdas::refitFromBytes, "model"_a, "path"_a = nullptr,
            OnnxParserRefitterDoc::refit_from_bytes, py::call_guard<py::gil_scoped_release>{})
        .def("refit_from_file", lambdas::refitFromFile, "model"_a, OnnxParserRefitterDoc::refit_from_file,
            py::call_guard<py::gil_scoped_release>{})
        .def_property_readonly("num_errors", &IParserRefitter::getNbErrors)
        .def("get_error", &IParserRefitter::getError, "index"_a, OnnxParserRefitterDoc::get_error)
        .def("clear_errors", &IParserRefitter::clearErrors, OnnxParserRefitterDoc::clear_errors);

    // Free functions.
    m.def("get_nv_onnx_parser_version", &getNvOnnxParserVersion, get_nv_onnx_parser_version);
}
} // namespace tensorrt
