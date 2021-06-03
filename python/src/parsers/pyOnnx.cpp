/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

// Implementation of PyBind11 Binding Code for OnnxParser
#include "NvOnnxParser.h"
#include "ForwardDeclarations.h"
#include "parsers/pyOnnxDoc.h"
#include <pybind11/stl_bind.h>

using namespace nvonnxparser;

namespace tensorrt
{
namespace lambdas
{
    static const auto error_code_str = [] (ErrorCode self) {
        switch (self) {
            case ErrorCode::kSUCCESS:
                return "SUCCESS";
            case ErrorCode::kINTERNAL_ERROR:
                return "INTERNAL_ERROR";
            case ErrorCode::kMEM_ALLOC_FAILED:
                return "MEM_ALLOC_FAILED";
            case ErrorCode::kMODEL_DESERIALIZE_FAILED:
                return "MODEL_DESERIALIZE_FAILED";
            case ErrorCode::kINVALID_VALUE:
                return "INVALID_VALUE";
            case ErrorCode::kINVALID_GRAPH:
                return "INVALID_GRAPH";
            case ErrorCode::kINVALID_NODE:
                return "INVALID_NODE";
            case ErrorCode::kUNSUPPORTED_GRAPH:
                return "UNSUPPORTED_GRAPH";
            case ErrorCode::kUNSUPPORTED_NODE:
                return "UNSUPPORTED_NODE";
        }
        return "UNKNOWN";
    };

    static const auto parser_error_str = [](IParserError& self) {
        return "In node " + std::to_string(self.node()) + " ("  + self.func() + "): " + error_code_str(self.code()) + ": " + self.desc();
    };

    // For ONNX Parser
    static const auto parse = [](IParser& self, const py::buffer& model, const char* path = nullptr) {
        py::buffer_info info = model.request();
        return self.parse(info.ptr, info.size * info.itemsize, path);
    };

    static const auto parseFromFile
        = [](IParser& self, const std::string& model) { return self.parseFromFile(model.c_str(), 0);
    };

    static const auto getRefitMap = [] (IParser& self)
    {
        int size = self.getRefitMap(nullptr, nullptr, nullptr);
        std::vector<const char*> weightNames(size);
        std::vector<const char*> layerNames(size);
        std::vector<nvinfer1::WeightsRole> roles(size);
        self.getRefitMap(weightNames.data(), layerNames.data(), roles.data());
        return std::tuple<std::vector<const char*>, std::vector<const char*>, std::vector<nvinfer1::WeightsRole>>{weightNames, layerNames, roles};
    };

    static const auto supportsModel = [](IParser& self, const py::buffer& model, const char* path = nullptr) {
        py::buffer_info info = model.request();
        SubGraphCollection_t subgraphs;
        const bool supported = self.supportsModel(info.ptr, info.size * info.itemsize, subgraphs, path);
        return std::make_pair(supported, subgraphs);
    };
} // namespace lambdas

    void bindOnnx(py::module& m)
    {
        py::bind_vector<std::vector<size_t>>(m, "NodeIndices");
        py::bind_vector<SubGraphCollection_t>(m, "SubGraphCollection");

        py::class_<IParser, std::unique_ptr<IParser, py::nodelete>>(m, "OnnxParser", OnnxParserDoc::descr)
            .def(py::init(&nvonnxparser::createParser), "network"_a, "logger"_a, OnnxParserDoc::init)
            .def("parse", lambdas::parse, "model"_a, "path"_a = nullptr, OnnxParserDoc::parse,
                py::call_guard<py::gil_scoped_release>{})
            .def("parse_from_file", lambdas::parseFromFile, "model"_a, OnnxParserDoc::parseFromFile,
                py::call_guard<py::gil_scoped_release>{})
            .def("supports_operator", &IParser::supportsOperator, "op_name"_a, OnnxParserDoc::supports_operator)
            .def("supports_model", lambdas::supportsModel, "model"_a, "path"_a = nullptr,
                OnnxParserDoc::supports_model)
            .def_property_readonly("num_errors", &IParser::getNbErrors)
            .def("get_error", &IParser::getError, "index"_a, OnnxParserDoc::get_error)
            .def("clear_errors", &IParser::clearErrors, OnnxParserDoc::clear_errors)
            .def("get_refit_map", lambdas::getRefitMap, OnnxParserDoc::get_refit_map)
            .def("__del__", &IParser::destroy);

        py::enum_<ErrorCode>(m, "ErrorCode", ErrorCodeDoc::descr)
            .value("SUCCESS", ErrorCode::kSUCCESS)
            .value("INTERNAL_ERROR", ErrorCode::kINTERNAL_ERROR)
            .value("MEM_ALLOC_FAILED", ErrorCode::kMEM_ALLOC_FAILED)
            .value("MODEL_DESERIALIZE_FAILED", ErrorCode::kMODEL_DESERIALIZE_FAILED)
            .value("INVALID_VALUE", ErrorCode::kINVALID_VALUE)
            .value("INVALID_GRAPH", ErrorCode::kINVALID_GRAPH)
            .value("INVALID_NODE", ErrorCode::kINVALID_NODE)
            .value("UNSUPPORTED_GRAPH", ErrorCode::kUNSUPPORTED_GRAPH)
            .value("UNSUPPORTED_NODE", ErrorCode::kUNSUPPORTED_NODE)
            .def("__str__", lambdas::error_code_str)
            .def("__repr__", lambdas::error_code_str);

        py::class_<IParserError, std::unique_ptr<IParserError, py::nodelete>>(m, "ParserError")
            .def("code", &IParserError::code, ParserErrorDoc::code)
            .def("desc", &IParserError::desc, ParserErrorDoc::desc)
            .def("file", &IParserError::file, ParserErrorDoc::file)
            .def("line", &IParserError::line, ParserErrorDoc::line)
            .def("func", &IParserError::func, ParserErrorDoc::func)
            .def("node", &IParserError::node, ParserErrorDoc::node)
            .def("__str__", lambdas::parser_error_str)
            .def("__repr__", lambdas::parser_error_str);

        // Free functions.
        m.def("get_nv_onnx_parser_version", &getNvOnnxParserVersion, get_nv_onnx_parser_version);
    }
} // namespace tensorrt
