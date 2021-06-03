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

// Implementation of PyBind11 Binding Code for UffParser
#include "NvUffParser.h"
#include "NvInfer.h"
#include "parsers/pyUffDoc.h"
#include "ForwardDeclarations.h"

namespace tensorrt
{
    using namespace nvuffparser;

    namespace lambdas {
        static const auto create_plugin = [] (IPluginFactory& self, const std::string& layerName, const std::vector<nvinfer1::Weights>& weights, const FieldCollection& fc) {
            return self.createPlugin(layerName.c_str(), weights.data(), weights.size(), fc);
        };

        static const auto uff_parse_buffer = [] (IUffParser& self, py::buffer& buffer, nvinfer1::INetworkDefinition& network, nvinfer1::DataType weightsType = nvinfer1::DataType::kFLOAT) {
            py::buffer_info info = buffer.request();
            return self.parseBuffer(static_cast<const char*>(info.ptr), info.size * info.itemsize, network, weightsType);
        };
    } /* lambdas */

    void bindUff(py::module& m)
    {
		py::enum_<UffInputOrder>(m, "UffInputOrder", UffInputOrderDoc::descr)
		    .value("NCHW", UffInputOrder::kNCHW)
		    .value("NHWC", UffInputOrder::kNHWC)
		    .value("NC", UffInputOrder::kNC)
		;

		py::enum_<FieldType>(m, "FieldType", FieldTypeDoc::descr)
		    .value("FLOAT", FieldType::kFLOAT)
		    .value("INT32", FieldType::kINT32)
		    .value("CHAR", FieldType::kCHAR)
		    .value("DIMS", FieldType::kDIMS)
		    .value("DATATYPE", FieldType::kDATATYPE)
		    .value("UNKNOWN", FieldType::kUNKNOWN )
		;

		py::class_<FieldMap>(m, "FieldMap", FieldMapDoc::descr)
    	   	.def(py::init<const char*, const void*, const FieldType, int>(), "name"_a, "data"_a, "type"_a, "length"_a = 1)
    		.def_readwrite("name", &FieldMap::name)
    	    .def_readwrite("data", &FieldMap::data)
    	    .def_readwrite("type", &FieldMap::type)
    	    .def_readwrite("length", &FieldMap::length)
	    ;

		py::class_<FieldCollection>(m, "FieldCollection", FieldCollectionDoc::descr)
            .def_readwrite("num_fields", &FieldCollection::nbFields)
            .def_readwrite("fields", &FieldCollection::fields)
	    ;

        class pyUffIPluginFactory : public IPluginFactory
        {
            using IPluginFactory::IPluginFactory;

            bool isPlugin(const char* layerName) override
            {
                PYBIND11_OVERLOAD_PURE_NAME(bool, IPluginFactory, "is_plugin", isPlugin, layerName);
            }

            nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const FieldCollection fc) override
            {
                PYBIND11_OVERLOAD_PURE_NAME(nvinfer1::IPlugin*, IPluginFactory, "create_plugin", createPlugin, layerName, weights, nbWeights, fc);
            }
        };

		py::class_<IPluginFactory, pyUffIPluginFactory>(m, "IUffPluginFactory", IUffPluginFactoryDoc::descr)
            .def(py::init<>())
	    	.def("is_plugin", &IPluginFactory::isPlugin, "layer_name"_a, IUffPluginFactoryDoc::is_plugin)
			.def("create_plugin", lambdas::create_plugin, "layer_name"_a, "weights"_a, "field_collection"_a, IUffPluginFactoryDoc::create_plugin)
	    ;

		py::class_<IPluginFactoryExt>(m, "IUffPluginFactoryExt", IUffPluginFactoryExtDoc::descr)
	    	.def("get_version", &IPluginFactoryExt::getVersion, IUffPluginFactoryExtDoc::get_version)
			.def("is_plugin_ext", &IPluginFactoryExt::isPluginExt, "layer_name"_a, IUffPluginFactoryExtDoc::is_plugin_ext)
	    ;

	    py::class_<IUffParser, std::unique_ptr<IUffParser, py::nodelete> >(m, "UffParser", UffParserDoc::descr)
	        .def(py::init(&createUffParser))
	        .def_property_readonly("uff_required_version_major", &IUffParser::getUffRequiredVersionMajor)
	        .def_property_readonly("uff_required_version_minor", &IUffParser::getUffRequiredVersionMinor)
	        .def_property_readonly("uff_required_version_patch", &IUffParser::getUffRequiredVersionPatch)
	        .def_property("plugin_factory", nullptr, py::cpp_function(&IUffParser::setPluginFactory, py::keep_alive<1, 2>{}))
	        .def_property("plugin_factory_ext", nullptr, py::cpp_function(&IUffParser::setPluginFactoryExt, py::keep_alive<1, 2>{}))
	        .def_property("plugin_namespace", nullptr, py::cpp_function(&IUffParser::setPluginNamespace, py::keep_alive<1, 2>{}))
	        .def("register_input", &IUffParser::registerInput, "name"_a, "shape"_a, "order"_a = UffInputOrder::kNCHW, UffParserDoc::register_input)
	        .def("register_output", &IUffParser::registerOutput, "name"_a, UffParserDoc::register_output)
	        .def("parse", &IUffParser::parse, "file"_a, "network"_a, "weights_type"_a = nvinfer1::DataType::kFLOAT, UffParserDoc::parse)
	        .def("parse_buffer", lambdas::uff_parse_buffer, "buffer"_a, "network"_a, "weights_type"_a = nvinfer1::DataType::kFLOAT, UffParserDoc::parse_buffer)
            .def("__del__", &IUffParser::destroy)
    	;
    }
} /* tensorrt */
