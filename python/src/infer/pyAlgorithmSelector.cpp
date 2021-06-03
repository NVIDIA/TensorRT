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

// This contains the fundamental types, i.e. Dims, Weights, dtype
#include "NvInfer.h"
#include "utils.h"
#include "infer/pyAlgorithmSelectorDoc.h"
#include "ForwardDeclarations.h"
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>



namespace tensorrt
{
    using namespace nvinfer1;

    namespace lambda
    {
        // For IAlgorithmContext
        static const auto get_shape = [] (IAlgorithmContext& self, int32_t index) -> std::vector<Dims>
        {
            std::vector<Dims> shapes{};
            Dims minShape = self.getDimensions(index, OptProfileSelector::kMIN);
            if (minShape.nbDims != -1)
            {
                shapes.emplace_back(minShape);
                shapes.emplace_back(self.getDimensions(index, OptProfileSelector::kOPT));
                shapes.emplace_back(self.getDimensions(index, OptProfileSelector::kMAX));
            }
            return shapes;
        };

        // For IAlgorithm
        static const auto get_algorithm_io_info = [] (IAlgorithm& self, int32_t index) -> const IAlgorithmIOInfo&
        {
            return self.getAlgorithmIOInfo(index);
        };
    } //lambda

    class IAlgorithmSelectorTrampoline : public IAlgorithmSelector
    {
        public:
            using IAlgorithmSelector::IAlgorithmSelector;

            virtual int32_t selectAlgorithms(const IAlgorithmContext& context, const IAlgorithm* const* choices, int32_t nbChoices, int32_t* selection) override
            {
                py::gil_scoped_acquire gil{};
                py::function pySelectAlgorithms = utils::getOverload(this, "select_algorithms");
                std::vector<const IAlgorithm*> choices_vector;
                std::copy(choices, choices + nbChoices, std::back_inserter(choices_vector));

                py::object result_uncast = pySelectAlgorithms(&context, choices_vector);

                std::pair<int32_t, std::vector<int32_t>> result = result_uncast.cast<std::pair<int32_t, std::vector<int32_t>>>();

                int32_t ret_value = std::get<0>(result);
                int32_t* selection_ptr = std::get<1>(result).data();
                std::copy(selection_ptr, selection_ptr + std::get<1>(result).size(), selection);
                return ret_value;
            }

            virtual void reportAlgorithms(const IAlgorithmContext* const* algoContexts, const IAlgorithm* const* algoChoices, int32_t size) override
            {
                py::gil_scoped_acquire gil{};

                std::vector<const IAlgorithmContext*> contexts;
                std::copy(algoContexts, algoContexts + size, std::back_inserter(contexts));
                std::vector<const IAlgorithm*> choices;
                std::copy(algoChoices, algoChoices + size, std::back_inserter(choices));
                py::function pyReportAlgorithms = utils::getOverload(this, "report_algorithms");
                pyReportAlgorithms(contexts, choices);

            }
    }; // IAlgorithmSelectorTrampoline

    void bindAlgorithm(py::module& m)
    {
        // IAlgorithmIOInfo
        py::class_<IAlgorithmIOInfo, std::unique_ptr<IAlgorithmIOInfo, py::nodelete>>(m, "IAlgorithmIOInfo", IAlgorithmIOInfoDOC::descr)
            .def_property_readonly("tensor_format", &IAlgorithmIOInfo::getTensorFormat)
            .def_property_readonly("dtype", &IAlgorithmIOInfo::getDataType)
            .def_property_readonly("strides", &IAlgorithmIOInfo::getStrides)
        ;

        // IAlgorithmVariant
        py::class_<IAlgorithmVariant, std::unique_ptr<IAlgorithmVariant, py::nodelete>>(m, "IAlgorithmVariant", IAlgorithmVariantDOC::descr)
            .def_property_readonly("implementation", &IAlgorithmVariant::getImplementation)
            .def_property_readonly("tactic", &IAlgorithmVariant::getTactic)
        ;

        // IAlgorithmContext
        py::class_<IAlgorithmContext, std::unique_ptr<IAlgorithmContext,  py::nodelete>>(m, "IAlgorithmContext",  IAlgorithmContextDoc::descr)
            .def_property_readonly("name", &IAlgorithmContext::getName)
            .def("get_shape", lambda::get_shape, "index"_a, IAlgorithmContextDoc::get_shape)
            .def_property_readonly("num_inputs", &IAlgorithmContext::getNbInputs)
            .def_property_readonly("num_outputs", &IAlgorithmContext::getNbOutputs)
        ;

        // IAlgorithm
        py::class_<IAlgorithm, std::unique_ptr<IAlgorithm,  py::nodelete>>(m, "IAlgorithm",  IAlgorithmDoc::descr)
            .def("get_algorithm_io_info", lambda::get_algorithm_io_info, "index"_a, IAlgorithmDoc::get_algorithm_io_info)
            .def_property_readonly("algorithm_variant", &IAlgorithm::getAlgorithmVariant)
            .def_property_readonly("timing_msec", &IAlgorithm::getTimingMSec)
            .def_property_readonly("workspace_size", &IAlgorithm::getWorkspaceSize)
        ;

        // IAlgorithmSelector
        py::class_<IAlgorithmSelector, IAlgorithmSelectorTrampoline, std::unique_ptr<IAlgorithmSelector,  py::nodelete>>(m, "IAlgorithmSelector", IAlgorithmSelectorDoc::descr)
            .def(py::init_alias<>())
        ;
    }// bindAlgorithm
} /* tensorrt */
