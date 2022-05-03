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

// This contains the fundamental types, i.e. Dims, Weights, dtype
#include "ForwardDeclarations.h"
#include "utils.h"
#include <pybind11/stl.h>

#include "infer/pyAlgorithmSelectorDoc.h"
#include <cuda_runtime_api.h>
#include <vector>

namespace tensorrt
{
using namespace nvinfer1;

namespace lambdas
{
// For IAlgorithmContext
static const auto get_shape = [](IAlgorithmContext& self, int32_t index) -> std::vector<Dims> {
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
} // namespace lambdas

class IAlgorithmSelectorTrampoline : public IAlgorithmSelector
{
public:
    using IAlgorithmSelector::IAlgorithmSelector;

    virtual int32_t selectAlgorithms(const IAlgorithmContext& context, const IAlgorithm* const* choices,
        int32_t nbChoices, int32_t* selection) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            std::vector<const IAlgorithm*> choicesVector;
            std::copy(choices, choices + nbChoices, std::back_inserter(choicesVector));

            py::function pySelectAlgorithms
                = utils::getOverride(static_cast<IAlgorithmSelector*>(this), "select_algorithms");
            if (!pySelectAlgorithms)
            {
                return -1;
            }

            py::object pyResult = pySelectAlgorithms(&context, choicesVector);

            std::vector<int32_t> result;
            try
            {
                result = pyResult.cast<decltype(result)>();
            }
            catch (const py::cast_error& e)
            {
                std::cerr << "[ERROR] Return value of select_algorithms() could not be interpreted as a List[int]"
                        << std::endl;
                return -1;
            }

            std::copy(result.data(), result.data() + result.size(), selection);
            return static_cast<int32_t>(result.size());
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in select_algorithms(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in select_algorithms()" << std::endl;
        }
        return -1;
    }

    virtual void reportAlgorithms(const IAlgorithmContext* const* algoContexts, const IAlgorithm* const* algoChoices,
        int32_t size) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            std::vector<const IAlgorithmContext*> contexts;
            std::copy(algoContexts, algoContexts + size, std::back_inserter(contexts));
            std::vector<const IAlgorithm*> choices;
            std::copy(algoChoices, algoChoices + size, std::back_inserter(choices));

            py::function pyReportAlgorithms
                = utils::getOverride(static_cast<IAlgorithmSelector*>(this), "report_algorithms");
            if (!pyReportAlgorithms)
            {
                return;
            }

            pyReportAlgorithms(contexts, choices);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in report_algorithms(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in report_algorithms()" << std::endl;
        }
    }
}; // IAlgorithmSelectorTrampoline

// NOTE: Fake bindings are provided for some of the application-implemented functions here.
// These are solely for documentation purposes. The user is meant to override these functions
// in their own code, and the bindings here will never be called.

std::vector<int32_t> select_algorithms(
    IAlgorithmSelector&, const IAlgorithmContext&, const std::vector<const IAlgorithm*>&)
{
    return {};
}

void report_algorithms(
    IAlgorithmSelector&, const std::vector<const IAlgorithmContext*>&, const std::vector<const IAlgorithm*>&)
{
}

void bindAlgorithm(py::module& m)
{
    // IAlgorithmIOInfo
    py::class_<IAlgorithmIOInfo, std::unique_ptr<IAlgorithmIOInfo, py::nodelete>>(
        m, "IAlgorithmIOInfo", IAlgorithmIOInfoDOC::descr)
        .def_property_readonly("tensor_format", &IAlgorithmIOInfo::getTensorFormat)
        .def_property_readonly("dtype", &IAlgorithmIOInfo::getDataType)
        .def_property_readonly("strides", &IAlgorithmIOInfo::getStrides);

    // IAlgorithmVariant
    py::class_<IAlgorithmVariant, std::unique_ptr<IAlgorithmVariant, py::nodelete>>(
        m, "IAlgorithmVariant", IAlgorithmVariantDOC::descr)
        .def_property_readonly("implementation", &IAlgorithmVariant::getImplementation)
        .def_property_readonly("tactic", &IAlgorithmVariant::getTactic);

    // IAlgorithmContext
    py::class_<IAlgorithmContext, std::unique_ptr<IAlgorithmContext, py::nodelete>>(
        m, "IAlgorithmContext", IAlgorithmContextDoc::descr)
        .def_property_readonly("name", &IAlgorithmContext::getName)
        .def("get_shape", lambdas::get_shape, "index"_a, IAlgorithmContextDoc::get_shape)
        .def_property_readonly("num_inputs", &IAlgorithmContext::getNbInputs)
        .def_property_readonly("num_outputs", &IAlgorithmContext::getNbOutputs);

    // IAlgorithm
    py::class_<IAlgorithm, std::unique_ptr<IAlgorithm, py::nodelete>>(m, "IAlgorithm", IAlgorithmDoc::descr)
        .def("get_algorithm_io_info", &IAlgorithm::getAlgorithmIOInfoByIndex, "index"_a,
            IAlgorithmDoc::get_algorithm_io_info, py::return_value_policy::reference_internal)
        .def_property_readonly("algorithm_variant", &IAlgorithm::getAlgorithmVariant)
        .def_property_readonly("timing_msec", &IAlgorithm::getTimingMSec)
        .def_property_readonly("workspace_size", &IAlgorithm::getWorkspaceSize);

    // IAlgorithmSelector
    py::class_<IAlgorithmSelector, IAlgorithmSelectorTrampoline>(m, "IAlgorithmSelector", IAlgorithmSelectorDoc::descr)
        .def(py::init_alias<>()) // Always initialize trampoline class.
        .def(
            "select_algorithms", &select_algorithms, "context"_a, "choices"_a, IAlgorithmSelectorDoc::select_algorithms)
        .def("report_algorithms", &report_algorithms, "contexts"_a, "choices"_a,
            IAlgorithmSelectorDoc::report_algorithms);
} // bindAlgorithm
} // namespace tensorrt
