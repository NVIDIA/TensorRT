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

#include "bert_infer.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct BertInferenceRunner
{
    BertInferenceRunner(
        const std::string& enginePath, const int maxBatchSize, const int maxSeqLength, const bool enableGraph)
        : bert{enginePath, maxBatchSize, maxSeqLength, enableGraph}
    {
    }

    void prepare(const int batchSize)
    {
        bert.prepare(0, batchSize);
    }

    py::array_t<float> run(py::array_t<int> inputIds, py::array_t<int> segmentIds, py::array_t<int> inputMask)
    {

        const void* inputIdsPtr = inputIds.request().ptr;
        const void* segmentIdsPtr = segmentIds.request().ptr;
        const void* inputMaskPtr = inputMask.request().ptr;

        bert.run(inputIdsPtr, segmentIdsPtr, inputMaskPtr, 0, 1);

        auto output = py::array_t<float>(bert.mOutputDims, (float*) bert.mHostOutput.data());

        return output;
    }

    BertInference bert;
};

PYBIND11_MODULE(infer_c, m)
{
    m.doc() = "Pybind11 plugin for Bert inference";

    py::class_<BertInferenceRunner>(m, "bert_inf")
        .def(py::init<const std::string&, const int, const int, const bool>())
        .def("prepare", &BertInferenceRunner::prepare)
        .def("run", &BertInferenceRunner::run);
}
