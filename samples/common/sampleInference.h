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

#ifndef TRT_SAMPLE_INFERENCE_H
#define TRT_SAMPLE_INFERENCE_H

#include "sampleEngines.h"
#include "sampleReporting.h"
#include "sampleUtils.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferSafeRuntime.h"

namespace sample
{

struct InferenceEnvironment
{
    InferenceEnvironment() = delete;
    InferenceEnvironment(InferenceEnvironment const& other) = delete;
    InferenceEnvironment(InferenceEnvironment&& other) = delete;
    InferenceEnvironment(BuildEnvironment& bEnv) : engine(std::move(bEnv.engine)), safe(bEnv.engine.isSafe())
    {
    }

    LazilyDeserializedEngine engine;
    std::unique_ptr<Profiler> profiler;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;
    std::vector<std::unique_ptr<Bindings>> bindings;
    bool error{false};

    bool safe{false};
    std::vector<std::unique_ptr<nvinfer1::safe::IExecutionContext>> safeContexts;

    template <class ContextType>
    inline ContextType* getContext(int32_t streamIdx);
};

template <>
inline nvinfer1::IExecutionContext* InferenceEnvironment::getContext(int32_t streamIdx)
{
    return contexts[streamIdx].get();
}

template <>
inline nvinfer1::safe::IExecutionContext* InferenceEnvironment::getContext(int32_t streamIdx)
{
    return safeContexts[streamIdx].get();
}

//!
//! \brief Set up contexts and bindings for inference
//!
bool setUpInference(InferenceEnvironment& iEnv, InferenceOptions const& inference);

//!
//! \brief Deserialize the engine and time how long it takes.
//!
bool timeDeserialize(InferenceEnvironment& iEnv);

//!
//! \brief Run inference and collect timing, return false if any error hit during inference
//!
bool runInference(
    InferenceOptions const& inference, InferenceEnvironment& iEnv, int32_t device, std::vector<InferenceTrace>& trace);

//!
//! \brief Get layer information of the engine.
//!
std::string getLayerInformation(
    nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, nvinfer1::LayerInformationFormat format);

struct Binding
{
    bool isInput{false};
    std::unique_ptr<IMirroredBuffer> buffer;
    int64_t volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    void fill(std::string const& fileName);

    void fill();

    void dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
        std::string const separator = " ") const;
};

class Bindings
{
public:
    Bindings() = delete;
    explicit Bindings(bool useManaged)
        : mUseManaged(useManaged)
    {
    }

    void addBinding(int b, std::string const& name, bool isInput, int64_t volume, nvinfer1::DataType dataType,
        std::string const& fileName = "");

    void** getDeviceBuffers();

    void transferInputToDevice(TrtCudaStream& stream);

    void transferOutputToHost(TrtCudaStream& stream);

    void fill(int binding, std::string const& fileName)
    {
        mBindings[binding].fill(fileName);
    }

    void fill(int binding)
    {
        mBindings[binding].fill();
    }

    void dumpBindingDimensions(int binding, nvinfer1::IExecutionContext const& context, std::ostream& os) const;

    void dumpBindingValues(nvinfer1::IExecutionContext const& context, int binding, std::ostream& os,
        std::string const& separator = " ", int32_t batch = 1) const;

    void dumpInputs(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto isInput = [](Binding const& b) { return b.isInput; };
        dumpBindings(context, isInput, os);
    }

    void dumpOutputs(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto isOutput = [](Binding const& b) { return !b.isInput; };
        dumpBindings(context, isOutput, os);
    }

    void dumpBindings(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto all = [](Binding const& b) { return true; };
        dumpBindings(context, all, os);
    }

    void dumpBindings(nvinfer1::IExecutionContext const& context, std::function<bool(Binding const&)> predicate,
        std::ostream& os) const;

    std::unordered_map<std::string, int> getInputBindings() const
    {
        auto isInput = [](Binding const& b) { return b.isInput; };
        return getBindings(isInput);
    }

    std::unordered_map<std::string, int> getOutputBindings() const
    {
        auto isOutput = [](Binding const& b) { return !b.isInput; };
        return getBindings(isOutput);
    }

    std::unordered_map<std::string, int> getBindings() const
    {
        auto all = [](Binding const& b) { return true; };
        return getBindings(all);
    }

    std::unordered_map<std::string, int> getBindings(std::function<bool(Binding const&)> predicate) const;

private:
    std::unordered_map<std::string, int32_t> mNames;
    std::vector<Binding> mBindings;
    std::vector<void*> mDevicePointers;
    bool mUseManaged{false};
};

} // namespace sample

#endif // TRT_SAMPLE_INFERENCE_H
