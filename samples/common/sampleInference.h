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

#ifndef TRT_SAMPLE_INFERENCE_H
#define TRT_SAMPLE_INFERENCE_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"

#include "sampleReporting.h"
#include "sampleUtils.h"

namespace sample
{

struct InferenceEnvironment
{
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<Profiler> profiler;
    std::vector<TrtUniquePtr<nvinfer1::IExecutionContext>> context;
    std::vector<std::unique_ptr<Bindings>> bindings;
    bool error{false};
};

//!
//! \brief Set up contexts and bindings for inference
//!
bool setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference);

//!
//! \brief Deserialize the engine and time how long it takes.
//!
bool timeDeserialize(InferenceEnvironment& iEnv);

//!
//! \brief Run inference and collect timing, return false if any error hit during inference
//!
bool runInference(const InferenceOptions& inference, InferenceEnvironment& iEnv, int device, std::vector<InferenceTrace>& trace);

} // namespace sample

#endif // TRT_SAMPLE_INFERENCE_H
