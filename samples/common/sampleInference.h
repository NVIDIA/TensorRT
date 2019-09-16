/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <iostream>
#include <vector>
#include <string>

#include "NvInfer.h"

#include "sampleUtils.h"
#include "sampleReporting.h"

namespace sample
{

struct InferenceEnvironment
{
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IProfiler> profiler;
    std::vector<TrtUniquePtr<nvinfer1::IExecutionContext>> context;
    std::vector<Bindings> bindings;
};

//!
//! \brief Set up contexts and bindings for inference
//!
void setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference);

//!
//! \brief Run inference and collect timing
//!
void runInference(const InferenceOptions& inference, InferenceEnvironment& iEnv, std::vector<InferenceTime>& trace);

} // namespace sample

#endif // TRT_SAMPLE_INFERENCE_H
