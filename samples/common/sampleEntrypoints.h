/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_SAMPLE_ENTRYPOINTS_H
#define TRT_SAMPLE_ENTRYPOINTS_H

//! \file sampleEntrypoints.h
//!
//! Declares and conditionally defines entrypoints needed to create base TensorRT objects, depending
//! on whether the given sample uses TRT at link time or dynamically.  Since common code is built once
//! and shared across all samples (both link-time and dynamic TRT), it does not define these entrypoints,
//! so each sample must define them individually.
//!
//! Samples that use TRT at link time can define DEFINE_TRT_ENTRYPOINTS before including this header to
//! pick up the definitions here.

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "logger.h"

extern nvinfer1::IBuilder* createBuilder();
extern nvinfer1::IRuntime* createRuntime();
extern nvinfer1::IRefitter* createRefitter(nvinfer1::ICudaEngine& engine);

extern nvonnxparser::IParser* createONNXParser(nvinfer1::INetworkDefinition& network);

extern nvcaffeparser1::ICaffeParser* sampleCreateCaffeParser();
extern void shutdownCaffeParser();

extern nvuffparser::IUffParser* sampleCreateUffParser();
extern void shutdownUffParser();

#if !defined(DEFINE_TRT_ENTRYPOINTS)
#define DEFINE_TRT_ENTRYPOINTS 0
#endif

// Allow opting out of individual entrypoints that are unused by the sample
#if !defined(DEFINE_TRT_BUILDER_ENTRYPOINT)
#define DEFINE_TRT_BUILDER_ENTRYPOINT 1
#endif
#if !defined(DEFINE_TRT_RUNTIME_ENTRYPOINT)
#define DEFINE_TRT_RUNTIME_ENTRYPOINT 1
#endif
#if !defined(DEFINE_TRT_REFITTER_ENTRYPOINT)
#define DEFINE_TRT_REFITTER_ENTRYPOINT 1
#endif
#if !defined(DEFINE_TRT_ONNX_PARSER_ENTRYPOINT)
#define DEFINE_TRT_ONNX_PARSER_ENTRYPOINT 1
#endif
#if !defined(DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT)
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 1
#endif

#if DEFINE_TRT_ENTRYPOINTS
nvinfer1::IBuilder* createBuilder()
{
#if DEFINE_TRT_BUILDER_ENTRYPOINT
    return nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
#else
    return {};
#endif
}

nvinfer1::IRuntime* createRuntime()
{
#if DEFINE_TRT_RUNTIME_ENTRYPOINT
    return nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
#else
    return {};
#endif
}

nvinfer1::IRefitter* createRefitter(nvinfer1::ICudaEngine& engine)
{
#if DEFINE_TRT_REFITTER_ENTRYPOINT
    return nvinfer1::createInferRefitter(engine, sample::gLogger.getTRTLogger());
#else
    return {};
#endif
}

nvonnxparser::IParser* createONNXParser(nvinfer1::INetworkDefinition& network)
{
#if DEFINE_TRT_ONNX_PARSER_ENTRYPOINT
    return nvonnxparser::createParser(network, sample::gLogger.getTRTLogger());
#else
    return {};
#endif
}

nvcaffeparser1::ICaffeParser* sampleCreateCaffeParser()
{
#if DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT
    return nvcaffeparser1::createCaffeParser();
#else
    return {};
#endif
}

void shutdownCaffeParser()
{
#if DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT
    nvcaffeparser1::shutdownProtobufLibrary();
#endif
}

nvuffparser::IUffParser* sampleCreateUffParser()
{
#if DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT
    return nvuffparser::createUffParser();
#else
    return {};
#endif
}

void shutdownUffParser()
{
#if DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT
    nvuffparser::shutdownProtobufLibrary();
#endif
}

#endif // DEFINE_TRT_ENTRYPOINTS

#endif // TRT_SAMPLE_ENTRYPOINTS_H
