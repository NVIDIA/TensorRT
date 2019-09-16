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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleOptions.h"
#include "sampleEngines.h"

using namespace nvinfer1;
using namespace sample;

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

bool doInference(ICudaEngine& engine, const InferenceOptions& inference, const ReportingOptions& reporting)
{
    IExecutionContext* context = engine.createExecutionContext();

    // Dump inferencing time per layer basis
    SimpleProfiler profiler("Layer time");
    if (reporting.profile)
    {
        context->setProfiler(&profiler);
    }

    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (!engine.bindingIsInput(b))
        {
            continue;
        }
        auto dims = context->getBindingDimensions(b);
        if (dims.d[0] == -1)
        {
            auto shape = inference.shapes.find(engine.getBindingName(b));
            if (shape == inference.shapes.end())
            {
                gLogError << "Missing dynamic batch size in inference" << std::endl;
                return false;
            }
            dims.d[0] = shape->second.d[0];
            context->setBindingDimensions(b, dims);
        }
    }

    // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
    std::shared_ptr<ICudaEngine> emptyPtr{};
    std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
    samplesCommon::BufferManager bufferManager(aliasPtr, inference.batch, inference.batch ? nullptr : context);
    std::vector<void*> buffers = bufferManager.getDeviceBindings();

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    std::vector<float> times(reporting.avgs);
    for (int j = 0; j < inference.iterations; j++)
    {
        float totalGpu{0};  // GPU timer
        float totalHost{0}; // Host timer

        for (int i = 0; i < reporting.avgs; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            if (inference.batch)
            {
                context->enqueue(inference.batch, &buffers[0], stream, nullptr);
            }
            else
            {
                context->enqueueV2(&buffers[0], stream, nullptr);
            }
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }

        totalGpu /= reporting.avgs;
        totalHost /= reporting.avgs;
        gLogInfo << "Average over " << reporting.avgs << " runs is " << totalGpu << " ms (host walltime is "
                 << totalHost << " ms, " << static_cast<int>(reporting.percentile) << "\% percentile time is "
                 << percentile(reporting.percentile, times) << ")." << std::endl;
    }

    if (reporting.output)
    {
        bufferManager.copyOutputToHost();
        int nbBindings = engine.getNbBindings();
        for (int i = 0; i < nbBindings; i++)
        {
            if (!engine.bindingIsInput(i))
            {
                const char* tensorName = engine.getBindingName(i);
                gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
                bufferManager.dumpBuffer(gLogInfo, tensorName);
            }
        }
    }

    if (reporting.profile)
    {
        gLogInfo << profiler;
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();

    return true;
}

int main(int argc, char** argv)
{
    const std::string sampleName = "TensorRT.trtexec";
    auto sampleTest = gLogger.defineTest(sampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    Arguments args = argsToArgumentsMap(argc, argv);
    AllOptions options;

    if (!args.empty())
    {
        bool failed{false};
        try
        {
            options.parse(args);

            if (!args.empty())
            {
                for (const auto& arg : args)
                {
                    gLogError << "Unknown option: " << arg.first << " " << arg.second << std::endl;
                }
                failed = true;
            }
        }
        catch (const std::invalid_argument& arg)
        {
            gLogError << arg.what() << std::endl;
            failed = true;
        }

        if (failed)
        {
            AllOptions::help(std::cout);
            std::cout << "Note: the following options are not fully supported in trtexec:"
                         " dynamic shapes, multistream/threads, cuda graphs, json logs,"
                         " and actual data IO"
                      << std::endl;
            return gLogger.reportFail(sampleTest);
        }
    }
    else
    {
        options.helps = true;
    }

    if (options.helps)
    {
        AllOptions::help(std::cout);
        std::cout << "Note: the following options are not fully supported in trtexec:"
                     " dynamic shapes, multistream/threads, cuda graphs, json logs,"
                     " and actual data IO"
                  << std::endl;
        return gLogger.reportPass(sampleTest);
    }

    gLogInfo << options;
    if (options.reporting.verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    cudaSetDevice(options.system.device);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    for (const auto& pluginPath : options.system.plugins)
    {
        gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
        samplesCommon::loadLibrary(pluginPath);
    }

    ICudaEngine* engine{nullptr};
    if (options.build.load)
    {
        engine = loadEngine(options.build.engine, options.system.DLACore, gLogError);
    }
    else
    {
        engine = modelToEngine(options.model, options.build, options.system, gLogError);
    }
    if (!engine)
    {
        gLogError << "Engine could not be created" << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (options.build.save)
    {
        saveEngine(*engine, options.build.engine, gLogError);
    }

    if (!options.inference.skip)
    {
        if (options.build.safe && options.system.DLACore >= 0)
        {
            gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                        "then use dla_safety_runtime to run inference with saved DLA loadable, "
                        "or alternatively run with your own application"
                     << std::endl;
            return gLogger.reportFail(sampleTest);
        }
        if (!doInference(*engine, options.inference, options.reporting))
        {
            gLogError << "Inference failure" << std::endl;
            return gLogger.reportFail(sampleTest);
        }
    }
    engine->destroy();

    return gLogger.reportPass(sampleTest);
}
