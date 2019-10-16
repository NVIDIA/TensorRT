/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
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

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
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
        if (engine.bindingIsInput(b))
        {
            auto dims = context->getBindingDimensions(b);
            const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim){ return dim == -1; });
            if (isDynamicInput)
            {
                auto shape = inference.shapes.find(engine.getBindingName(b));
                if (shape == inference.shapes.end())
                {
                    gLogError << "Missing dynamic batch size in inference" << std::endl;
                    return false;
                }
                dims = shape->second;
                context->setBindingDimensions(b, dims);
            }
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
    unsigned int cudaEventFlags = inference.spin ? cudaEventDefault : cudaEventBlockingSync;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

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
                         " and actual data IO" << std::endl;
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
                     " and actual data IO" << std::endl;
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
                        "or alternatively run with your own application" << std::endl;
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
