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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"

using namespace nvinfer1;
using namespace sample;

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

bool printLayerInfo(const ReportingOptions& reporting, const InferenceEnvironment& iEnv)
{
    if (reporting.layerInfo)
    {
        sample::gLogInfo << "Layer Information:" << std::endl;
        sample::gLogInfo << getLayerInformation(iEnv, nvinfer1::LayerInformationFormat::kONELINE) << std::flush;
    }
    if (!reporting.exportLayerInfo.empty())
    {
        std::ofstream os(reporting.exportLayerInfo, std::ofstream::trunc);
        os << getLayerInformation(iEnv, nvinfer1::LayerInformationFormat::kJSON) << std::flush;
    }
    return true;
}

void printPerformanceProfile(const ReportingOptions& reporting, const InferenceEnvironment& iEnv)
{
    if (reporting.profile)
    {
        iEnv.profiler->print(sample::gLogInfo);
    }
    if (!reporting.exportProfile.empty())
    {
        iEnv.profiler->exportJSONProfile(reporting.exportProfile);
    }
}

void printOutput(const ReportingOptions& reporting, const InferenceEnvironment& iEnv, int32_t batch)
{
    if (reporting.output)
    {
        dumpOutputs(*iEnv.context.front(), *iEnv.bindings.front(), sample::gLogInfo);
    }
    if (!reporting.exportOutput.empty())
    {
        exportJSONOutput(*iEnv.context.front(), *iEnv.bindings.front(), reporting.exportOutput, batch);
    }
}

int main(int argc, char** argv)
{
    const std::string sampleName = "TensorRT.trtexec";

    auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    Arguments args = argsToArgumentsMap(argc, argv);
    AllOptions options;

    if (parseHelp(args))
    {
        AllOptions::help(std::cout);
        return EXIT_SUCCESS;
    }

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
                    sample::gLogError << "Unknown option: " << arg.first << " " << arg.second << std::endl;
                }
                failed = true;
            }
        }
        catch (const std::invalid_argument& arg)
        {
            sample::gLogError << arg.what() << std::endl;
            failed = true;
        }

        if (failed)
        {
            AllOptions::help(std::cout);
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    else
    {
        options.helps = true;
    }

    if (options.helps)
    {
        AllOptions::help(std::cout);
        return sample::gLogger.reportPass(sampleTest);
    }

    sample::gLogInfo << options;
    if (options.reporting.verbose)
    {
        sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
    }

    setCudaDevice(options.system.device, sample::gLogInfo);
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "TensorRT version: " << getInferLibVersion() << std::endl;
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    for (const auto& pluginPath : options.system.plugins)
    {
        sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
        samplesCommon::loadLibrary(pluginPath);
    }

    if (options.build.safe && !sample::hasSafeRuntime())
    {
        sample::gLogError << "Safety is not supported because safety runtime library is unavailable." << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!options.build.safe && options.build.consistency)
    {
        sample::gLogInfo << "Skipping consistency checker on non-safety mode." << std::endl;
        options.build.consistency = false;
    }

    InferenceEnvironment iEnv;
    TrtUniquePtr<INetworkDefinition> networkForRefit;
    Parser parserHoldingWeightsMem;
    {
        // Scope the build phase so any held memory is released before moving to inference phase.
        BuildEnvironment bEnv;
        const time_point buildStartTime{std::chrono::high_resolution_clock::now()};
        bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, bEnv, sample::gLogError);
        const time_point buildEndTime{std::chrono::high_resolution_clock::now()};

        if (!buildPass)
        {
            sample::gLogError << "Engine set up failed" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        sample::gLogInfo << "Engine " << (options.build.load ? "loaded" : "built") << " in "
                         << duration(buildEndTime - buildStartTime).count() << " sec." << std::endl;

        std::swap(iEnv.engine, bEnv.engine);
        std::swap(networkForRefit, bEnv.network);
        std::swap(iEnv.serializedEngine, bEnv.serializedEngine);
        parserHoldingWeightsMem = std::move(bEnv.parser);
        std::swap(iEnv.safeEngine, bEnv.safeEngine);
    }
    iEnv.safe = options.build.safe;

    if (!options.build.safe && iEnv.engine.get()->isRefittable())
    {
        if (options.reporting.refit)
        {
            dumpRefittable(*iEnv.engine.get());
        }
        if (options.inference.timeRefit)
        {
            if (networkForRefit.operator bool())
            {
                const bool success = timeRefit(*networkForRefit, *iEnv.engine);
                if (!success)
                {
                    sample::gLogError << "Engine refit failed." << std::endl;
                    return sample::gLogger.reportFail(sampleTest);
                }
            }
            else
            {
                sample::gLogWarning << "Network not available, skipped timing refit." << std::endl;
            }
        }
    }
    // release resources for refit only.
    parserHoldingWeightsMem = Parser{};
    // network released after parser! parser destructor depends on network.
    networkForRefit.reset();

    if (options.inference.timeDeserialize)
    {
        if (timeDeserialize(iEnv))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
        return sample::gLogger.reportPass(sampleTest);
    }
    else
    {
        // Release the serialized memory when not in use before allocating bindings in order to
        // reduce memory usage.
        iEnv.serializedEngine.reset();
    }

    printLayerInfo(options.reporting, iEnv);

    if (options.inference.skip)
    {
        return sample::gLogger.reportPass(sampleTest);
    }

    if (options.build.safe && options.system.DLACore >= 0)
    {
        sample::gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                            "then use dla_safety_runtime to run inference with saved DLA loadable, "
                            "or alternatively run with your own application"
                         << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    const bool profilerEnabled = options.reporting.profile || !options.reporting.exportProfile.empty();
    if (profilerEnabled && !options.inference.rerun)
    {
        iEnv.profiler.reset(new Profiler);
        if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
        {
            options.inference.graph = false;
            sample::gLogWarning << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                                   "and disabled CUDA graph."
                                << std::endl;
        }
    }

    if (!setUpInference(iEnv, options.inference))
    {
        sample::gLogError << "Inference set up failed" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    std::vector<InferenceTrace> trace;
    sample::gLogInfo << "Starting inference" << std::endl;

    if (!runInference(options.inference, iEnv, options.system.device, trace))
    {
        sample::gLogError << "Error occurred during inference" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    if (profilerEnabled && !options.inference.rerun)
    {
        sample::gLogWarning << "The network timing report will not be accurate due to extra synchronizations "
                               "when profiler is enabled." << std::endl;
        sample::gLogWarning << "Add --separateProfileRun to profile layer timing in a separate run."
                            << std::endl;
    }

    printPerformanceReport(trace, options.reporting, static_cast<float>(options.inference.warmup),
        options.inference.batch, sample::gLogInfo, sample::gLogWarning, sample::gLogVerbose);
    printOutput(options.reporting, iEnv, options.inference.batch);

    if (profilerEnabled && options.inference.rerun)
    {
        auto* profiler = new Profiler;
        iEnv.profiler.reset(profiler);
        iEnv.context.front()->setProfiler(profiler);
        if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
        {
            options.inference.graph = false;
            sample::gLogWarning << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                                   "and disabled CUDA graph."
                                << std::endl;
        }
        if (!runInference(options.inference, iEnv, options.system.device, trace))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    printPerformanceProfile(options.reporting, iEnv);

    return sample::gLogger.reportPass(sampleTest);
}
