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

void printPerformanceProfile(const ReportingOptions& reporting, const InferenceEnvironment& iEnv, std::ostream& os)
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

void printOutput(const ReportingOptions& reporting, const InferenceEnvironment& iEnv, std::ostream& os, int32_t batch)
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

    InferenceEnvironment iEnv;
    TrtUniquePtr<INetworkDefinition> networkForRefit;
    Parser parserHoldingWeightsMem;
    const time_point buildStartTime{std::chrono::high_resolution_clock::now()};
    std::tie(iEnv.engine, networkForRefit, parserHoldingWeightsMem) = getEngineNetworkParserTuple(options.model, options.build, options.system, sample::gLogError);
    const time_point buildEndTime{std::chrono::high_resolution_clock::now()};
    if (iEnv.engine)
    {
        sample::gLogInfo << "Engine " << (options.build.load ? "loaded" : "built")
            << " in " << duration(buildEndTime - buildStartTime).count() << " sec." << std::endl;
    }
    else
    {
        sample::gLogError << "Engine set up failed" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    if (iEnv.engine.get()->isRefittable())
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

    if ((options.reporting.profile || !options.reporting.exportProfile.empty()) && !options.inference.rerun)
    {
        iEnv.profiler.reset(new Profiler);
        if (options.inference.graph)
        {
            options.inference.graph = false;
            sample::gLogWarning << "Profiler does not work when CUDA graph is enabled. Ignored --useCudaGraph flag "
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

    printPerformanceReport(trace, options.reporting, static_cast<float>(options.inference.warmup),
        options.inference.batch, sample::gLogInfo, sample::gLogWarning, sample::gLogVerbose);
    printOutput(options.reporting, iEnv, sample::gLogInfo, options.inference.batch);

    if ((options.reporting.profile || !options.reporting.exportProfile.empty()) && options.inference.rerun)
    {
        auto* profiler = new Profiler;
        iEnv.profiler.reset(profiler);
        iEnv.context.front()->setProfiler(profiler);
        if (options.inference.graph)
        {
            options.inference.graph = false;
            sample::gLogWarning << "Profiler does not work when CUDA graph is enabled. Ignored --useCudaGraph flag "
                                   "and disabled CUDA graph in the second run with the profiler."
                                << std::endl;
        }
        if (!runInference(options.inference, iEnv, options.system.device, trace))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    printPerformanceProfile(options.reporting, iEnv, sample::gLogInfo);

    return sample::gLogger.reportPass(sampleTest);
}
