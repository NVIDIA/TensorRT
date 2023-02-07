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

bool printLayerInfo(
    ReportingOptions const& reporting, nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context)
{
    if (reporting.layerInfo)
    {
        sample::gLogInfo << "Layer Information:" << std::endl;
        sample::gLogInfo << getLayerInformation(engine, context, nvinfer1::LayerInformationFormat::kONELINE)
                         << std::flush;
    }
    if (!reporting.exportLayerInfo.empty())
    {
        std::ofstream os(reporting.exportLayerInfo, std::ofstream::trunc);
        os << getLayerInformation(engine, context, nvinfer1::LayerInformationFormat::kJSON) << std::flush;
    }
    return true;
}

void printPerformanceProfile(ReportingOptions const& reporting, InferenceEnvironment const& iEnv)
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

namespace details
{
template <typename ContextType>
void dump(std::unique_ptr<ContextType> const& context, std::unique_ptr<Bindings> const& binding, ReportingOptions const& reporting, int32_t batch)
{
    if (!context)
    {
        sample::gLogError << "Empty context! Skip printing outputs." << std::endl;
        return;
    }
    if (reporting.output)
    {
        dumpOutputs(*context, *binding, sample::gLogInfo);
    }
    if (!reporting.exportOutput.empty())
    {
        exportJSONOutput(*context, *binding, reporting.exportOutput, batch);
    }
}
} // namespace details

void printOutput(ReportingOptions const& reporting, InferenceEnvironment const& iEnv, int32_t batch)
{
    auto const& binding = iEnv.bindings.at(0);
    if (!binding)
    {
        sample::gLogError << "Empty bindings! Skip printing outputs." << std::endl;
        return;
    }

    if (iEnv.safe)
    {
        auto const& context = iEnv.safeContexts.at(0);
        details::dump(context, binding, reporting, batch);
    }
    else
    {
        auto const& context = iEnv.contexts.at(0);
        details::dump(context, binding, reporting, batch);
    }
}

int main(int argc, char** argv)
{
    std::string const sampleName = "TensorRT.trtexec";

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
                for (auto const& arg : args)
                {
                    sample::gLogError << "Unknown option: " << arg.first << " " << arg.second << std::endl;
                }
                failed = true;
            }
        }
        catch (std::invalid_argument const& arg)
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
    sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR
        << "." << NV_TENSORRT_PATCH << std::endl;
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    for (auto const& pluginPath : options.system.plugins)
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

    // Start engine building phase.
    std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(options.build.safe, options.system.DLACore));

    time_point const buildStartTime{std::chrono::high_resolution_clock::now()};
    bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError);
    time_point const buildEndTime{std::chrono::high_resolution_clock::now()};

    if (!buildPass)
    {
        sample::gLogError << "Engine set up failed" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogInfo << "Engine " << (options.build.load ? "loaded" : "built") << " in "
                        << duration(buildEndTime - buildStartTime).count() << " sec." << std::endl;

    if (!options.build.safe && options.build.refittable)
    {
        auto* engine = bEnv->engine.get();
        if (options.reporting.refit)
        {
            dumpRefittable(*engine);
        }
        if (options.inference.timeRefit)
        {
            if (bEnv->network.operator bool())
            {
                bool const success = timeRefit(*bEnv->network, *engine, options.inference.threads);
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

    if (options.build.buildOnly)
    {
        if (!options.build.safe)
        {
            printLayerInfo(options.reporting, bEnv->engine.get(), nullptr);
        }
        sample::gLogInfo << "Skipped inference phase since --buildOnly is added." << std::endl;
        return sample::gLogger.reportPass(sampleTest);
    }

    // Start inference phase.
    std::unique_ptr<InferenceEnvironment> iEnv(new InferenceEnvironment(*bEnv));

    // Delete build environment.
    bEnv.reset();

    if (options.inference.timeDeserialize)
    {
        if (timeDeserialize(*iEnv))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
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

    bool const profilerEnabled = options.reporting.profile || !options.reporting.exportProfile.empty();

    if (iEnv->safe && profilerEnabled)
    {
        sample::gLogError << "Safe runtime does not support --dumpProfile or --exportProfile=<file>, please use "
                             "--verbose to print profiling info."
                          << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    if (profilerEnabled && !options.inference.rerun)
    {
        iEnv->profiler.reset(new Profiler);
        if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
        {
            options.inference.graph = false;
            sample::gLogWarning << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                                   "and disabled CUDA graph."
                                << std::endl;
        }
    }

    if (!setUpInference(*iEnv, options.inference, options.system))
    {
        sample::gLogError << "Inference set up failed" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!options.build.safe)
    {
        printLayerInfo(options.reporting, iEnv->engine.get(), iEnv->contexts.front().get());
    }

    std::vector<InferenceTrace> trace;
    sample::gLogInfo << "Starting inference" << std::endl;

    if (!runInference(options.inference, *iEnv, options.system.device, trace))
    {
        sample::gLogError << "Error occurred during inference" << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    if (profilerEnabled && !options.inference.rerun)
    {
        sample::gLogInfo << "The e2e network timing is not reported since it is inaccurate due to the extra "
                         << "synchronizations when the profiler is enabled." << std::endl;
        sample::gLogInfo << "To show e2e network timing report, add --separateProfileRun to profile layer timing in a "
                         << "separate run or remove --dumpProfile to disable the profiler."
                         << std::endl;
    }
    else
    {
        printPerformanceReport(trace, options.reporting, static_cast<float>(options.inference.warmup),
            options.inference.batch, sample::gLogInfo, sample::gLogWarning, sample::gLogVerbose);
    }

    printOutput(options.reporting, *iEnv, options.inference.batch);

    if (profilerEnabled && options.inference.rerun)
    {
        auto* profiler = new Profiler;
        iEnv->profiler.reset(profiler);
        iEnv->contexts.front()->setProfiler(profiler);
        iEnv->contexts.front()->setEnqueueEmitsProfile(false);
        if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
        {
            options.inference.graph = false;
            sample::gLogWarning << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                                   "and disabled CUDA graph."
                                << std::endl;
        }
        if (!runInference(options.inference, *iEnv, options.system.device, trace))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    printPerformanceProfile(options.reporting, *iEnv);

    return sample::gLogger.reportPass(sampleTest);
}
