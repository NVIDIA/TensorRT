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
#include <memory>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleOptions.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleReporting.h"

using namespace nvinfer1;
using namespace sample;

int main(int argc, char** argv)
{
    const std::string sampleName = "TensorRT.trtexec";
    const std::string supportNote{"Note: CUDA graphs is not supported in this version."};

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
            std::cout << supportNote << std::endl;
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
        std::cout << supportNote << std::endl;
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

    InferenceEnvironment iEnv;
    iEnv.engine = getEngine(options.model, options.build, options.system, gLogError);
    if (!iEnv.engine)
    {
        gLogError << "Engine set up failed" << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (options.inference.skip)
    {
        return gLogger.reportPass(sampleTest);
    }

    if (options.build.safe && options.system.DLACore >= 0)
    {
        gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                    "then use dla_safety_runtime to run inference with saved DLA loadable, "
                    "or alternatively run with your own application" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (options.reporting.profile || !options.reporting.exportTimes.empty())
    {
        iEnv.profiler.reset(new Profiler);
    }

    setUpInference(iEnv, options.inference);
    std::vector<InferenceTrace> trace;
    runInference(options.inference, iEnv, trace);

    printPerformanceReport(trace, options.reporting, static_cast<float>(options.inference.warmup), options.inference.batch, gLogInfo);

    if (options.reporting.output)
    {
        dumpOutputs(*iEnv.context.front(), *iEnv.bindings.front(), gLogInfo);
    }
    if (!options.reporting.exportOutput.empty())
    {
        exportJSONOutput(*iEnv.context.front(), *iEnv.bindings.front(), options.reporting.exportOutput);
    }
    if (!options.reporting.exportTimes.empty())
    {
        exportJSONTrace(trace, options.reporting.exportTimes);
    }
    if (options.reporting.profile)
    {
        iEnv.profiler->print(gLogInfo);
    }
    if (!options.reporting.exportProfile.empty())
    {
        iEnv.profiler->exportJSONProfile(options.reporting.exportProfile);
    }

    return gLogger.reportPass(sampleTest);
}
