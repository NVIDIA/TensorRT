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
using namespace samplesCommon;

namespace
{
using LibraryPtr = std::unique_ptr<DynamicLibrary>;

#if !TRT_STATIC
#if defined(_WIN32)
std::string const kNVINFER_PLUGIN_LIBNAME{"nvinfer_plugin.dll"};
std::string const kNVINFER_LIBNAME{"nvinfer.dll"};
std::string const kNVONNXPARSER_LIBNAME{"nvonnxparser.dll"};
std::string const kNVPARSERS_LIBNAME{"nvparsers.dll"};
std::string const kNVINFER_LEAN_LIBNAME{"nvinfer_lean.dll"};
std::string const kNVINFER_DISPATCH_LIBNAME{"nvinfer_dispatch.dll"};

std::string const kMANGLED_UFF_PARSER_CREATE_NAME{"?createUffParser@nvuffparser@@YAPEAVIUffParser@1@XZ"};
std::string const kMANGLED_CAFFE_PARSER_CREATE_NAME{"?createCaffeParser@nvcaffeparser1@@YAPEAVICaffeParser@1@XZ"};
std::string const kMANGLED_UFF_PARSER_SHUTDOWN_NAME{"?shutdownProtobufLibrary@nvuffparser@@YAXXZ"};
std::string const kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME{"?shutdownProtobufLibrary@nvcaffeparser1@@YAXXZ"};
#else
std::string const kNVINFER_PLUGIN_LIBNAME = std::string{"libnvinfer_plugin.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LIBNAME = std::string{"libnvinfer.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVONNXPARSER_LIBNAME = std::string{"libnvonnxparser.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVPARSERS_LIBNAME = std::string{"libnvparsers.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LEAN_LIBNAME = std::string{"libnvinfer_lean.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_DISPATCH_LIBNAME
    = std::string{"libnvinfer_dispatch.so."} + std::to_string(NV_TENSORRT_MAJOR);

std::string const kMANGLED_UFF_PARSER_CREATE_NAME{"_ZN11nvuffparser15createUffParserEv"};
std::string const kMANGLED_CAFFE_PARSER_CREATE_NAME{"_ZN14nvcaffeparser117createCaffeParserEv"};
std::string const kMANGLED_UFF_PARSER_SHUTDOWN_NAME{"_ZN11nvuffparser23shutdownProtobufLibraryEv"};
std::string const kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME{"_ZN14nvcaffeparser123shutdownProtobufLibraryEv"};
#endif
#endif // !TRT_STATIC
std::function<void*(void*, int32_t)>
    pCreateInferRuntimeInternal{};
std::function<void*(void*, void*, int32_t)> pCreateInferRefitterInternal{};
std::function<void*(void*, int32_t)> pCreateInferBuilderInternal{};
std::function<void*(void*, void*, int)> pCreateNvOnnxParserInternal{};
std::function<nvuffparser::IUffParser*()> pCreateUffParser{};
std::function<nvcaffeparser1::ICaffeParser*()> pCreateCaffeParser{};
std::function<void()> pShutdownUffLibrary{};
std::function<void(void)> pShutdownCaffeLibrary{};

//! Track runtime used for the execution of trtexec.
//! Must be tracked as a global variable due to how library init functions APIs are organized.
RuntimeMode gUseRuntime = RuntimeMode::kFULL;

#if !TRT_STATIC
inline std::string const& getRuntimeLibraryName(RuntimeMode const mode)
{
    switch (mode)
    {
    case RuntimeMode::kFULL: return kNVINFER_LIBNAME;
    case RuntimeMode::kDISPATCH: return kNVINFER_DISPATCH_LIBNAME;
    case RuntimeMode::kLEAN: return kNVINFER_LEAN_LIBNAME;
    }
    throw std::runtime_error("Unknown runtime mode");
}

template <typename FetchPtrs>
bool initLibrary(LibraryPtr& libPtr, std::string const& libName, FetchPtrs fetchFunc)
{
    if (libPtr != nullptr)
    {
        return true;
    }
    try
    {
        libPtr.reset(new DynamicLibrary{libName});
        fetchFunc(libPtr.get());
    }
    catch (std::exception const& e)
    {
        libPtr.reset();
        sample::gLogError << "Could not load library " << libName << ": " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        libPtr.reset();
        sample::gLogError << "Could not load library " << libName << std::endl;
        return false;
    }

    return true;
}
#endif // !TRT_STATIC

bool initNvinfer()
{
#if !TRT_STATIC
    static LibraryPtr libnvinferPtr{};
    auto fetchPtrs = [](DynamicLibrary* l) {
        pCreateInferRuntimeInternal = l->symbolAddress<void*(void*, int32_t)>("createInferRuntime_INTERNAL");

        if (gUseRuntime == RuntimeMode::kFULL)
        {
            pCreateInferRefitterInternal
                = l->symbolAddress<void*(void*, void*, int32_t)>("createInferRefitter_INTERNAL");
            pCreateInferBuilderInternal = l->symbolAddress<void*(void*, int32_t)>("createInferBuilder_INTERNAL");
        }
    };
    return initLibrary(libnvinferPtr, getRuntimeLibraryName(gUseRuntime), fetchPtrs);
#else
    pCreateInferRuntimeInternal = createInferRuntime_INTERNAL;
    pCreateInferRefitterInternal = createInferRefitter_INTERNAL;
    pCreateInferBuilderInternal = createInferBuilder_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

bool initNvonnxparser()
{
#if !TRT_STATIC
    static LibraryPtr libnvonnxparserPtr{};
    auto fetchPtrs = [](DynamicLibrary* l) {
        pCreateNvOnnxParserInternal = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParser_INTERNAL");
    };
    return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
    pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

bool initNvparsers()
{
#if !TRT_STATIC
    static LibraryPtr libnvparsersPtr{};
    auto fetchPtrs = [](DynamicLibrary* l) {
        // TODO: get equivalent Windows symbol names
        pCreateUffParser = l->symbolAddress<nvuffparser::IUffParser*()>(kMANGLED_UFF_PARSER_CREATE_NAME.c_str());
        pCreateCaffeParser
            = l->symbolAddress<nvcaffeparser1::ICaffeParser*()>(kMANGLED_CAFFE_PARSER_CREATE_NAME.c_str());
        pShutdownUffLibrary = l->symbolAddress<void()>(kMANGLED_UFF_PARSER_SHUTDOWN_NAME.c_str());
        pShutdownCaffeLibrary = l->symbolAddress<void(void)>(kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME.c_str());
    };
    return initLibrary(libnvparsersPtr, kNVPARSERS_LIBNAME, fetchPtrs);
#else
    pCreateUffParser = nvuffparser::createUffParser;
    pCreateCaffeParser = nvcaffeparser1::createCaffeParser;
    pShutdownUffLibrary = nvuffparser::shutdownProtobufLibrary;
    pShutdownCaffeLibrary = nvcaffeparser1::shutdownProtobufLibrary;
    return true;
#endif // !TRT_STATIC
}

} // namespace

IRuntime* createRuntime()
{
    if (!initNvinfer())
    {
        return {};
    }
    ASSERT(pCreateInferRuntimeInternal != nullptr);
    return static_cast<IRuntime*>(pCreateInferRuntimeInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

IBuilder* createBuilder()
{
    if (!initNvinfer())
    {
        return {};
    }
    ASSERT(pCreateInferBuilderInternal != nullptr);
    return static_cast<IBuilder*>(pCreateInferBuilderInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

IRefitter* createRefitter(ICudaEngine& engine)
{
    if (!initNvinfer())
    {
        return {};
    }
    ASSERT(pCreateInferRefitterInternal != nullptr);
    return static_cast<IRefitter*>(pCreateInferRefitterInternal(&engine, &gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

nvonnxparser::IParser* createONNXParser(INetworkDefinition& network)
{
    if (!initNvonnxparser())
    {
        return {};
    }
    ASSERT(pCreateNvOnnxParserInternal != nullptr);
    return static_cast<nvonnxparser::IParser*>(
        pCreateNvOnnxParserInternal(&network, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

nvcaffeparser1::ICaffeParser* sampleCreateCaffeParser()
{
    if (!initNvparsers())
    {
        return {};
    }
    ASSERT(pCreateCaffeParser != nullptr);
    return pCreateCaffeParser();
}

void shutdownCaffeParser()
{
    if (!initNvparsers())
    {
        return;
    }
    ASSERT(pShutdownCaffeLibrary != nullptr);
    pShutdownCaffeLibrary();
}

nvuffparser::IUffParser* sampleCreateUffParser()
{
    if (!initNvparsers())
    {
        return {};
    }
    ASSERT(pCreateUffParser != nullptr);
    return pCreateUffParser();
}

void shutdownUffParser()
{
    if (!initNvparsers())
    {
        return;
    }
    ASSERT(pShutdownUffLibrary != nullptr);
    pShutdownUffLibrary();
}

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

int main(int argc, char** argv)
{
    std::string const sampleName = "TensorRT.trtexec";

    auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);

    try
    {
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
        sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
                         << NV_TENSORRT_PATCH << std::endl;

        // Record specified runtime
        gUseRuntime = options.build.useRuntime;

#if !TRT_STATIC
        LibraryPtr nvinferPluginLib{};
#endif
        std::vector<LibraryPtr> pluginLibs;
        if (gUseRuntime == RuntimeMode::kFULL)
        {
            if (!options.build.versionCompatible)
            {
                sample::gLogInfo << "Loading standard plugins" << std::endl;
#if !TRT_STATIC
                nvinferPluginLib = loadLibrary(kNVINFER_PLUGIN_LIBNAME);
                auto pInitLibNvinferPlugins
                    = nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
#else
                auto pInitLibNvinferPlugins = initLibNvInferPlugins;
#endif
                ASSERT(pInitLibNvinferPlugins != nullptr);
                pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "");
            }
            else
            {
                sample::gLogInfo << "Not loading standard plugins since --versionCompatible is specified." << std::endl;
            }
            for (auto const& pluginPath : options.system.plugins)
            {
                sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
                pluginLibs.emplace_back(loadLibrary(pluginPath));
            }
        }
        else if (!options.system.plugins.empty())
        {
            throw std::runtime_error("TRT-18412: Plugins require --useRuntime=full.");
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
        std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(options.build.safe, options.build.versionCompatible,
            options.system.DLACore, options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath));

        time_point const buildStartTime{std::chrono::high_resolution_clock::now()};
        bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError);
        time_point const buildEndTime{std::chrono::high_resolution_clock::now()};

        if (!buildPass)
        {
            sample::gLogError << "Engine set up failed" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // dynamicPlugins may have been updated by getEngineBuildEnv above
        bEnv->engine.setDynamicPlugins(options.system.dynamicPlugins);

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

        if (options.build.skipInference)
        {
            if (!options.build.safe)
            {
                printLayerInfo(options.reporting, bEnv->engine.get(), nullptr);
            }
            sample::gLogInfo << "Skipped inference phase since --skipInference is added." << std::endl;
            return sample::gLogger.reportPass(sampleTest);
        }

        // Start inference phase.
        std::unique_ptr<InferenceEnvironment> iEnv(new InferenceEnvironment(*bEnv));

        // Delete build environment.
        bEnv.reset();

        if (options.inference.timeDeserialize)
        {
            if (timeDeserialize(*iEnv, options.system))
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
                sample::gLogWarning
                    << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
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
            sample::gLogInfo
                << "To show e2e network timing report, add --separateProfileRun to profile layer timing in a "
                << "separate run or remove --dumpProfile to disable the profiler." << std::endl;
        }
        else
        {
            printPerformanceReport(trace, options.reporting, options.inference, sample::gLogInfo, sample::gLogWarning,
                sample::gLogVerbose);
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
                sample::gLogWarning
                    << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
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
    catch (std::exception const& e)
    {
        sample::gLogError << "Uncaught exception detected: " << e.what() << std::endl;
    }
    return sample::gLogger.reportFail(sampleTest);
}
