/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <functional>
#include <iostream>
#include <memory>
#include <sys/stat.h>
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

#if ENABLE_UNIFIED_BUILDER
using namespace nvinfer2::safe;
__attribute__((weak)) std::shared_ptr<sample::SampleSafeRecorder> gSafeRecorder
    = std::make_shared<sample::SampleSafeRecorder>(nvinfer2::safe::Severity::kINFO);
#endif

namespace
{
using LibraryPtr = std::unique_ptr<DynamicLibrary>;

std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};
std::function<void*(void*, void*, int32_t)> pCreateInferRefitterInternal{};
std::function<void*(void*, int32_t)> pCreateInferBuilderInternal{};
std::function<void*(void*, void*, int)> pCreateNvOnnxParserInternal{};
std::function<void*(void*, void*, int)> pCreateNvOnnxRefitterInternal{};

//! Track runtime used for the execution of trtexec.
//! Must be tracked as a global variable due to how library init functions APIs are organized.
RuntimeMode gUseRuntime = RuntimeMode::kFULL;

bool initNvinfer()
{
#if !TRT_STATIC
    static LibraryPtr libnvinferPtr{};
    auto fetchPtrs = [](DynamicLibrary* l) {
        pCreateInferRuntimeInternal = l->symbolAddress<void*(void*, int32_t)>("createInferRuntime_INTERNAL");
        try
        {
            pCreateInferRefitterInternal
                = l->symbolAddress<void*(void*, void*, int32_t)>("createInferRefitter_INTERNAL");
        }
        catch (const std::exception& e)
        {
            sample::gLogWarning << "Could not load function createInferRefitter_INTERNAL : " << e.what() << std::endl;
        }

        if (gUseRuntime == RuntimeMode::kFULL)
        {
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
        pCreateNvOnnxRefitterInternal
            = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParserRefitter_INTERNAL");
    };
    return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
    pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
    pCreateNvOnnxRefitterInternal = createNvOnnxParserRefitter_INTERNAL;
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

nvonnxparser::IParserRefitter* createONNXRefitter(nvinfer1::IRefitter& refitter)
{
    if (!initNvonnxparser())
    {
        return {};
    }
    ASSERT(pCreateNvOnnxRefitterInternal != nullptr);
    return static_cast<nvonnxparser::IParserRefitter*>(
        pCreateNvOnnxRefitterInternal(&refitter, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

#if ENABLE_UNIFIED_BUILDER

bool processSafetyPluginLibrary(nvinfer2::safe::ISafePluginRegistry* safetyPluginRegistry, DynamicLibrary* libPtr,
    samplesSafeCommon::SafetyPluginLibraryArgument const& pluginArgs)
{
    if (libPtr == nullptr)
    {
        sample::gLogError << "Cannot open safety plugin library " << pluginArgs.libraryName << std::endl;
        return false;
    }
    std::string const pluginGetterSymbolName{"getSafetyPluginCreator"};
    auto pGetSafetyPluginCreator
        = libPtr->symbolAddress<void*(char const*, char const*)>(pluginGetterSymbolName.c_str());
    if (pGetSafetyPluginCreator == nullptr)
    {
        sample::gLogError << "Cannot find plugin creator getter symbol from plugin library: " << pluginArgs.libraryName
                          << std::endl;
        sample::gLogError << "Please ensure interface function is correctly implemented and exported." << std::endl;
        return false;
    }

    for (auto const& pluginAttr : pluginArgs.pluginAttrs)
    {
        auto pluginCreator = static_cast<IPluginCreatorInterface*>(
            pGetSafetyPluginCreator(pluginAttr.pluginNamespace.c_str(), pluginAttr.pluginName.c_str()));
        if (pluginCreator == nullptr)
        {
            sample::gLogInfo << "Cannot find plugin " << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                             << " in the safety plugin library: " << pluginArgs.libraryName << std::endl;
            continue;
        }
        sample::gLogInfo << "Registering " << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                         << " for TensorRT safety." << std::endl;
        safetyPluginRegistry->registerCreator(*pluginCreator, pluginAttr.pluginNamespace.c_str(), *gSafeRecorder);
    }
    return true;
}
#endif

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
                    AllOptions::help(std::cout);
                    for (auto const& arg : args)
                    {
                        sample::gLogError << "Unknown option: " << arg.first << " " << arg.second.first << std::endl;
                    }
                    failed = true;
                }
            }
            catch (std::invalid_argument const& arg)
            {
                AllOptions::help(std::cout);
                sample::gLogError << arg.what() << std::endl;
                failed = true;
            }

            if (failed)
            {
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
        std::string const jitInVersion;
        setCudaDevice(options.system.device, sample::gLogInfo);
        sample::gLogInfo << std::endl;
        sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
                         << NV_TENSORRT_PATCH << jitInVersion << std::endl;

        // Record specified runtime
        gUseRuntime = options.build.useRuntime;
#if !TRT_STATIC
        LibraryPtr nvinferPluginLib{};
#endif /* TRT_STATIC */
        std::vector<LibraryPtr> pluginLibs;
        if (gUseRuntime == RuntimeMode::kFULL && !options.build.safe)
        {
            sample::gLogInfo << "Loading standard plugins" << std::endl;
#if !TRT_STATIC
            nvinferPluginLib = loadLibrary(kNVINFER_PLUGIN_LIBNAME);
            auto pInitLibNvinferPlugins
                = nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
#else /* TRT_STATIC */
            auto pInitLibNvinferPlugins = initLibNvInferPlugins;
#endif /* TRT_STATIC */
            ASSERT(pInitLibNvinferPlugins != nullptr);
            pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "");
            for (auto const& pluginPath : options.system.plugins)
            {
                sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
                pluginLibs.emplace_back(loadLibrary(pluginPath));
            }
        }
        else if (gUseRuntime == RuntimeMode::kFULL && options.build.safe)
        {
            sample::gLogInfo << "Skipping standard plugin loading due to --safe flag" << std::endl;
        }
        else if (!options.system.plugins.empty())
        {
            throw std::runtime_error("TRT-18412: Plugins require --useRuntime=full.");
        }
#if ENABLE_UNIFIED_BUILDER
        auto safetyPluginRegistry = sample::safe::getSafePluginRegistry(*gSafeRecorder);
        ASSERT(safetyPluginRegistry != nullptr);

        if (!options.system.safetyPlugins.empty())
        {
            for (auto const& safetyPluginArg : options.system.safetyPlugins)
            {
                sample::gLogInfo << "Loading supplied safety plugin library with manual registration: "
                                 << safetyPluginArg.libraryName << std::endl;
                auto pluginLib = loadLibrary(safetyPluginArg.libraryName);
                processSafetyPluginLibrary(safetyPluginRegistry, pluginLib.get(), safetyPluginArg);
                pluginLibs.emplace_back(std::move(pluginLib));
            }
        }
#endif // ENABLE_UNIFIED_BUILDER
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

        if (options.build.safe)
        {
            sample::gLogInfo << "StronglyTyped is enabled by default on safety mode." << std::endl;
            options.build.stronglyTyped = true;
        }

       // Start engine building phase.
        std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(options.build.safe, options.build.versionCompatible,
            options.system.DLACore, options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath,
            sampleTest.getCmdline()));

        bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError);

        if (!buildPass)
        {
            sample::gLogError << "Engine set up failed" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

#if ENABLE_UNIFIED_BUILDER
        safetyPluginRegistry->setSafeRecorder(*gSafeRecorder);
#endif // ENABLE_UNIFIED_BUILDER

        // Exit as version is already printed during getEngineBuildEnv
        if (options.build.getPlanVersionOnly)
        {
            return sample::gLogger.reportPass(sampleTest);
        }


        // dynamicPlugins may have been updated by getEngineBuildEnv above
        bEnv->engine.setDynamicPlugins(options.system.dynamicPlugins);
       // When some options are enabled, engine deserialization is not supported on the platform that the engine was
       // built.
        bool const supportDeserialization = !options.build.safe && !options.build.buildDLAStandalone
            && options.build.runtimePlatform == nvinfer1::RuntimePlatform::kSAME_AS_BUILD;

        if (supportDeserialization && options.build.refittable)
        {
            auto* engine = bEnv->engine.get();
            if (options.reporting.refit)
            {
                dumpRefittable(*engine);
            }
            // Refit from ONNX model
            if (!options.inference.refitOnnxModel.empty())
            {
                bool const success = refitFromOnnx(*engine, options.inference.refitOnnxModel, options.inference.threads);
                if (!success)
                {
                    sample::gLogError << "Engine refit from ONNX model failed." << std::endl;
                    return sample::gLogger.reportFail(sampleTest);
                }
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
            if (supportDeserialization)
            {
                printLayerInfo(options.reporting, bEnv->engine.get(), nullptr);
                printOptimizationProfileInfo(options.reporting, bEnv->engine.get());
            }
            sample::gLogInfo << "Skipped inference phase since --skipInference is added." << std::endl;
            return sample::gLogger.reportPass(sampleTest);
        }

        std::unique_ptr<InferenceEnvironmentBase> iEnv;

        if (!options.build.safe)
        {
            iEnv = std::make_unique<InferenceEnvironmentStd>(*bEnv);
        }
        else
        {
#if ENABLE_UNIFIED_BUILDER
            iEnv = std::make_unique<InferenceEnvironmentSafe>(*bEnv);
#else
            sample::gLogInfo << "--safe flag is enabled but application is not compatible with safety." << std::endl;
            return sample::gLogger.reportFail(sampleTest);
#endif
        }

        // We avoid re-loading some dynamic plugins while deserializing
        // if they were already serialized with `setPluginsToSerialize`.
        std::vector<std::string> dynamicPluginsNotSerialized;
        for (auto& pluginName : options.system.dynamicPlugins)
        {
            if (std::find(options.system.setPluginsToSerialize.begin(), options.system.setPluginsToSerialize.end(),
                    pluginName)
                == options.system.setPluginsToSerialize.end())
            {
                dynamicPluginsNotSerialized.emplace_back(pluginName);
            }
        }

        iEnv->engine.setDynamicPlugins(dynamicPluginsNotSerialized);
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

        bool const layerInfoEnabled = options.reporting.layerInfo || !options.reporting.exportLayerInfo.empty();
        if (iEnv->safe && (profilerEnabled || layerInfoEnabled))
        {
            sample::gLogError << "Safe runtime does not support --dumpProfile or --exportProfile=<file> or "
                                 "--dumpLayerInfo or --exportLayerInfo=<file>, please use "
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
            printLayerInfo(options.reporting, iEnv->engine.get(),
                static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front().get());
            printOptimizationProfileInfo(options.reporting, iEnv->engine.get());
        }
        std::vector<InferenceTrace> trace;
        sample::gLogInfo << "Starting inference" << std::endl;

        if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
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
            static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front()->setProfiler(profiler);
            static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front()->setEnqueueEmitsProfile(false);
            if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
            {
                options.inference.graph = false;
                sample::gLogWarning
                    << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                       "and disabled CUDA graph."
                    << std::endl;
            }
            if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
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
