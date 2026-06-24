/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! \file trtexec.cpp
//!
//! \brief Reusable trtexec implementation, separated from main() so that custom
//! command-line tools can be built on top of trtexec's engine-building and
//! inference workflow. See trtexec.h for details.

#include "trtexec.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sys/stat.h>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <system_error>
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
#include "sampleTuning.h"
#include "sampleUtils.h"

#include <nlohmann/json.hpp>

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
            sample::gLogWarning << "Plugin interface getSafetyPluginCreator return nullptr for "
                                << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                                << " in the safety plugin library: " << pluginArgs.libraryName << std::endl;
            sample::gLogWarning
                << "Please ensure interface function is implemented correctly and plugin name/namespace is matched."
                << std::endl;
            continue;
        }
        sample::gLogInfo << "Registering " << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                         << " for TensorRT safety." << std::endl;
        ErrorCode errorCode
            = safetyPluginRegistry->registerCreator(*pluginCreator, pluginAttr.pluginNamespace.c_str(), *gSafeRecorder);
        if (errorCode != ErrorCode::kSUCCESS)
        {
            sample::gLogWarning << "Failed to register safety plugin " << pluginAttr.pluginNamespace
                                << "::" << pluginAttr.pluginName << std::endl;
            if (errorCode == ErrorCode::kINVALID_ARGUMENT)
            {
                sample::gLogWarning << "Is getPluginName/getPluginNamespace/getPluginVersion interface implemented and "
                                       "return non-nullptr?"
                                    << std::endl;
            }
        }
    }
    return true;
}
#endif

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

// Sentinel returned by parseArgs() to signal that main() should continue into build/infer.
constexpr int32_t kCONTINUE_MAIN{-1};

//! Prepare \p iEnv for the profile run. Caller must have already populated \p iEnv.profiler.
//! On TRT-RTX, CUDA graphs capture obscure per-layer profile events, so the existing context is torn down and rebuilt
//! with CUDA graphs disabled. On TRT-Enterprise, the existing context is reused and the profiler is attached in place.
bool prepareProfileRun(InferenceEnvironmentBase& iEnv, InferenceOptions& infOpts, SystemOptions const& sysOpts)
{
    IExecutionContext& ctx = *static_cast<InferenceEnvironmentStd&>(iEnv).contexts.front();
    ctx.setProfiler(iEnv.profiler.get());
    ctx.setEnqueueEmitsProfile(false);
    return true;
}

//! Print the knob database JSON (from IBuilderConfig::getAllBuildRoutes()) and return EXIT_SUCCESS,
//! or EXIT_FAILURE on error. With a non-empty knobName, parse the JSON and emit only the
//! tuner_options entry whose `option` field matches (matching is leading-dash-insensitive so
//! `--helpBuildRoute=conv_use_long_w` and `--helpBuildRoute=-conv_use_long_w` behave identically).
//! No match → EXIT_FAILURE with a clear "no such knob" diagnostic.
//! Guarded by ENABLE_FEATURE_GLOBAL_PERF_TUNER: without the feature, getAllBuildRoutes() is not
//! linked in, so this returns EXIT_FAILURE with a diagnostic.
int32_t printBuildRouteHelp(std::string const& knobName)
{
    std::unique_ptr<IBuilder> builder{createBuilder()};
    if (!builder)
    {
        sample::gLogError << "Failed to create builder for --helpBuildRoute" << std::endl;
        return EXIT_FAILURE;
    }

    std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        sample::gLogError << "Failed to create builder config for --helpBuildRoute" << std::endl;
        return EXIT_FAILURE;
    }

    char const* allBuildRoutes = config->getAllBuildRoutes();
    if (allBuildRoutes == nullptr)
    {
        sample::gLogError << "getAllBuildRoutes() returned null" << std::endl;
        return EXIT_FAILURE;
    }

    // No filter → emit the database verbatim and exit.
    if (knobName.empty())
    {
        std::cout << allBuildRoutes << std::endl;
        return EXIT_SUCCESS;
    }

    // Filter mode: parse the JSON, pick the matching tuner_options entry, re-emit.
    nlohmann::ordered_json root;
    try
    {
        root = nlohmann::ordered_json::parse(allBuildRoutes);
    }
    catch (nlohmann::json::parse_error const& e)
    {
        sample::gLogError << "Failed to parse knob database JSON: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    if (!root.contains("tuner_options") || !root["tuner_options"].is_array())
    {
        sample::gLogError << "Knob database JSON has no 'tuner_options' array." << std::endl;
        return EXIT_FAILURE;
    }

    // Database stores names with a leading '-'; accept the user's input both with and without
    // the dash by comparing the unprefixed substrings.
    auto stripDash = [](std::string const& s) -> std::string {
        return (!s.empty() && s.front() == '-') ? s.substr(1) : s;
    };
    std::string const wantedKnob = stripDash(knobName);

    nlohmann::ordered_json filteredOptions = nlohmann::ordered_json::array();
    for (auto const& opt : root["tuner_options"])
    {
        if (opt.contains("option") && opt["option"].is_string()
            && stripDash(opt["option"].get<std::string>()) == wantedKnob)
        {
            filteredOptions.push_back(opt);
        }
    }
    if (filteredOptions.empty())
    {
        sample::gLogError << "--helpBuildRoute=" << knobName << ": no such knob in the database. "
                          << "Run --helpBuildRoute (no value) to see the full list." << std::endl;
        return EXIT_FAILURE;
    }

    nlohmann::ordered_json filtered;
    if (root.contains("tuner_version"))
    {
        filtered["tuner_version"] = root["tuner_version"];
    }
    filtered["tuner_options"] = std::move(filteredOptions);
    std::cout << filtered.dump(/*indent=*/2) << std::endl;
    return EXIT_SUCCESS;
}

// \param quiet If true, suppress option banner printing (used by child workers to avoid
//              repeating the same option dump that the parent already printed).
int32_t parseArgs(Logger::TestAtom& sampleTest, Arguments& args, AllOptions& options, bool quiet = false)
{
    // For DLA pre-procssing
    bool const kENABLE_STATIC_PLUGINS = true;
    options.system.enableStaticPlugins = kENABLE_STATIC_PLUGINS;

    // Start parsing
    static_assert(kCONTINUE_MAIN != EXIT_SUCCESS, "kCONTINUE_MAIN must not be EXIT_SUCCESS");
    static_assert(kCONTINUE_MAIN != EXIT_FAILURE, "kCONTINUE_MAIN must not be EXIT_FAILURE");
    if (parseHelp(args))
    {
        AllOptions::help(std::cout, kENABLE_STATIC_PLUGINS);
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
                AllOptions::help(std::cout, kENABLE_STATIC_PLUGINS);
                for (auto const& arg : args)
                {
                    sample::gLogError << "Unknown option: " << arg.first << " " << arg.second.first << std::endl;
                }
                failed = true;
            }
        }
        catch (std::invalid_argument const& arg)
        {
            AllOptions::help(std::cout, kENABLE_STATIC_PLUGINS);
            sample::gLogError << arg.what() << std::endl;
            failed = true;
        }

        if (failed)
        {
            return EXIT_FAILURE;
        }
    }
    else
    {
        options.helps = true;
    }

    if (options.helps)
    {
        AllOptions::help(std::cout, kENABLE_STATIC_PLUGINS);
        return EXIT_SUCCESS;
    }

#if defined(_WIN32)
    if (options.tuning.helpBuildRoute || !options.build.buildRoute.empty())
    {
        sample::gLogError << "--helpBuildRoute and --setBuildRoute are not supported on Windows." << std::endl;
        return EXIT_FAILURE;
    }
#endif

    // --helpBuildRoute prints the knob database JSON and exits. Suppresses all other flags
    // (they are parsed for validation but ignored). Lower precedence than --help (above),
    // matching the tests' expected ordering.
    if (options.tuning.helpBuildRoute)
    {
        return printBuildRouteHelp(options.tuning.helpBuildRouteKnob);
    }

    // Print the parsed options banner. Suppressed in child workers (quiet=true)
    // to avoid repeating the same dump that the parent already printed.
    if (!quiet)
    {
        sample::gLogInfo << options;
    }
    return kCONTINUE_MAIN;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int32_t runOnceBuildAndInfer(
    Logger::TestAtom& sampleTest, AllOptions& options, sample::PostConfigCallback const& postConfigHook = nullptr)
{
    if (options.reporting.verbose)
    {
        sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
    }
    std::string const jitInVersion;
    if (!options.build.cpuOnly)
    {
        setCudaDevice(options.system.device, sample::gLogInfo);
    }
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
#else  /* TRT_STATIC */
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

    // getSafePluginRegistry() mutates the singleton's stored ISafeRecorder on every call. The ONNX parser calls it
    // during plugin lookup with its own library-local static recorder; at process exit libnvonnxparser.so is
    // unloaded before libnvinfer_safe.so, leaving the singleton with a pointer into unmapped memory that its
    // destructor then dereferences via decRefCount(). Restore the recorder to this executable's process-lifetime
    // gSafeRecorder on every return path. The restore runs during stack unwind while both DSOs are still mapped,
    // relying on libnvonnxparser.so staying loaded for the duration of trtexecMain.
    struct RestoreSafeRecorderGuard
    {
        nvinfer2::safe::ISafePluginRegistry* mRegistry;
        ~RestoreSafeRecorderGuard() noexcept
        {
            if (mRegistry != nullptr && gSafeRecorder)
            {
                mRegistry->setSafeRecorder(*gSafeRecorder);
            }
        }
    } restoreSafeRecorderGuard{safetyPluginRegistry};

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
        return EXIT_FAILURE;
    }

    if (!options.build.safe && options.build.consistency)
    {
        sample::gLogInfo << "Skipping consistency checker on non-safety mode." << std::endl;
        options.build.consistency = false;
    }



    // Windows does not have setenv call
#if !defined(_WIN32)
    // Set CPU-only environment variable if the option is enabled
    if (options.build.cpuOnly)
    {
        // The use of `TRT_INTERNAL_OPTIONS` is special to TensorRT 10.15 and will disappear in later releases.
        sample::gLogInfo << "Setting CPU-only mode" << std::endl;
        char* internalOptions = std::getenv("TRT_INTERNAL_OPTIONS");
        std::string internalOptionsStr;
        if (internalOptions)
        {
            internalOptionsStr = std::string(internalOptions) + " --cpu_only=1";
        }
        else
        {
            internalOptionsStr = "--cpu_only=1";
        }
        setenv("TRT_INTERNAL_OPTIONS", internalOptionsStr.c_str(), 1);
    }
#endif // !defined(_WIN32)
    // Start engine building phase.
    std::unique_ptr<BuildEnvironment> bEnv(
        new BuildEnvironment(options.build.safe, options.build.versionCompatible, options.system.DLACore,
            options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath, sampleTest.getCmdline()));

    bool buildPass
        = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError, postConfigHook);

    if (!buildPass)
    {
        sample::gLogError << "Engine set up failed" << std::endl;
        return EXIT_FAILURE;
    }

    // Exit as version is already printed during getEngineBuildEnv
    if (options.build.getPlanVersionOnly)
    {
        return EXIT_SUCCESS;
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
                return EXIT_FAILURE;
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
                    return EXIT_FAILURE;
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
        return EXIT_SUCCESS;
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
        return EXIT_FAILURE;
#endif
    }

    // We avoid re-loading some dynamic plugins while deserializing
    // if they were already serialized with `setPluginsToSerialize`.
    std::vector<std::string> dynamicPluginsNotSerialized;
    for (auto& pluginName : options.system.dynamicPlugins)
    {
        if (std::find(
                options.system.setPluginsToSerialize.begin(), options.system.setPluginsToSerialize.end(), pluginName)
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
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    if (options.build.safe && options.system.DLACore >= 0)
    {
        sample::gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                            "then use dla_safety_runtime to run inference with saved DLA loadable, "
                            "or alternatively run with your own application"
                         << std::endl;
        return EXIT_FAILURE;
    }
    bool const profilerEnabled = options.reporting.profile || !options.reporting.exportProfile.empty();

    bool const layerInfoEnabled = options.reporting.layerInfo || !options.reporting.exportLayerInfo.empty();
    if (iEnv->safe && (profilerEnabled || layerInfoEnabled))
    {
        sample::gLogError << "Safe runtime does not support --dumpProfile or --exportProfile=<file> or "
                             "--dumpLayerInfo or --exportLayerInfo=<file>, please use "
                             "--verbose to print profiling info."
                          << std::endl;
        return EXIT_FAILURE;
    }
    if (!setUpInference(*iEnv, options.inference, options.system))
    {
        sample::gLogError << "Inference set up failed" << std::endl;
        return EXIT_FAILURE;
    }

    if (!options.build.safe)
    {
        printLayerInfo(options.reporting, iEnv->engine.get(),
            static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front().get());
        printOptimizationProfileInfo(options.reporting, iEnv->engine.get());
    }
    std::vector<InferenceTrace> trace;
    sample::gLogInfo << "Starting inference" << std::endl;

#if !defined(_WIN32)
    // Load ALL reference outputs for accuracy validation (if provided)
    // This loads all refPairs upfront since they will be used in the inner loop
    if (!options.build.safe)
    {
        auto* iEnvStd = static_cast<InferenceEnvironmentStd*>(iEnv.get());
        for (size_t pairIdx = 0; pairIdx < options.inference.refPairs.size(); ++pairIdx)
        {
            loadRefOutputs(*iEnv, options.inference, *iEnvStd->contexts.front(), pairIdx);
        }
    }
#if ENABLE_UNIFIED_BUILDER
    else
    {
        auto* iEnvSafe = static_cast<InferenceEnvironmentSafe*>(iEnv.get());
        for (size_t pairIdx = 0; pairIdx < options.inference.refPairs.size(); ++pairIdx)
        {
            loadRefOutputs(*iEnv, options.inference, *iEnvSafe->mClonedGraphs.front(), pairIdx);
        }
    }
#endif
#endif // !defined(_WIN32) && !TRT_WINML

    if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
    {
        sample::gLogError << "Error occurred during inference" << std::endl;
        return EXIT_FAILURE;
    }

    printPerformanceReport(
        trace, options.reporting, options.inference, sample::gLogInfo, sample::gLogWarning, sample::gLogVerbose);

    printOutput(options.reporting, *iEnv, options.inference.batch);

    if (profilerEnabled)
    {
        iEnv->profiler = std::make_unique<Profiler>();
        if (!prepareProfileRun(*iEnv, options.inference, options.system))
        {
            return EXIT_FAILURE;
        }
        if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return EXIT_FAILURE;
        }
    }
    printPerformanceProfile(options.reporting, *iEnv);

    // --tuningResultFile is the hidden parent->child IPC channel used by the
    // tuning loop. Write a compact JSON with gpu_time_ms, accuracy_failed, and
    // per-tensor accuracy_loss so the parent can update its cache + best
    // tracking after waitpid(). The flag is omitted from --help; an end user
    // who passes it manually still gets the same JSON, which is intentional —
    // it makes a tuning iteration reproducible by `--setBuildRoute=<route>
    // --tuningResultFile=...`.
    if (!options.tuning.tuningResultFile.empty())
    {
        double meanGpuTimeMs{0.0};
        if (!trace.empty())
        {
            double sum{0.0};
            for (auto const& t : trace)
            {
                sum += static_cast<double>(t.computeEnd - t.computeStart);
            }
            meanGpuTimeMs = sum / static_cast<double>(trace.size());
        }
        nlohmann::json j;
        j["gpu_time_ms"] = meanGpuTimeMs;
        j["accuracy_failed"] = iEnv->accuracyFailed;
        auto lossJson = nlohmann::json::object();
        for (auto const& [name, loss] : iEnv->accuracyLossValues)
        {
            lossJson[name] = loss;
        }
        j["accuracy_loss"] = std::move(lossJson);
        std::ofstream out(options.tuning.tuningResultFile);
        if (!out)
        {
            sample::gLogError << "Cannot open --tuningResultFile for writing: "
                              << options.tuning.tuningResultFile << std::endl;
        }
        else
        {
            out << j.dump(2) << std::endl;
        }
    }

    // Check if accuracy validation failed
    if (iEnv->accuracyFailed)
    {
        sample::gLogError << "Accuracy validation FAILED: one or more tensors exceeded the threshold." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int sample::trtexecMain(int argc, char** argv, PostConfigCallback const& postConfigHook)
{
    std::string const sampleName = "TensorRT.trtexec";

    auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);

    AllOptions options;
    try
    {
        sample::gLogger.reportTestStart(sampleTest);
        Arguments args = argsToArgumentsMap(argc, argv);
        int32_t const parseResult = parseArgs(sampleTest, args, options);
        if (parseResult != kCONTINUE_MAIN)
        {
            return parseResult;
        }
        int32_t const result = runOnceBuildAndInfer(sampleTest, options, postConfigHook);
        if (result != EXIT_SUCCESS)
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Uncaught exception detected: " << e.what() << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    return sample::gLogger.reportPass(sampleTest);
}

// ============================================================================
// Tuning-loop driver: parent side of --tuneBuildRoutes / --continue.
//
// Architecture: the parent never builds an engine itself. It enumerates routes,
// fork+execs a child trtexec per iteration with `--setBuildRoute=<route>` and
// `--tuningResultFile=<json>` injected, then waitpid()s and reads the JSON the
// child wrote. trtexecMain is therefore reused unmodified — every iteration is
// reproducible by re-running the child's argv by hand.
//
// Windows path: tuning isn't supported (no fork). runTuningLoop returns an
// error there. Linux is the only target until a separate cross-platform
// implementation is needed.
// ============================================================================

#if defined(_WIN32)

int32_t sample::runTuningLoop(int32_t /*argc*/, char** /*argv*/)
{
    sample::gLogError << "--tuneBuildRoutes is not supported on Windows (no fork())." << std::endl;
    return EXIT_FAILURE;
}

#else // POSIX && ENABLE_FEATURE_GLOBAL_PERF_TUNER

namespace
{

//! Per-iteration result extracted from the child's --tuningResultFile JSON.
struct IterationResult
{
    bool crashed{false};        //!< Child crashed or fork/waitpid failed.
    bool accuracyFailed{false}; //!< Child reported accuracy-threshold failure.
    int32_t exitCode{0};        //!< Child exit code (or -1 on fork/waitpid error).
    double gpuTimeMs{0.0};      //!< Mean GPU compute time (ms) from the trace.
    std::string errorMessage;   //!< Brief diagnostic for the parent log.
    std::unordered_map<std::string, double> accuracyLossValues;
};

//! Read the child's tuning result JSON. Returns a partial IterationResult
//! with crashed=true if the file is missing or unparseable.
IterationResult readChildResult(std::string const& jsonPath)
{
    IterationResult r;
    std::ifstream in(jsonPath);
    if (!in)
    {
        r.crashed = true;
        r.errorMessage = "missing tuning result file " + jsonPath
            + " (child likely crashed before writing)";
        return r;
    }
    try
    {
        nlohmann::json j;
        in >> j;
        r.gpuTimeMs = j.value("gpu_time_ms", 0.0);
        r.accuracyFailed = j.value("accuracy_failed", false);
        if (j.contains("accuracy_loss") && j["accuracy_loss"].is_object())
        {
            for (auto const& [k, v] : j["accuracy_loss"].items())
            {
                r.accuracyLossValues[k] = v.get<double>();
            }
        }
    }
    catch (std::exception const& e)
    {
        r.crashed = true;
        r.errorMessage = std::string{"failed to parse tuning result JSON: "} + e.what();
    }
    return r;
}

//! Fork+exec one child trtexec invocation for the given route. Reads the
//! resulting JSON and returns an IterationResult. Never throws — failures
//! are reported as `crashed=true` with a brief errorMessage.
IterationResult runChildForOneRoute(int32_t argc, char** argv, BigInt const& globalIndex, std::string const& route,
    std::string const& enginePath, std::string const& resultJsonPath)
{
    std::vector<std::string> storage;
    auto const childArgv = buildTuningChildArgv(argc, argv, route, enginePath, resultJsonPath, storage);

    sample::gLogInfo << "Tuning iteration [" << globalIndex.toString() << "]: " << route << std::endl;
    // Ensure stale result files from a previous iteration aren't mistaken for this one's output.
    std::remove(resultJsonPath.c_str());

    pid_t const pid = fork();
    if (pid < 0)
    {
        IterationResult r;
        r.crashed = true;
        r.exitCode = -1;
        r.errorMessage = std::string{"fork() failed: "} + std::strerror(errno);
        sample::gLogError << r.errorMessage << std::endl;
        return r;
    }
    if (pid == 0)
    {
        // Child: replace ourselves with a fresh trtexec invocation. execvp searches PATH for argv[0].
        execvp(childArgv[0], childArgv.data());
        // execvp returns only on failure.
        std::cerr << "execvp() failed: " << std::strerror(errno) << std::endl;
        _exit(127);
    }
    // Parent: wait for the child.
    int32_t status{};
    pid_t const w = waitpid(pid, &status, 0);
    if (w < 0)
    {
        IterationResult r;
        r.crashed = true;
        r.exitCode = -1;
        r.errorMessage = std::string{"waitpid() failed: "} + std::strerror(errno);
        return r;
    }

    IterationResult r = readChildResult(resultJsonPath);
    if (WIFEXITED(status))
    {
        r.exitCode = WEXITSTATUS(status);
        if (r.exitCode != EXIT_SUCCESS && !r.crashed)
        {
            // Child wrote a result file but exited non-zero — e.g. accuracy validation
            // exceeded the threshold. Not a crash; the result is still valid for tuning.
            r.errorMessage = "child exited with status " + std::to_string(r.exitCode);
        }
    }
    else if (WIFSIGNALED(status))
    {
        r.crashed = true;
        r.exitCode = -1;
        int32_t const sig = WTERMSIG(status);
        r.errorMessage = std::string{"child killed by signal "} + std::to_string(sig);
        char const* name = strsignal(sig);
        if (name != nullptr)
        {
            r.errorMessage += " (";
            r.errorMessage += name;
            r.errorMessage += ")";
        }
    }
    return r;
}

//! Resolve the tuning context up-front: parse the user's expression, query the
//! knob database from --setBuildRoute's empty-route side-effect path (the
//! default-constructed BuilderConfig populates `allBuildRoutes` from Myelin),
//! and expand into a TuningContext.
//!
//! Returns false (and logs) if the expression is malformed or empty.
bool buildTuningContext(TuningOptions const& tuning, TuningContext& ctx)
{
    // Get the knob database JSON by spinning up a default BuilderConfig.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>{createBuilder()};
    if (!builder)
    {
        sample::gLogError << "buildTuningContext: createBuilder failed" << std::endl;
        return false;
    }
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError << "buildTuningContext: createBuilderConfig failed" << std::endl;
        return false;
    }
    char const* allRoutesC = config->getAllBuildRoutes();
    std::string const allRoutesStr = allRoutesC != nullptr ? allRoutesC : "";
    if (allRoutesStr.empty())
    {
        sample::gLogError << "IBuilderConfig::getAllBuildRoutes() returned empty — "
                          << "tuning requires the Myelin knob database, which is only available on "
                          << "enterprise TRT (non-RTX, non-WoA)." << std::endl;
        return false;
    }
    BuildRouteKnobDatabase db;
    if (!db.loadFromJsonString(allRoutesStr))
    {
        sample::gLogError << "Failed to parse knob database from getAllBuildRoutes()." << std::endl;
        return false;
    }
    BuildRouteExprParser parser(db);
    auto parsed = parser.parse(tuning.tuningExpr);
    if (!parsed.has_value())
    {
        sample::gLogError << "Failed to parse --tuneBuildRoutes expression: " << parser.getError() << std::endl;
        return false;
    }
    ctx.parsedExprs = std::move(*parsed);
    ctx.searchAlgorithm = tuning.tuningSearchAlgorithm;
    ctx.tunerVersion = db.getTunerVersion();
    ctx.defaultBuildRoute = db.buildDefaultPath();
    // Fill defaultValues parallel to parsedExprs (used by fast / mixed mode).
    ctx.defaultValues.clear();
    ctx.defaultValues.reserve(ctx.parsedExprs.size());
    for (auto const& p : ctx.parsedExprs)
    {
        ctx.defaultValues.emplace_back(db.getDefaultValue(p.mKnobName));
    }
    ctx.totalCount = ctx.count();
    sample::gLogInfo << "Expanded to " << ctx.totalCount.toString() << " build route configurations" << std::endl;
    return true;
}

//! Storage + view of an argv reconstructed from a tuning-cache header. The caller
//! reads `argc` / `argv()` and passes them to parseArgs(); `resumeFromIter` is the
//! iteration index to skip up to. `storage` owns the strings backing `argvPtrs`.
struct ContinueResumeState
{
    std::vector<std::string> storage;
    std::vector<char*> argvPtrs;
    int32_t argc{0};
    int64_t resumeFromIter{0};

    [[nodiscard]] char** argv() noexcept
    {
        return argvPtrs.data();
    }
};

//! \brief Validate the bare-`--continue` invocation and rebuild argv from the cache header.
//!
//! The user must pass exactly `--continue` + `--tuningCacheFile=<path>` (in any order) and
//! no other options — the cache header is the source of truth for everything else. The
//! function logs and returns false on validation or I/O errors so the caller can fail fast.
//!
//! On success, `state` carries the reconstructed argv (with absolute paths) and the resume
//! iteration index. Caller must pass the SAME argv[0] as the running binary, since the cache
//! stores the original argv[0] which may not match a relocated trtexec.
bool reconstructArgvForContinue(int32_t argc, char** argv, ContinueResumeState& state)
{
    constexpr char const* kCACHE_FLAG = "--tuningCacheFile=";
    uint64_t const flagLen = std::strlen(kCACHE_FLAG);

    // Pre-check: the only allowed args are `--continue` and `--tuningCacheFile=<path>`.
    std::string cachePath;
    for (int32_t i = 1; i < argc; ++i)
    {
        if (argv[i] == nullptr)
        {
            continue;
        }
        std::string const arg(argv[i]);
        bool const isContinue = (arg == "--continue");
        bool const isCacheFile = (std::strncmp(argv[i], kCACHE_FLAG, flagLen) == 0);
        if (!isContinue && !isCacheFile)
        {
            sample::gLogError << "--continue requires exactly --tuningCacheFile=<path>; "
                              << "no other options are allowed (got '" << arg << "'). "
                              << "All options come from the cache header." << std::endl;
            return false;
        }
        if (isCacheFile)
        {
            cachePath = argv[i] + flagLen;
        }
    }
    if (cachePath.empty())
    {
        sample::gLogError << "--continue requires --tuningCacheFile=<path> on the command line." << std::endl;
        return false;
    }

    auto header = readTuningCacheHeader(cachePath);
    if (!header.has_value())
    {
        sample::gLogError << "--continue: failed to read cache header from " << cachePath << std::endl;
        return false;
    }
    state.resumeFromIter = header->completedIterations;
    state.storage = reconstructArgvFromCacheHeader(*header, argv[0], cachePath);
    state.argvPtrs.reserve(state.storage.size() + 1);
    for (auto& s : state.storage)
    {
        state.argvPtrs.emplace_back(s.data());
    }
    state.argvPtrs.emplace_back(nullptr);
    state.argc = static_cast<int32_t>(state.storage.size());
    sample::gLogInfo << "--continue: resuming from iteration " << state.resumeFromIter << " using " << state.argc
                     << " args reconstructed from cache header." << std::endl;
    return true;
}

//! \brief Cross-flag validation specific to the tuning loop. Returns false (and logs)
//! when --saveEngine is missing in a mode that requires it. Called after parseArgs
//! and after the cache-header re-assert.
bool validateTuningOptions(AllOptions const& options)
{
    // --saveEngine is optional for pure benchmarking, but required when --loadRefOutputs is set:
    // the accuracy-validation path picks a "best" engine and needs somewhere to persist it.
    if (options.build.engine.empty() && !options.tuning.dryRun)
    {
        bool const hasRefOutputs = !options.inference.refPairs.empty() && !options.inference.refPairs[0].second.empty();
        if (hasRefOutputs)
        {
            sample::gLogError << "--tuneBuildRoutes with --loadRefOutputs requires --saveEngine to "
                              << "persist the best engine selected by accuracy validation." << std::endl;
            return false;
        }
        sample::gLogWarning << "--tuneBuildRoutes without --saveEngine: best engine will not be saved." << std::endl;
    }
    return true;
}

//! \brief Print the dryRun enumeration of routes. No engine work is performed.
void emitDryRunListing(TuningContext const& ctx)
{
    sample::gLogInfo << "--dryRun: " << ctx.totalCount.toString() << " build routes would be tried:" << std::endl;
    for (BigInt i{0}; i < ctx.totalCount; ++i)
    {
        sample::gLogInfo << "[" << i.toString() << "]:" << ctx.getPathAtIndex(i) << std::endl;
    }
}

//! Shared mutable state for a phase (or phase 1 + phase 2 of mixed-mode) of the tuning loop.
//! Aggregated into a single struct so runOnePhase can be a free function instead of a deeply
//! nested lambda.
struct PhaseState
{
    AllOptions const& options;                           //!< Parsed options for this run.
    Logger::TestAtom const& sampleTest;                  //!< For TASK_BEGIN/END/ABORT banners.
    pid_t const ppid{};                                  //!< Parent PID, used in temp filenames.
    std::chrono::steady_clock::time_point const startTime; //!< Loop start, for --tuningTimeOut.
    int32_t const argc{};                                  //!< Parent argv (passed verbatim to children).
    char** const argv{};                                   //!< Parent argv (passed verbatim to children).
    // Mutable across iterations and across phases:
    BigInt successCount{0};
    double bestGpuTimeMs{std::numeric_limits<double>::infinity()};
    std::string bestEnginePath;
    std::string bestRoute;
};

//! Build the temp-engine path for one iteration. With --saveAllEngines, use a stable
//! per-iteration name under the user's --saveEngine (matches the format the turtle
//! tests assert against: `<engine>.iter<N>`). Without it, use a pid-scoped temp
//! that gets unlinked at promotion time. The phase label disambiguates the temp
//! path so mixed-mode phase 2 cannot overwrite a phase-1 file mid-flight.
std::string makeIterationEnginePath(PhaseState const& state, char const* phaseLabel, BigInt const& i)
{
    if (state.options.build.saveAllEngines)
    {
        return state.options.build.engine + ".iter" + i.toString();
    }
    return "/tmp/trtexec_tuning_" + std::to_string(state.ppid) + "_iter" + phaseLabel + "_" + i.toString() + ".plan";
}

//! \brief Run one phase of the tuning loop (phase 1, or phase 2 of mixed mode).
//!
//! Iterates phaseCtx.totalCount times, fork+execs a child per iteration, and updates
//! `state` with the best route seen. When `positiveKnobs` is non-null and we're past
//! the baseline iteration (i==0), records each iteration that beats the baseline so
//! mixed-mode can build its phase-2 sub-context.
//!
//! Returns false if --tuningTimeOut tripped (so the caller stops chaining phases),
//! true on normal completion.
//!
//! NOLINT: orchestrator function that touches every per-iteration concern (timeout, route
//! enumeration, child invocation, best-tracker update, mixed-mode positive-knob collection,
//! cache writing, temp-file cleanup, TASK_BEGIN/END/ABORT logging). Further extraction would
//! either fragment one logical iteration step across multiple functions or require passing
//! the same PhaseState everywhere — both make the per-iteration story harder to read.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool runOnePhase(PhaseState& state, TuningContext const& phaseCtx, char const* phaseLabel,
    std::vector<MixedSearchKnobResult>* positiveKnobs, double* baselineGpuTimeMsOut, int64_t skipUntil)
{
    sample::gLogInfo << "Tuning " << phaseLabel << ": " << phaseCtx.totalCount.toString() << " iterations." << std::endl;
    double baselineGpuTimeMs = std::numeric_limits<double>::infinity();
    for (BigInt i{0}; i < phaseCtx.totalCount; ++i)
    {
        // --continue: skip iterations already in the cache.
        if (skipUntil > 0 && i < BigInt{static_cast<uint64_t>(skipUntil)})
        {
            continue;
        }
        if (state.options.tuning.timeout > 0)
        {
            auto const elapsedS = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - state.startTime).count();
            if (elapsedS >= state.options.tuning.timeout)
            {
                sample::gLogInfo << "Tuning timeout reached (" << state.options.tuning.timeout
                                 << "s); stopping early." << std::endl;
                return false;
            }
        }

        std::string const route = phaseCtx.getPathAtIndex(i);
        std::string const enginePath = makeIterationEnginePath(state, phaseLabel, i);
        std::string const jsonPath = "/tmp/trtexec_tuning_" + std::to_string(state.ppid)
            + "_iter" + i.toString() + ".json";

        sample::gLogger.reportTaskBegin(state.sampleTest, i.toString(), route);

        IterationResult const result = runChildForOneRoute(state.argc, state.argv, i, route, enginePath, jsonPath);

        if (!result.crashed && result.exitCode == EXIT_SUCCESS)
        {
            ++state.successCount;
            if (result.gpuTimeMs < state.bestGpuTimeMs)
            {
                state.bestGpuTimeMs = result.gpuTimeMs;
                state.bestEnginePath = enginePath;
                state.bestRoute = route;
            }
            // Track baseline (index 0) so mixed-mode can decide "positive knob".
            if (i.isZero())
            {
                baselineGpuTimeMs = result.gpuTimeMs;
            }
            sample::gLogger.reportTaskEnd(state.sampleTest, i.toString(), route);
        }
        else
        {
            sample::gLogWarning << "Iteration [" << i.toString() << "] failed: "
                                << (result.errorMessage.empty() ? "(no message)" : result.errorMessage) << std::endl;
            sample::gLogger.reportTaskAbort(state.sampleTest, i.toString(), route);
        }
        // For mixed-mode phase 1, collect knobs that beat the baseline.
        if (positiveKnobs != nullptr && !i.isZero() && baselineGpuTimeMs != std::numeric_limits<double>::infinity())
        {
            collectPositiveKnobFromResult(
                result.crashed, result.gpuTimeMs, baselineGpuTimeMs, i, phaseCtx, *positiveKnobs);
        }
        // Append this iteration's result to the tuning cache file (--tuningCacheFile).
        if (!state.options.tuning.tuningCacheFile.empty())
        {
            writeTuningCacheIteration(state.options.tuning.tuningCacheFile, i.toUint64(), route, result.crashed,
                result.errorMessage, result.accuracyLossValues, result.gpuTimeMs);
        }
        std::remove(jsonPath.c_str());
    }
    if (baselineGpuTimeMsOut != nullptr)
    {
        *baselineGpuTimeMsOut = baselineGpuTimeMs;
    }
    return true;
}

//! \brief Copy the best-iteration engine to the user's --saveEngine path and emit the
//! final summary. Cleans up the per-iteration temp engine if --saveAllEngines was off.
//! Returns the trtexec exit code (pass if any iteration succeeded, fail otherwise).
int32_t finalizeBestEngine(PhaseState const& state, BigInt const& totalCount)
{
    if (state.bestEnginePath.empty())
    {
        sample::gLogError << "No tuning iteration succeeded; no engine written." << std::endl;
        return sample::gLogger.reportFail(state.sampleTest);
    }
    if (state.bestEnginePath != state.options.build.engine)
    {
        std::ifstream src(state.bestEnginePath, std::ios::binary);
        std::ofstream dst(state.options.build.engine, std::ios::binary);
        dst << src.rdbuf();
        if (!state.options.build.saveAllEngines)
        {
            // Per-iteration engine was a temp; clean it up now that it's been promoted.
            std::remove(state.bestEnginePath.c_str());
        }
    }
    sample::gLogInfo << "Best iteration: " << state.bestRoute << " (gpu_time_ms=" << state.bestGpuTimeMs << ")"
                     << std::endl;
    sample::gLogInfo << "Tuning summary: " << state.successCount.toString() << " / " << totalCount.toString()
                     << " iterations succeeded." << std::endl;
    return sample::gLogger.reportPass(state.sampleTest);
}

} // namespace

int32_t sample::runTuningLoop(int32_t argc, char** argv)
{
    std::string const sampleName =
        "TensorRT.trtexec";
    auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);

    // 1. --continue: rebuild argv from the cache header. The header stores the original argv
    //    with absolute paths and the expanded tuning expression; everything else is rejected.
    ContinueResumeState resume;
    int32_t effectiveArgc = argc;
    char** effectiveArgv = argv;
    if (peekArg(argc, argv, "--continue"))
    {
        if (!reconstructArgvForContinue(argc, argv, resume))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
        effectiveArgc = resume.argc;
        effectiveArgv = resume.argv();
    }

    // 2. Parse options against the effective argv (cache-reconstructed for --continue, raw otherwise).
    AllOptions options;
    try
    {
        Arguments args = argsToArgumentsMap(effectiveArgc, effectiveArgv);
        int32_t const parseResult = parseArgs(sampleTest, args, options);
        if (parseResult != kCONTINUE_MAIN)
        {
            return parseResult;
        }
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Argument parse error in tuning loop: " << e.what() << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }

    // 3. Resume mode: the reconstructed argv has no --continue, so parseArgs left continueFromCache
    //    false. Re-assert it now so the cache-header writer below preserves existing rows.
    if (effectiveArgv != argv)
    {
        options.tuning.continueFromCache = true;
    }

    if (!validateTuningOptions(options))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // 4. Expand the tuning expression into a TuningContext.
    TuningContext ctx;
    if (!buildTuningContext(options.tuning, ctx))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (options.tuning.dryRun)
    {
        emitDryRunListing(ctx);
        return sample::gLogger.reportPass(sampleTest);
    }

    // 5. Fresh tuning runs write a cache header; resumed runs skip it (the file already has one).
    if (!options.tuning.tuningCacheFile.empty() && !options.tuning.continueFromCache)
    {
        writeTuningCacheHeader(
            options.tuning.tuningCacheFile, options, argc, argv, ctx.tunerVersion, ctx.defaultBuildRoute);
    }

    // 6. Phase 1 — always runs. For mixed mode, may be followed by a phase 2 over positive knobs.
    // Children must see the reconstructed argv when resuming — the user's bare
    // `--continue --tuningCacheFile=path` argv has none of the original build/inference flags.
    PhaseState state{options, sampleTest, getpid(), std::chrono::steady_clock::now(), effectiveArgc, effectiveArgv};
    std::vector<MixedSearchKnobResult> positiveKnobs;
    double phase1BaselineMs{std::numeric_limits<double>::infinity()};
    bool const isMixed = options.tuning.tuningSearchAlgorithm == TuningSearchAlgorithm::kMIXED;
    bool const phase1Completed
        = runOnePhase(state, ctx, "phase1", isMixed ? &positiveKnobs : nullptr, &phase1BaselineMs, resume.resumeFromIter);

    if (phase1Completed && isMixed && positiveKnobs.size() > 1)
    {
        sample::gLogInfo << "Mixed search: " << positiveKnobs.size()
                         << " positive knobs identified; entering phase 2." << std::endl;
        TuningContext const phase2Ctx = buildMixedPhase2Context(ctx, positiveKnobs);
        // Phase 2 always starts fresh (no resume mid-phase-2).
        (void) runOnePhase(state, phase2Ctx, "phase2", nullptr, nullptr, 0);
    }
    else if (isMixed)
    {
        sample::gLogInfo << "Mixed search: " << positiveKnobs.size()
                         << " positive knob(s); skipping phase 2 (need >1)." << std::endl;
    }

    // 7. Promote the best iteration's engine to the user's --saveEngine path.
    return finalizeBestEngine(state, ctx.totalCount);
}

#endif // POSIX && ENABLE_FEATURE_GLOBAL_PERF_TUNER
