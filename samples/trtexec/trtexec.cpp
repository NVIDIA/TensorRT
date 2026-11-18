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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    auto fetchPtrs = [](DynamicLibrary& l) {
        pCreateInferRuntimeInternal = l.symbolAddress<void*(void*, int32_t)>("createInferRuntime_INTERNAL");
        try
        {
            pCreateInferRefitterInternal
                = l.symbolAddress<void*(void*, void*, int32_t)>("createInferRefitter_INTERNAL");
        }
        catch (std::exception const& e)
        {
            sample::gLogWarning << "Could not load function createInferRefitter_INTERNAL : " << e.what() << std::endl;
        }

        if (gUseRuntime == RuntimeMode::kFULL)
        {
            pCreateInferBuilderInternal = l.symbolAddress<void*(void*, int32_t)>("createInferBuilder_INTERNAL");
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
    auto fetchPtrs = [](DynamicLibrary& l) {
        pCreateNvOnnxParserInternal = l.symbolAddress<void*(void*, void*, int)>("createNvOnnxParser_INTERNAL");
        pCreateNvOnnxRefitterInternal
            = l.symbolAddress<void*(void*, void*, int)>("createNvOnnxParserRefitter_INTERNAL");
    };
    return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
    pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
    pCreateNvOnnxRefitterInternal = createNvOnnxParserRefitter_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

} // namespace

int32_t sample::reportRuntimeVersion()
{
    int32_t major{};
    int32_t minor{};
    int32_t patch{};
    int32_t build{};

#if !TRT_STATIC
    LibraryPtr libnvinferPtr{};
    auto getVersion = [&](samplesCommon::DynamicLibrary& library) {
        major = library.symbolAddress<int32_t()>("getInferLibMajorVersion")();
        minor = library.symbolAddress<int32_t()>("getInferLibMinorVersion")();
        patch = library.symbolAddress<int32_t()>("getInferLibPatchVersion")();
        build = library.symbolAddress<int32_t()>("getInferLibBuildVersion")();
    };
    if (!initLibrary(libnvinferPtr, getRuntimeLibraryName(RuntimeMode::kFULL), getVersion))
    {
        return EXIT_FAILURE;
    }
#else
    major = getInferLibMajorVersion();
    minor = getInferLibMinorVersion();
    patch = getInferLibPatchVersion();
    build = getInferLibBuildVersion();
#endif

    std::cout << major << "." << minor << "." << patch << "." << build << std::endl;
    return EXIT_SUCCESS;
}

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
    auto stripDash
        = [](std::string const& s) -> std::string { return (!s.empty() && s.front() == '-') ? s.substr(1) : s; };
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
    auto safetyPluginRegistry
        = sample::safe::getSafePluginRegistry(*gSafeRecorder, options.build.getSafeRuntimeSettings());
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

    if (!options.build.safe && options.build.reference)
    {
        sample::gLogInfo << "Skipping reference checker on non-safety mode." << std::endl;
        options.build.reference = false;
    }



    // Start engine building phase.
    std::unique_ptr<BuildEnvironment> bEnv(
        new BuildEnvironment(options.build.safe, options.build.versionCompatible, options.system.DLACore,
            options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath, sampleTest.getCmdline()));

    if (!bEnv->greenContexts.initialize(options.build, options.system.device, sample::gLogError))
    {
        sample::gLogError << "CUDA green context set up failed" << std::endl;
        return EXIT_FAILURE;
    }

    bEnv->engine.setDLAWorkspaceAllocationStrategy(options.system.dlaWorkspaceAllocationStrategy);


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
        iEnv = std::make_unique<InferenceEnvironmentSafe>(*bEnv, options.build.getSafeRuntimeSettings());
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
            sample::gLogError << "Cannot open --tuningResultFile for writing: " << options.tuning.tuningResultFile
                              << std::endl;
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
        r.errorMessage = "missing tuning result file " + jsonPath + " (child likely crashed before writing)";
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

//! \brief Fork and execute one child trtexec invocation for a concrete route.
//! Strip parent-only tuning arguments, then supply the route, engine path, and result-JSON path.
//! Wait for the child and combine its reported timing/accuracy with its exit status. Process launch,
//! wait, signal, or missing-result failures are returned as crashed results with an error message.
//! A reported accuracy failure may have a nonzero exit status without being classified as a crash.
IterationResult runChildForOneRoute(int32_t argc, char** argv, std::string const& route, std::string const& enginePath,
    std::string const& resultJsonPath)
{
    std::vector<std::string> storage;
    auto const childArgv = buildTuningChildArgv(argc, argv, route, enginePath, resultJsonPath, storage);

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

//! \brief Parse the tuning expression and construct its search strategy without expanding routes.
//! Query getAllBuildRoutes() on a builder configuration for knob definitions, validate candidate
//! values, and retain parsed expressions, defaults, and version metadata in the context.
//!
//! \param tuning Parsed expression, algorithm selection, and cache path.
//! \param[out] ctx Context owning the configured strategy and its common search inputs.
//! \return False with a diagnostic if builder/configuration creation fails, knob metadata is
//! unavailable or invalid, or the expression cannot be parsed; true when the context is ready.
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
                          << "tuning requires the knob database, which is only available on "
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
    ctx.searchStrategy = createSearchStrategy(tuning.tuningSearchAlgorithm, tuning.tuningCacheFile);
    ctx.tunerVersion = db.getTunerVersion();
    ctx.defaultBuildRoute = db.buildDefaultPath();
    // Fill defaultValues parallel to parsedExprs (used by fast / mixed mode).
    ctx.defaultValues.clear();
    ctx.defaultValues.reserve(ctx.parsedExprs.size());
    for (auto const& p : ctx.parsedExprs)
    {
        ctx.defaultValues.emplace_back(db.getDefaultValue(p.mKnobName));
    }
    if (auto const total = ctx.count())
    {
        sample::gLogInfo << "Expanded to " << total->toString() << " build route configurations" << std::endl;
    }
    return true;
}

//! \brief Own the arguments and cached progress needed to start a continued run.
//!
//! The cache is read before option parsing because its header supplies the original arguments.
//! Keep the record count and best measurement here until runTuningLoop() constructs TuningContext
//! and TuningState. They initialize the execution index and winner without a second full cache scan.
//! The default values also describe a fresh run: start at zero with no successful measurement.
struct ContinueResumeState
{
    std::vector<std::string> storage; //!< Owns the strings used by the reconstructed argv.
    std::vector<char*> argvPtrs;      //!< Pointers into storage, followed by a null sentinel.
    int32_t argc{0};                  //!< Argument count, excluding the sentinel.
    BigInt resumeFromIter{0};         //!< Completed record count; becomes ctx.globalIndex, without narrowing.
    BigInt successCount{0};           //!< Successful cached measurements, including the saved winner.

    //! Fastest successful cached measurement; absent for an empty cache or when every trial failed.
    //! Its time, route, and engine path preserve the winner even if no new trial improves on it.
    std::optional<CachedIterationResult> bestResult;

    //! \brief Return the null-terminated argument array owned by this state.
    //! The pointers remain valid while storage is unchanged and this state remains alive.
    [[nodiscard]] char** argv() noexcept
    {
        return argvPtrs.data();
    }
};

//! \brief Restore the arguments and completed progress for a --continue invocation.
//!
//! The user may pass `--tuningCacheFile=<path>` with `--continue` to override the default
//! cache in the current directory. The cache header supplies all other options. The
//! function scans iteration records line by line to recover the count, success count, and winner.
//! Saved route strings and iteration numbers are not compared with generated routes or indices.
//!
//! \param argc Number of arguments in the current invocation.
//! \param argv Current arguments, including the running executable at argv[0]. This executable
//! replaces the cached argv[0], which may name an older build or a relocated binary.
//! \param[out] state Owns the reconstructed arguments and the summary of completed iterations.
//! \return True on success; false with a diagnostic for invalid arguments, malformed cache data,
//! or an I/O error. Does not launch children or write the cache.
bool reconstructArgvForContinue(int32_t argc, char** argv, ContinueResumeState& state)
{
    constexpr char const* kCACHE_FLAG = "--tuningCacheFile=";
    uint64_t const flagLen = std::strlen(kCACHE_FLAG);

    // Pre-check: the only allowed args are `--continue` and `--tuningCacheFile=<path>`.
    std::string cachePath = TuningOptions{}.tuningCacheFile;
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
            sample::gLogError << "--continue accepts only an optional --tuningCacheFile=<path>; "
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
        sample::gLogError << "--continue requires a nonempty --tuningCacheFile path." << std::endl;
        return false;
    }

    auto header = readTuningCacheHeader(cachePath);
    if (!header.has_value())
    {
        sample::gLogError << "--continue: failed to read cache header from " << cachePath << std::endl;
        return false;
    }
    state.resumeFromIter = header->completedIterations;
    state.successCount = header->successCount;
    state.bestResult = std::move(header->bestResult);
    state.storage = reconstructArgvFromCacheHeader(*header, argv[0], cachePath);
    state.argvPtrs.reserve(state.storage.size() + 1);
    for (auto& s : state.storage)
    {
        state.argvPtrs.emplace_back(s.data());
    }
    state.argvPtrs.emplace_back(nullptr);
    state.argc = static_cast<int32_t>(state.storage.size());
    sample::gLogInfo << "--continue: resuming from iteration " << state.resumeFromIter.toString() << " using "
                     << state.argc << " args reconstructed from cache header." << std::endl;
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

//! \brief Print indexed routes for a search whose total count is already known.
//! Generates one route at a time without launching children or changing ctx.globalIndex.
//! \throws std::invalid_argument if the search needs measurements to determine its routes.
void emitDryRunListing(TuningContext& ctx)
{
    auto const total = ctx.count();
    if (!total)
    {
        throw std::invalid_argument("This search requires measurements and does not support --dryRun");
    }
    sample::gLogInfo << "--dryRun: " << total->toString() << " build routes would be tried:" << std::endl;
    for (BigInt i{0}; i < *total; ++i)
    {
        sample::gLogInfo << "[" << i.toString() << "]:" << ctx.getBuildRouteAtIndex(i).value() << std::endl;
    }
}

//! \brief Execution state shared by measurement, checkpointing, and final engine recovery.
//! References remain valid for the lifetime of runTuningLoop(). Cached progress initializes the
//! success count and winner; subsequent measurements update them without retaining trial history.
struct TuningState
{
    AllOptions const& options;                             //!< Parsed options for this run.
    TuningContext& ctx;                                    //!< Owns the strategy and next unfinished index.
    Logger::TestAtom const& sampleTest;                    //!< For TASK_BEGIN/END/ABORT banners.
    pid_t const ppid{};                                    //!< Parent PID, used in temporary filenames.
    std::chrono::steady_clock::time_point const startTime; //!< Start of this invocation's tuning budget.
    int32_t const argc{};                                  //!< Argument count used to construct child commands.
    char** const argv{};                                   //!< Original build arguments, restored when continuing.
    BigInt successCount{0};                                //!< Successful cached and newly measured iterations.
    double bestGpuTimeMs{std::numeric_limits<double>::infinity()}; //!< Infinity until a trial succeeds.
    std::string bestEnginePath;                                    //!< Winning iteration's engine, if it was retained.
    std::string bestRoute;                                         //!< Route used to build the winning engine.
    std::string temporaryBestEnginePath;                           //!< Winning temporary file owned by this process.
    bool failed{false}; //!< Checkpoint/cache failure requiring the run to stop.
};

//! \brief Name the engine by global iteration, using a process-local path unless all engines are saved.
//! With --saveAllEngines, return <saveEngine>.iter<N>. Otherwise include the parent PID and global
//! index so different iterations in this invocation do not overwrite each other's temporary engine.
//! Uses state.ctx.globalIndex without advancing it; does not create the file.
std::string makeIterationEnginePath(TuningState const& state)
{
    auto const index = state.ctx.globalIndex.toString();
    if (state.options.build.saveAllEngines)
    {
        return state.options.build.engine + ".iter" + index;
    }
    return "/tmp/trtexec_tuning_" + std::to_string(state.ppid) + "_iter_" + index + ".plan";
}

//! \brief Copy the winner to --saveEngine through a temporary file and rename.
//! The rename replaces the destination only after the copy completes. The iteration's source file
//! remains available for cache-based recovery if the process stops before its record is appended.
//! This protects the engine replacement; it is not a transaction spanning the engine and cache.
//! \return True when no copy is needed or the replacement succeeds. On I/O failure, return false
//! and set state.failed so the caller stops without recording the new winner.
bool checkpointBestEngine(TuningState& state)
{
    auto const& destination = state.options.build.engine;
    if (destination.empty() || destination == state.bestEnginePath)
    {
        return true;
    }
    std::string const temporary = destination + ".tuning." + std::to_string(state.ppid) + ".tmp";
    std::ifstream src(state.bestEnginePath, std::ios::binary);
    std::ofstream dst(temporary, std::ios::binary);
    if (!src || !dst || !(dst << src.rdbuf()))
    {
        sample::gLogError << "Could not checkpoint best engine to " << destination << std::endl;
        state.failed = true;
        return false;
    }
    dst.close();
    if (!dst || std::rename(temporary.c_str(), destination.c_str()) != 0)
    {
        sample::gLogError << "Could not replace best engine at " << destination << std::endl;
        state.failed = true;
        return false;
    }
    // Keep the winning iteration's file: the cache points to that exact engine,
    // even if interruption occurs between replacing saveEngine and appending a row.
    return true;
}

//! \brief Run and checkpoint one new iteration, then append its cache row.
//!
//! Uses state.ctx.globalIndex to name and report the trial, leaving advancement to runTuningLoop().
//! A failed build, accuracy check, or unusable timing is recorded as a failed trial. Successful
//! measurements compete with the restored winner using mean GPU time; equal times keep that winner.
//! A better engine is checkpointed before its row is appended. The previous temporary winner owned
//! by this process is deleted only after the replacement measurement has been recorded successfully.
//!
//! \param[in,out] state Child arguments, execution index, winner, and persistence-error status.
//! \param route One concrete build route to pass to the child.
//! \throws std::overflow_error before launching a child if the index cannot fit the cache format.
//! Checkpoint or cache-write errors set state.failed; an individual trial failure does not.
void measureIteration(TuningState& state, std::string const& route)
{
    auto const& index = state.ctx.globalIndex;
    if (!state.options.tuning.tuningCacheFile.empty() && index > BigInt{std::numeric_limits<uint64_t>::max()})
    {
        throw std::overflow_error("Iteration index exceeds the tuning cache integer range");
    }
    auto const previousTemporaryBest = state.temporaryBestEnginePath;
    std::string const enginePath = makeIterationEnginePath(state);
    std::string const jsonPath
        = "/tmp/trtexec_tuning_" + std::to_string(state.ppid) + "_iter" + index.toString() + ".json";
    sample::gLogger.reportTaskBegin(state.sampleTest, index.toString(), route);
    sample::gLogInfo << "Tuning iteration [" << index.toString() << "]: " << route << std::endl;
    auto result = runChildForOneRoute(state.argc, state.argv, route, enginePath, jsonPath);
    bool const succeeded = !result.crashed && result.exitCode == EXIT_SUCCESS && !result.accuracyFailed
        && std::isfinite(result.gpuTimeMs) && result.gpuTimeMs > 0.0;
    if (succeeded)
    {
        ++state.successCount;
        if (result.gpuTimeMs < state.bestGpuTimeMs)
        {
            state.bestGpuTimeMs = result.gpuTimeMs;
            state.bestEnginePath = enginePath;
            if (!state.options.build.saveAllEngines)
            {
                state.temporaryBestEnginePath = enginePath;
            }
            state.bestRoute = route;
            checkpointBestEngine(state);
        }
        sample::gLogger.reportTaskEnd(state.sampleTest, index.toString(), route);
    }
    else
    {
        if (result.errorMessage.empty())
        {
            result.errorMessage = "Iteration failed accuracy validation or produced no finite positive GPU time";
        }
        sample::gLogWarning << "Iteration [" << index.toString() << "] failed: " << result.errorMessage << std::endl;
        sample::gLogger.reportTaskAbort(state.sampleTest, index.toString(), route);
    }
    if (!state.failed && !state.options.tuning.tuningCacheFile.empty())
    {
        state.failed = !writeTuningCacheIteration(state.options.tuning.tuningCacheFile, index.toUint64(), route,
            result.crashed, result.errorMessage, result.accuracyLossValues, result.gpuTimeMs,
            succeeded && !state.options.build.engine.empty() ? enginePath : std::string{});
    }
    // Delete only files created by this process, after the replacement is recorded.
    if (!state.failed && !previousTemporaryBest.empty() && previousTemporaryBest != state.temporaryBestEnginePath)
    {
        std::remove(previousTemporaryBest.c_str());
    }
    std::remove(jsonPath.c_str());
    if (!state.options.build.saveAllEngines
        && (enginePath != state.bestEnginePath || state.options.build.engine.empty()))
    {
        std::remove(enginePath.c_str());
    }
}

//! \brief Whether the wall-clock tuning budget has expired.
//! A nonpositive timeout disables the budget. The loop checks between trials, allowing a running
//! child to finish and its measurement to be recorded before stopping.
bool tuningTimedOut(TuningState const& state)
{
    auto const elapsed
        = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - state.startTime).count();
    if (state.options.tuning.timeout > 0 && elapsed >= state.options.tuning.timeout)
    {
        sample::gLogInfo << "Tuning timeout reached (" << state.options.tuning.timeout << "s); stopping early."
                         << std::endl;
        return true;
    }
    return false;
}

//! \brief Recover a missing saved engine for the best cached route without redoing the search.
//! Copy the retained winning engine when available; otherwise rebuild bestRoute in one child and
//! require a successful build and accuracy check. Recovery neither appends a trial nor advances
//! ctx.globalIndex. With no --saveEngine destination, there is no artifact to recover.
//! \return Whether the winning engine is available at the requested destination.
bool recoverBestEngine(TuningState& state)
{
    if (state.options.build.engine.empty())
    {
        return true;
    }
    if (std::ifstream(state.bestEnginePath, std::ios::binary).good())
    {
        return checkpointBestEngine(state);
    }
    sample::gLogInfo << "Rebuilding the best cached route because its saved engine is unavailable." << std::endl;
    auto const enginePath = "/tmp/trtexec_tuning_" + std::to_string(state.ppid) + "_best_recovery.plan";
    auto const jsonPath = enginePath + ".json";
    auto const result = runChildForOneRoute(state.argc, state.argv, state.bestRoute, enginePath, jsonPath);
    std::remove(jsonPath.c_str());
    if (result.crashed || result.exitCode != EXIT_SUCCESS || result.accuracyFailed)
    {
        sample::gLogError << "Could not rebuild the best cached engine: " << result.errorMessage << std::endl;
        return false;
    }
    state.bestEnginePath = enginePath;
    return checkpointBestEngine(state);
}

//! \brief Emit the best result, including results restored from an already completed cache.
//! Recover the winning artifact before reporting success. Fail if persistence failed, no measurement
//! succeeded, or recovery failed. If the strategy's final count is still unknown after a timeout,
//! report the completed count instead of treating the initial scan's count as the final total.
int32_t finalizeBestEngine(TuningState& state)
{
    if (state.failed || state.successCount.isZero() || !recoverBestEngine(state))
    {
        sample::gLogError << "Tuning did not produce an available successful result." << std::endl;
        return sample::gLogger.reportFail(state.sampleTest);
    }
    sample::gLogInfo << "Best iteration: " << state.bestRoute << " (gpu_time_ms=" << state.bestGpuTimeMs << ")"
                     << std::endl;
    auto const totalCount = state.ctx.count().value_or(state.ctx.globalIndex);
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

    // --continue rebuilds argv from the cache header and accepts only a cache-path override.
    // The header stores resolved file paths and the tuning expression, without expanding its routes.
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

    // Parse options against the effective argv (cache-reconstructed for --continue, raw otherwise).
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

    // The reconstructed argv has no --continue, so parseArgs left continueFromCache false.
    // Re-assert it now so the cache-header writer below preserves existing rows.
    if (effectiveArgv != argv)
    {
        options.tuning.continueFromCache = true;
    }

    if (!validateTuningOptions(options))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // Parse the expression and select its enumeration strategy; do not expand the route space.
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

    // Fresh tuning runs write a cache header; resumed runs preserve the existing header and rows.
    if (!options.tuning.tuningCacheFile.empty() && !options.tuning.continueFromCache)
    {
        if (!writeTuningCacheHeader(
                options.tuning.tuningCacheFile, options, argc, argv, ctx.tunerVersion, ctx.defaultBuildRoute))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }

    // Children use the original build/inference arguments reconstructed from the cache.
    TuningState state{
        options, ctx, sampleTest, getpid(), std::chrono::steady_clock::now(), effectiveArgc, effectiveArgv};
    ctx.globalIndex = resume.resumeFromIter;
    state.successCount = resume.successCount;
    // Preserve earlier improvements, including when the cache already covers the whole search.
    if (resume.bestResult)
    {
        state.bestGpuTimeMs = resume.bestResult->gpuTimeMs;
        state.bestRoute = resume.bestResult->buildRoute;
        state.bestEnginePath = resume.bestResult->enginePath;
    }
    try
    {
        // Route lookup determines whether more work exists; adaptive counts may still be unknown.
        // Start directly at the cache's record count and advance only after the trial is recorded.
        for (auto route = ctx.getBuildRouteAtIndex(ctx.globalIndex); route && !tuningTimedOut(state);
             route = ctx.getBuildRouteAtIndex(ctx.globalIndex))
        {
            measureIteration(state, *route);
            if (state.failed)
            {
                return sample::gLogger.reportFail(sampleTest);
            }
            ++ctx.globalIndex;
        }
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Tuning failed: " << e.what() << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    return finalizeBestEngine(state);
}

#endif // POSIX && ENABLE_FEATURE_GLOBAL_PERF_TUNER
