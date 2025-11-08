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

//! \file sampleEditableTimingCache.cpp
//!
//! \brief This file contains the implementation of the editable
//! timing cache sample.
//!
//! It builds two engines from a simple network. The second build
//! reuses a timing cache generated during the first build but made
//! some modifications, specifically assigning a different tactic to a
//! layer.
//!
//! The goal of this sample is to show how to build an engine with
//! desired tactics by modifying the timing cache.
//!
//! It can be run with the following command line:
//! Command: ./sample_editable_timing_cache

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cstdlib> // for strtoull

#define DEFINE_TRT_ENTRYPOINTS 1
#include "NvInfer.h"
#include "common.h"
#include "logger.h"

using namespace nvinfer1;

using samplesCommon::SampleUniquePtr;

namespace
{

std::string const kSAMPLE_NAME = "TensorRT.sample_editable_timing_cache";

using Name = std::string;

//! \brief A hash string which starts with `0x` followed by some
//! hexadecimal digits.
using Hash = std::string;

//! \brief A pair that denotes a tactic of some op.
struct Tactic
{
    Hash hash;   //!< Hash string which uniquely identifies the tactic.
    Name kernel; //!< Name of the kernel used by the tactic.
};

//! \brief A structure recording the profiling result of an op.
struct ProfilingRecord
{
    Name op; //!< Name of the op.

    Hash key; //!< Hash string which uniquely identifies the op. Its' used
              //!< as a key in Timing Cache.

    std::vector<Tactic> tactics; //!< Available tactics.

    Hash selected; //!< Hash string which uniquely identifies the
                   //!< tactic finally used by the op.
};

//! \brief A mapping from the name of an op to its profiling result.
using ProfilingTable = std::unordered_map<Name, ProfilingRecord>;

void printProfilingTable(ProfilingTable const& table)
{
    sample::gLogInfo << "Profiling table:\n";

    for (auto const& [op, record] : table)
    {
        sample::gLogInfo << "\top: " << op << "\n";
        sample::gLogInfo << "\t\tkey: " << record.key << "\n";
        sample::gLogInfo << "\t\tselected: " << record.selected << "\n";
        sample::gLogInfo << "\t\tavailable tactics:\n";

        for (auto const& [hash, kernel] : record.tactics)
        {
            sample::gLogInfo << "\t\t\t" << hash << " " << kernel << "\n";
        }

        sample::gLogInfo << "\n\n";
    }
}

// The implementation of std::regex is not entirely reliable on some
// platforms, so we use basic string interfaces for pattern matching.
namespace patterns
{

struct OpKey
{
    Name op;
    Hash key;
};

//! Searches \p text for a sub string like `Autotuning op matMul1(key: 0x1814870c44ff0f8574df6e3dda04cbd7)`
//! where in this example the field `op` of the returned `OpKey` would be assigned `matMul1`
//! and the field `key` would be assigned `0x181487...`.
[[nodiscard]] std::optional<OpKey> matchOpKey(char const* const text)
{
    char const* const kPREFIX = "Autotuning op ";

    char const* const substr = std::strstr(text, kPREFIX);
    if (!substr)
    {
        return std::nullopt;
    }

    char op[128 + 1]{}; //< Plus one for the null terminator.
    char key[128 + 1]{}; //< Plus one for the null terminator.

    int numReceived = std::sscanf(substr + std::strlen(kPREFIX), "%128[^(](key: %128[^)])", op, key);
    if (numReceived != 2)
    {
        return std::nullopt;
    }

    return OpKey{Name(op), Hash(key)};
}

[[nodiscard]] bool matchTacticHeader(std::string_view text)
{
    return text.find("tactic_id, cost(in ms), cost/fastest_cost") != text.npos;
}

struct TacticKernel
{
    Hash tactic;
    Name kernel;
};

//! Searches \p text for a sub string like `4, 0.00520, 1.00, 0.883, sm86_xmma_gemm, 0x533a71cee0d0e,`
//! where in this example the field `tactic` of the returned `TacticKernel` would be assigned `0x533a71cee0d0e`
//! and the field `kernel` would be assigned `sm86_xmma_gemm`.
[[nodiscard]] std::optional<TacticKernel> matchTacticKernel(char const* const text)
{
    char const* const kDIGITS = "0123456789";

    char const* const substr = std::strpbrk(text, kDIGITS);
    if (!substr)
    {
        return std::nullopt;
    }

    char kernel[128 + 1]{}; //< Plus one for the null terminator.
    char tactic[128 + 1]{}; //< Plus one for the null terminator.

    int numReceived = std::sscanf(substr, "%*d, %*f, %*f, %*f, %128[^,], %128[^,]", kernel, tactic);
    if (numReceived != 2)
    {
        return std::nullopt;
    }

    return TacticKernel{Hash(tactic), Name(kernel)};
}

//! Searches \p text for a sub string like `The selected tactic is (tactic hash, cost(in ms)):0x533a71cee0d0e,
//! 0.0050048` where in this example the returned `Hash` would be `0x533a71cee0d0e`.
[[nodiscard]] std::optional<Hash> matchSelection(char const* const text)
{
    char const* const kPREFIX = "(tactic hash, cost(in ms)):";

    char const* const substr = std::strstr(text, kPREFIX);
    if (!substr)
    {
        return std::nullopt;
    }

    char tactic[128 + 1]{}; //< Plus one for the null terminator.

    int numReceived = sscanf(substr + std::strlen(kPREFIX), "%128[^,]", tactic);
    if (numReceived != 1)
    {
        return std::nullopt;
    }

    return Hash(tactic);
}

struct LayerKernel
{
    Name layer;
    Name kernel;
};

//! Searches \p text for a sub string like `Name: matMul2_myl0_3,
//! LayerType: ...., TacticName: sm80_xmma_gemm, StreamId: 0` where in
//! this example the field `layer` of the returned `LayerKernel` would be `matMul2_myl0_3`
//! and the field `kernel` would be `sm80_xmma_gemm`.
[[nodiscard]] std::optional<LayerKernel> matchLayerKernel(char const* const text)
{
    char const* const kLAYER_PREFIX = "Name: ";

    char const* const layerSubstr = std::strstr(text, kLAYER_PREFIX);
    if (!layerSubstr)
    {
        return std::nullopt;
    }

    char layer[128 + 1]{}; //< Plus one for the null terminator.

    int numReceived = std::sscanf(layerSubstr + std::strlen(kLAYER_PREFIX), "%128[^,]", layer);
    if (numReceived != 1)
    {
        return std::nullopt;
    }

    char const* const kKERNEL_PREFIX = "TacticName: ";

    char const* const kernelSubstr = std::strstr(text, kKERNEL_PREFIX);
    if (!kernelSubstr)
    {
        return std::nullopt;
    }

    char kernel[128 + 1]{}; //< Plus one for the null terminator.

    numReceived = std::sscanf(kernelSubstr + std::strlen(kKERNEL_PREFIX), "%128[^,]", kernel);
    if (numReceived != 1)
    {
        return std::nullopt;
    }

    return LayerKernel{Name(layer), Name(kernel)};
}

} // namespace patterns

//! \brief `ProfilingLogger` is a decorator of `ILogger`. It
//! dispatches the message to the decorated logger and extracts
//! profiling information from the message.
//!
//! \details This class overrides the method `log` of class `ILogger`
//! to analyze each line of the logs. Since the profiling information
//! are spread across different lines, it builds a simple state
//! machine to recognize and capture this information.
class ProfilingLogger : public nvinfer1::ILogger
{
private:
    enum class State
    {
        kEXPECT_KEY,
        kEXPECT_TACTIC_HEADER,
        kEXPECT_TACTIC,
        kEXPECT_SELECTION,
    };

public:
    ProfilingLogger(ILogger& logger)
        : mLogger(logger)
        , mState(State::kEXPECT_KEY)
    {
    }

    void log(Severity severity, AsciiChar const* msg) noexcept override
    {
        mLogger.log(severity, msg);

        bool resolved = false;

        while (!resolved)
        {
            resolved = true;

            switch (mState)
            {
            case State::kEXPECT_KEY:
            {
                if (auto optOpKey = patterns::matchOpKey(msg))
                {
                    mRecord.op = std::move(optOpKey->op);
                    mRecord.key = std::move(optOpKey->key);
                    mState = State::kEXPECT_TACTIC_HEADER;
                }

                break;
            }

            case State::kEXPECT_TACTIC_HEADER:
            {
                if (patterns::matchTacticHeader(msg))
                {
                    mState = State::kEXPECT_TACTIC;
                }

                break;
            }

            case State::kEXPECT_TACTIC:
            {
                if (auto optTacticKernel = patterns::matchTacticKernel(msg))
                {
                    mRecord.tactics.push_back(
                        Tactic{std::move(optTacticKernel->tactic), std::move(optTacticKernel->kernel)});
                }
                else
                {
                    mState = State::kEXPECT_SELECTION;
                    resolved = false;
                }

                break;
            }

            case State::kEXPECT_SELECTION:
            {
                if (auto optTactic = patterns::matchSelection(msg))
                {
                    mRecord.selected = std::move(*optTactic);
                    mTable[mRecord.op] = mRecord;
                    mRecord = ProfilingRecord{};
                    mState = State::kEXPECT_KEY;
                }

                break;
            }
            }
        }
    }

    //! \brief Get the profiling result and reset the state machine.
    ProfilingTable fetchTable()
    {
        mState = State::kEXPECT_KEY;
        mRecord = ProfilingRecord{};
        return std::exchange(mTable, ProfilingTable{});
    }

private:
    ILogger& mLogger;
    State mState;
    ProfilingTable mTable;
    ProfilingRecord mRecord;
};

//! \brief Build a simple graph with three nodes: MatMul -> SoftMax ->
//! MatMul.
//!
//! \details The two MatMuls are identical in all attributes
//! except for their names.
//!
//! \return a pointer to the first MatMul.
ILayer const* buildGraph(INetworkDefinition* network)
{
    auto input = network->addInput("input", DataType::kFLOAT, Dims2{128, 128});
    auto weight1 = network->addInput("weight1", DataType::kFLOAT, Dims2{128, 128});
    auto weight2 = network->addInput("weight2", DataType::kFLOAT, Dims2{128, 128});
    auto matMul1 = network->addMatrixMultiply(*input, MatrixOperation::kNONE, *weight1, MatrixOperation::kNONE);
    auto softmax = network->addSoftMax(*matMul1->getOutput(0));
    auto matMul2
        = network->addMatrixMultiply(*softmax->getOutput(0), MatrixOperation::kNONE, *weight2, MatrixOperation::kNONE);

    network->markOutput(*matMul2->getOutput(0));

    matMul1->setName("matMul1");
    softmax->setName("softmax");
    matMul2->setName("matMul2");

    return matMul1;
}

//! \brief Find a tactic different from the selected one in the
//! candidate set.
std::optional<Tactic> findDifferentTactic(ProfilingRecord const& record)
{
    auto it = std::find_if(record.tactics.cbegin(), record.tactics.cend(),
        [&](auto const& entry) { return entry.hash != record.selected; });

    return it == record.tactics.end() ? std::nullopt : std::make_optional(*it);
}

constexpr int64_t kNUM_PREFIX_CHARS = std::char_traits<char>::length("0x");
constexpr int64_t kCHARS_PER_BYTE = 2;

constexpr int64_t kBYTES_PER_KEY = 16;
constexpr int64_t kTOTAL_CHARS_PER_KEY = kNUM_PREFIX_CHARS + kBYTES_PER_KEY * kCHARS_PER_BYTE;

//! \brief Parse a TimingCacheKey from its text form.
//! \return false if an error occurs.
bool parseKey(std::string_view text, TimingCacheKey* key)
{
    CHECK_RETURN_W_MSG(static_cast<int64_t>(text.size()) == kTOTAL_CHARS_PER_KEY, false, "Unexpected length of key");

    for (int64_t i = 0, offset = kNUM_PREFIX_CHARS; i < kBYTES_PER_KEY; ++i, offset += kCHARS_PER_BYTE)
    {
        CHECK_RETURN(1 == sscanf(text.data() + offset, "%2" SCNx8, &key->data[i]), false);
    }

    return true;
}

constexpr int64_t kBYTES_PER_TACTIC = 8;
constexpr int64_t kTOTAL_CAHRS_PER_TACTIC = kNUM_PREFIX_CHARS + kBYTES_PER_TACTIC * kCHARS_PER_BYTE;

//! \brief Parse a tactic hash from its text form.
//! \return false if an error occurs.
bool parseTactic(std::string_view text, size_t* hash)
{
    CHECK_RETURN_W_MSG(
        static_cast<int64_t>(text.size()) <= kTOTAL_CAHRS_PER_TACTIC, false, "Unexpected length of tactic");

    char const* start = text.data() + kNUM_PREFIX_CHARS;
    char* end = nullptr;
    *hash = std::strtoull(start, &end, 16);
    CHECK_RETURN_W_MSG(end == text.data() + text.size(), false, "Found junk in the text.");

    return true;
}

//! \brief Set a new tactic for some key in the timing cache.
//! \return false if an error occurs.
bool setTactic(ITimingCache* cache, std::string_view keyText, std::string_view tacticText)
{
    TimingCacheKey key;
    CHECK_RETURN_W_MSG(parseKey(keyText, &key), false, "Failed to parse the key.");

    TimingCacheValue value;
    CHECK_RETURN_W_MSG(parseTactic(tacticText, &value.tacticHash), false, "Failed to parse the tactic hash");

    value.timingMSec = 1.0F;
    CHECK_RETURN_W_MSG(cache->update(key, value), false, "Failed to update the timing cache.");
    return true;
}

//! \brief A pair which denotes a layer in the engine.
struct LayerKernel
{
    Name layer;  //!< Name of the layer.
    Name kernel; //!< Name of the kernel used by the layer.
};

//! \brief Extract the name of each layer in the engine, along with
//! the kernel used by it.
void extractLayerKernels(ICudaEngine const* engine, std::vector<LayerKernel>& table)
{
    SampleUniquePtr<IEngineInspector> inspector{engine->createEngineInspector()};

    int32_t numLayers = engine->getNbLayers();

    for (int32_t i = 0; i < numLayers; ++i)
    {
        char const* line = inspector->getLayerInformation(i, LayerInformationFormat::kONELINE);

        if (auto optLayerKernel = patterns::matchLayerKernel(line))
        {
            table.push_back({std::move(optLayerKernel->layer), std::move(optLayerKernel->kernel)});
        }
    }
}

void printLayerKernels(std::vector<LayerKernel> const& table)
{
    for (size_t i = 0; i < table.size(); ++i)
    {
        auto const& [layer, kernel] = table[i];
        sample::gLogInfo << "#" << i << ": " << std::setw(30) << std::setfill(' ') << std::left << layer << " =uses=> "
                         << kernel << "\n";
    }
}

bool isPrefixOf(std::string_view shorter, std::string_view longer)
{
    return shorter.size() <= longer.size() && std::equal(shorter.begin(), shorter.end(), longer.begin());
}

//! \brief Find the layer derived from the op.
//!
//! \details In this sample, the name of a layer derived from a MatMul
//! op is prefixed with the op's name.
std::optional<LayerKernel> findLayer(std::vector<LayerKernel> const& table, std::string_view op)
{
    auto it = std::find_if(
        table.begin(), table.end(), [op](LayerKernel const& entry) { return isPrefixOf(op, entry.layer); });

    return it == table.end() ? std::nullopt : std::make_optional(*it);
}

} // namespace

#define FAIL_IF_NOT(status, errMsg)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << (errMsg) << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line "       \
                              << __LINE__ << std::endl;                                                                \
            return sample::gLogger.reportFail(sampleTest);                                                             \
        }                                                                                                              \
    } while (0)

int32_t main(int32_t argc, char* argv[])
{
    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    try
    {
        // Set the logging level to kVERBOSE to see the profiling
        // information.
        sample::gLogger.setReportableSeverity(ILogger::Severity::kVERBOSE);

        ProfilingLogger profilingLogger(sample::gLogger.getTRTLogger());

        SampleUniquePtr<IBuilder> builder{createInferBuilder(profilingLogger)};
        FAIL_IF_NOT(builder, "Failed to create inference builder.");

        SampleUniquePtr<INetworkDefinition> network{builder->createNetworkV2(0)};
        FAIL_IF_NOT(network, "Failed to create network.");

        ILayer const* matMul1 = buildGraph(network.get());
        std::string const opName = matMul1->getName();

        SampleUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
        FAIL_IF_NOT(config, "Failed to create builder config.");

        // Tell the builder to save the name of tactic used by each layer
        // in the engine.
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);

        // Enable the editable timing cache. In editable mode, the logs
        // will contain profiling results of all layers. Besides, each
        // layer will have its own tactics, which means that changes in
        // one layer will not affect others.
        config->setFlag(BuilderFlag::kEDITABLE_TIMING_CACHE);

        // Provide the builder with an empty timing cache.
        SampleUniquePtr<ITimingCache> timingCache{config->createTimingCache(nullptr, 0)};
        FAIL_IF_NOT(timingCache, "Failed to set timing cache.");

        FAIL_IF_NOT(config->setTimingCache(*timingCache, true), "Failed to set timing cache.");

        // Build the first engine.
        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        FAIL_IF_NOT(plan, "Failed to build serialized engine.");

        SampleUniquePtr<IRuntime> runtime{createInferRuntime(profilingLogger)};
        FAIL_IF_NOT(runtime, "Failed to create the runtime.");

        SampleUniquePtr<ICudaEngine> engine{runtime->deserializeCudaEngine(plan->data(), plan->size())};
        FAIL_IF_NOT(engine, "Failed to deserialize the engine.");

        // Extract layers' information of the first engine.
        std::vector<LayerKernel> layerKernels;
        extractLayerKernels(engine.get(), layerKernels);

        std::optional<LayerKernel> matMulLayer = findLayer(layerKernels, opName);
        FAIL_IF_NOT(matMulLayer.has_value(), "Cannot find the layer derived from the first MatMul node.");

        // Extract profiling results from the logs.
        ProfilingTable table = profilingLogger.fetchTable();

        // Find a different tactic for the first MatMul.
        ProfilingRecord const& opRecord = table.at(opName);

        std::optional<Tactic> newTactic = findDifferentTactic(opRecord);
        FAIL_IF_NOT(newTactic.has_value(), "No other tactics.");

        // Put the new tactic in the cache.
        CHECK_RETURN(setTactic(timingCache.get(), opRecord.key, newTactic->hash), EXIT_FAILURE);

        // Build the second engine, with the modified timing cache.
        SampleUniquePtr<IHostMemory> newPlan{builder->buildSerializedNetwork(*network, *config)};
        FAIL_IF_NOT(newPlan, "Failed to build the engine again.");

        SampleUniquePtr<ICudaEngine> newEngine{runtime->deserializeCudaEngine(newPlan->data(), newPlan->size())};
        FAIL_IF_NOT(newEngine, "Failed to deserialize the engine again.");

        // Extract layers' information of the second engine.
        std::vector<LayerKernel> newLayerKernels;
        extractLayerKernels(newEngine.get(), newLayerKernels);

        std::optional<LayerKernel> newMatMulLayer = findLayer(newLayerKernels, opName);

        FAIL_IF_NOT(newMatMulLayer.has_value(), "Cannot find the layer derived from the first MatMul node.");

        FAIL_IF_NOT(newMatMulLayer->kernel == newTactic->kernel, "The layer didn't use the assigned new kernel.");

        sample::gLogInfo << "\n";

        sample::gLogInfo << "Layers of the first engine:\n";
        printLayerKernels(layerKernels);

        sample::gLogInfo << "\n";

        printProfilingTable(table);

        sample::gLogInfo << "Originally, layer `" << matMulLayer->layer << "` used kernel `" << matMulLayer->kernel
                         << "`.\n";
        sample::gLogInfo << "Now, it should use the new kernel `" << newTactic->kernel << ".`\n";
        sample::gLogInfo << "\n";

        sample::gLogInfo << "Layers of the second engine:\n";
        printLayerKernels(newLayerKernels);

        sample::gLogInfo << "\n";

        return sample::gLogger.reportPass(sampleTest);
    }
    catch (std::exception const& err)
    {
        sample::gLogError << "Exception: " << err.what() << "\n";
        return sample::gLogger.reportFail(sampleTest);
    }
}
