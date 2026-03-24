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

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"

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

// MD code start
#include <nccl.h>
// MD code end

using namespace std;

//! Checks NCCL return codes and asserts on failure since NCCL errors are unrecoverable communication failures.
#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t const r = (cmd);                                                                                  \
        ASSERT(r == ncclSuccess);                                                                                      \
    } while (0)

#define CHECK_CUDA(status)                                                                                             \
    if (status != cudaSuccess)                                                                                         \
    {                                                                                                                  \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));                                               \
        exit(EXIT_FAILURE);                                                                                            \
    }

namespace
{

using LibraryPtr = std::unique_ptr<DynamicLibrary>;

#if !TRT_STATIC
#if defined(_WIN32)
std::string const kNVINFER_PLUGIN_LIBNAME
    = std::string{"nvinfer_plugin_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVINFER_LIBNAME = std::string{"nvinfer_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVONNXPARSER_LIBNAME
    = std::string{"nvonnxparser_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVINFER_LEAN_LIBNAME
    = std::string{"nvinfer_lean_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVINFER_DISPATCH_LIBNAME
    = std::string{"nvinfer_dispatch_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
#else
std::string const kNVINFER_PLUGIN_LIBNAME = std::string{"libnvinfer_plugin.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LIBNAME = std::string{"libnvinfer.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVONNXPARSER_LIBNAME = std::string{"libnvonnxparser.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LEAN_LIBNAME = std::string{"libnvinfer_lean.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_DISPATCH_LIBNAME
    = std::string{"libnvinfer_dispatch.so."} + std::to_string(NV_TENSORRT_MAJOR);
#endif
#endif // !TRT_STATIC
std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};
std::function<void*(void*, void*, int32_t)> pCreateInferRefitterInternal{};
std::function<void*(void*, int32_t)> pCreateInferBuilderInternal{};
std::function<void*(void*, void*, int)> pCreateNvOnnxParserInternal{};
std::function<void*(void*, void*, int)> pCreateNvOnnxRefitterInternal{};

//! Track runtime used for the execution of trtexec.
//! Must be tracked as a global variable due to how library init functions APIs are organized.
RuntimeMode gUseRuntime = RuntimeMode::kFULL;

#if !TRT_STATIC
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
    return initLibrary(libnvinferPtr, sample::getRuntimeLibraryName(gUseRuntime), fetchPtrs);
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
        pCreateNvOnnxRefitterInternal = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParserRefitter_INTERNAL");
    };
    return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
    pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
    pCreateNvOnnxRefitterInternal = createNvOnnxParserRefitter_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

[[nodiscard]] std::string toString(CollectiveOperation op)
{
    switch (op)
    {
    case CollectiveOperation::kALL_REDUCE: return "ALL_REDUCE";
    case CollectiveOperation::kALL_GATHER: return "ALL_GATHER";
    case CollectiveOperation::kBROADCAST: return "BROADCAST";
    case CollectiveOperation::kREDUCE: return "REDUCE";
    case CollectiveOperation::kREDUCE_SCATTER: return "REDUCE_SCATTER";
    }
    throw std::runtime_error("Unknown CollectiveOperation");
}

[[nodiscard]] bool icharEquals(char a, char b)
{
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
}

//! Case-insensitive string equality:
[[nodiscard]] bool iequals(std::string_view lhs, std::string_view rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), icharEquals);
}

//! Parse operation string to CollectiveOperation enum
[[nodiscard]] std::optional<CollectiveOperation> parseCollectiveOp(std::string_view opStr)
{
    if (iequals(opStr, "all_reduce"))
    {
        return CollectiveOperation::kALL_REDUCE;
    }
    if (iequals(opStr, "all_gather"))
    {
        return CollectiveOperation::kALL_GATHER;
    }
    if (iequals(opStr, "broadcast"))
    {
        return CollectiveOperation::kBROADCAST;
    }
    if (iequals(opStr, "reduce"))
    {
        return CollectiveOperation::kREDUCE;
    }
    if (iequals(opStr, "reduce_scatter"))
    {
        return CollectiveOperation::kREDUCE_SCATTER;
    }
    return std::nullopt;
}

void printUsage(char const* programName)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  Set TRT_MY_RANK, TRT_WORLD_SIZE, and TRT_NCCL_ID_FILE, then run " << programName
              << " --op <operation>" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --op <operation>  Specify the collective operation to test (required)." << std::endl;
    std::cout << "                    Valid operations: all_reduce, all_gather, broadcast, reduce, reduce_scatter"
              << std::endl;
    std::cout << "  --help, -h        Show this help message." << std::endl;
    std::cout << std::endl;
    std::cout << "Environment Variables (required):" << std::endl;
    std::cout << "  TRT_MY_RANK       The rank of this process (0 to WORLD_SIZE-1)." << std::endl;
    std::cout << "  TRT_WORLD_SIZE    The total number of processes." << std::endl;
    std::cout << "  TRT_NCCL_ID_FILE  Path to a shared file for NCCL ID coordination." << std::endl;
    std::cout << "                    Rank 0 writes the NCCL ID to this file, other ranks read from it." << std::endl;
    std::cout << "                    The file should be empty or non-existent before starting." << std::endl;
    std::cout << std::endl;
    std::cout << "Example commands:" << std::endl;
    std::cout << "  SLURM:" << std::endl;
    std::cout << "    srun --ntasks=2 bash -lc 'export TRT_MY_RANK=$SLURM_PROCID; \\" << std::endl;
    std::cout << "        export TRT_WORLD_SIZE=$SLURM_NTASKS; \\" << std::endl;
    std::cout << "        export TRT_NCCL_ID_FILE=/tmp/nccl_id.txt; \\" << std::endl;
    std::cout << "        " << programName << " --op all_reduce'" << std::endl;
    std::cout << std::endl;
    std::cout << "  Open MPI:" << std::endl;
    std::cout << "    mpirun -np 2 bash -lc 'export TRT_MY_RANK=$OMPI_COMM_WORLD_RANK; \\" << std::endl;
    std::cout << "        export TRT_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE; \\" << std::endl;
    std::cout << "        export TRT_NCCL_ID_FILE=/tmp/nccl_id.txt; \\" << std::endl;
    std::cout << "        " << programName << " --op all_reduce'" << std::endl;
}

//! Get rank from TRT_MY_RANK environment variable.
//! Users should set this variable via a launcher wrapper script.
[[nodiscard]] int32_t getRankFromEnv()
{
    char const* rankStr = std::getenv("TRT_MY_RANK");
    if (!rankStr)
    {
        sample::gLogError << "FATAL: TRT_MY_RANK environment variable is not set!" << std::endl;
        sample::gLogError << "Please set TRT_MY_RANK to the rank of this process (0 to WORLD_SIZE-1)." << std::endl;
        sample::gLogError << "Run with --help for example commands." << std::endl;
        ASSERT(false && "TRT_MY_RANK environment variable must be set");
    }
    return std::stoi(rankStr);
}

//! Get world size from TRT_WORLD_SIZE environment variable.
//! Users should set this variable via a launcher wrapper script.
[[nodiscard]] int32_t getWorldSizeFromEnv()
{
    char const* worldSizeStr = std::getenv("TRT_WORLD_SIZE");
    if (!worldSizeStr)
    {
        sample::gLogError << "FATAL: TRT_WORLD_SIZE environment variable is not set!" << std::endl;
        sample::gLogError << "Please set TRT_WORLD_SIZE to the total number of processes." << std::endl;
        sample::gLogError << "Run with --help for example commands." << std::endl;
        ASSERT(false && "TRT_WORLD_SIZE environment variable must be set");
    }
    return std::stoi(worldSizeStr);
}

//! Convert a hex character to its integer value
[[nodiscard]] int32_t hexCharToInt(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    return -1;
}

//! Convert NCCL unique ID bytes to hex string
[[nodiscard]] std::string ncclIdToHex(ncclUniqueId const& id)
{
    constexpr char kHEX_CHARS[] = "0123456789abcdef";
    std::string hexStr;
    hexStr.reserve(sizeof(ncclUniqueId) * 2);
    for (size_t i = 0; i < sizeof(ncclUniqueId); ++i)
    {
        auto const byte = static_cast<uint8_t>(id.internal[i]);
        hexStr += kHEX_CHARS[byte >> 4];
        hexStr += kHEX_CHARS[byte & 0x0F];
    }
    return hexStr;
}

//! Parse hex string to NCCL unique ID
[[nodiscard]] ncclUniqueId hexToNcclId(std::string const& hexStr)
{
    constexpr size_t kNCCL_UNIQUE_ID_BYTES = sizeof(ncclUniqueId);
    constexpr size_t kEXPECTED_HEX_LEN = kNCCL_UNIQUE_ID_BYTES * 2;

    if (hexStr.length() != kEXPECTED_HEX_LEN)
    {
        throw std::runtime_error("NCCL ID hex string has invalid length: " + std::to_string(hexStr.length())
            + " (expected " + std::to_string(kEXPECTED_HEX_LEN) + ")");
    }

    ncclUniqueId id;
    for (size_t i = 0; i < kNCCL_UNIQUE_ID_BYTES; ++i)
    {
        int32_t const high = hexCharToInt(hexStr[2 * i]);
        int32_t const low = hexCharToInt(hexStr[2 * i + 1]);
        if (high < 0 || low < 0)
        {
            throw std::runtime_error(
                "NCCL ID hex string contains invalid character at position " + std::to_string(2 * i));
        }
        id.internal[i] = static_cast<char>((high << 4) | low);
    }
    return id;
}

//! Get the NCCL ID file path from TRT_NCCL_ID_FILE environment variable.
[[nodiscard]] std::string getNcclIdFilePath()
{
    char const* filePath = std::getenv("TRT_NCCL_ID_FILE");
    if (!filePath)
    {
        sample::gLogError << "FATAL: TRT_NCCL_ID_FILE environment variable is not set!" << std::endl;
        sample::gLogError << "Please set TRT_NCCL_ID_FILE to a shared file path accessible by all ranks." << std::endl;
        sample::gLogError << "Run with --help for example commands." << std::endl;
        ASSERT(false && "TRT_NCCL_ID_FILE environment variable must be set");
    }
    return std::string(filePath);
}

//! Get NCCL unique ID using file-based coordination.
//! Rank 0 generates the ID and writes it to the file.
//! Other ranks wait for the file to be written and read the ID from it.
[[nodiscard]] ncclUniqueId getNcclIdViaFile(int32_t rank)
{
    std::string const filePath = getNcclIdFilePath();
    constexpr size_t kEXPECTED_HEX_LEN = sizeof(ncclUniqueId) * 2;
    constexpr int32_t kPOLL_INTERVAL_MS = 10;
    constexpr int32_t kTIMEOUT_MS = 30000; // 30 seconds timeout

    if (rank == 0)
    {
        // Rank 0: Check if stale file exists from a previous run
        std::ifstream checkFile(filePath);
        if (checkFile)
        {
            std::string content;
            std::getline(checkFile, content);
            if (!content.empty())
            {
                throw std::runtime_error(
                    "NCCL ID file already exists with content: " + filePath + "\n"
                    "This may be stale data from a previous run. Please delete it first:\n"
                    "  rm -f " + filePath);
            }
        }

        // Generate NCCL ID and write to file
        ncclUniqueId id;
        NCCLCHECK(ncclGetUniqueId(&id));

        std::string const hexStr = ncclIdToHex(id);

        std::ofstream outFile(filePath, std::ios::trunc);
        if (!outFile)
        {
            throw std::runtime_error("Failed to open NCCL ID file for writing: " + filePath);
        }
        outFile << hexStr;
        outFile.close();

        sample::gLogInfo << "Rank 0 - Generated NCCL ID and wrote to file: " << filePath << std::endl;
        return id;
    }
    else
    {
        // Other ranks: Wait for file to be written and read the ID
        int32_t elapsedMs = 0;
        std::string hexStr;

        while (elapsedMs < kTIMEOUT_MS)
        {
            std::ifstream inFile(filePath);
            if (inFile)
            {
                std::getline(inFile, hexStr);
                if (hexStr.length() == kEXPECTED_HEX_LEN)
                {
                    sample::gLogInfo << "Rank " << rank << " - Read NCCL ID from file: " << filePath << std::endl;
                    return hexToNcclId(hexStr);
                }
            }
            // File not ready yet, wait and retry
            std::this_thread::sleep_for(std::chrono::milliseconds(kPOLL_INTERVAL_MS));
            elapsedMs += kPOLL_INTERVAL_MS;
        }

        throw std::runtime_error("Timeout waiting for NCCL ID file to be written by rank 0");
    }
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

nvonnxparser::IParserRefitter* createONNXRefitter(IRefitter& refitter)
{
    if (!initNvonnxparser())
    {
        return {};
    }
    ASSERT(pCreateNvOnnxRefitterInternal != nullptr);
    return static_cast<nvonnxparser::IParserRefitter*>(
        pCreateNvOnnxRefitterInternal(&refitter, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

//! Helper struct to hold test configuration for each collective operation
struct CollectiveTestConfig
{
    CollectiveOperation op;
    std::vector<float> rank0Input;
    std::vector<float> rank1Input;
    std::vector<float> rank0ExpectedOutput;
    std::vector<float> rank1ExpectedOutput; // Different from rank0 for REDUCE_SCATTER
    int32_t outputElementCount;             // Number of output elements per rank
};

//! Get test configuration for a specific collective operation
CollectiveTestConfig getTestConfig(CollectiveOperation op, int32_t worldSize)
{
    // Input: 12 elements per rank [3, 4]
    // After transpose: [4, 3]
    constexpr int32_t kINPUT_SIZE = 12;

    switch (op)
    {
    case CollectiveOperation::kALL_GATHER:
    {
        // ALL_GATHER: Each rank contributes data, all ranks receive concatenated result
        // Output: 12 * worldSize = 24 elements
        std::vector<float> const expected = {0.0F, 1.0F, 2.0F, 3.0F, 100.0F, 101.0F, 102.0F, 103.0F, 4.0F, 5.0F, 6.0F,
            7.0F, 104.0F, 105.0F, 106.0F, 107.0F, 8.0F, 9.0F, 10.0F, 11.0F, 108.0F, 109.0F, 110.0F, 111.0F};
        return {CollectiveOperation::kALL_GATHER,
            {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F},
            {100.0F, 101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 106.0F, 107.0F, 108.0F, 109.0F, 110.0F, 111.0F}, expected,
            expected, // Both ranks get same result
            kINPUT_SIZE * worldSize};
    }

    case CollectiveOperation::kALL_REDUCE:
    {
        // ALL_REDUCE: Sum across all ranks, all ranks receive same result
        std::vector<float> const expected
            = {11.0F, 22.0F, 33.0F, 44.0F, 55.0F, 66.0F, 77.0F, 88.0F, 99.0F, 110.0F, 121.0F, 132.0F};
        return {CollectiveOperation::kALL_REDUCE,
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
            {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F, 90.0F, 100.0F, 110.0F, 120.0F}, expected,
            expected, // Both ranks get same result
            kINPUT_SIZE};
    }

    case CollectiveOperation::kBROADCAST:
    {
        // BROADCAST: Rank 0 sends data to all ranks
        std::vector<float> const expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
        return {CollectiveOperation::kBROADCAST,
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
            {99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F}, expected,
            expected, // Both ranks get same result
            kINPUT_SIZE};
    }

    case CollectiveOperation::kREDUCE:
        // REDUCE: Sum across all ranks, only root (rank 0) receives result
        // rank1's output is undefined, use empty vector
        return {CollectiveOperation::kREDUCE,
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
            {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F, 90.0F, 100.0F, 110.0F, 120.0F},
            {11.0F, 22.0F, 33.0F, 44.0F, 55.0F, 66.0F, 77.0F, 88.0F, 99.0F, 110.0F, 121.0F, 132.0F},
            {}, // rank1's output is undefined
            kINPUT_SIZE};

    case CollectiveOperation::kREDUCE_SCATTER:
        // REDUCE_SCATTER: Reduce then scatter - each rank gets a different portion
        // Input [3,4] -> transpose [4,3] -> reduce_scatter [2,3] -> transpose [3,2] = 6 elements
        // After reduce: [[11,55,99],[22,66,110],[33,77,121],[44,88,132]]
        // rank0 gets first half [[11,55,99],[22,66,110]], after transpose: [[11,22],[55,66],[99,110]]
        // rank1 gets second half [[33,77,121],[44,88,132]], after transpose: [[33,44],[77,88],[121,132]]
        return {CollectiveOperation::kREDUCE_SCATTER,
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
            {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F, 90.0F, 100.0F, 110.0F, 120.0F},
            {11.0F, 22.0F, 55.0F, 66.0F, 99.0F, 110.0F},  // rank0 expected
            {33.0F, 44.0F, 77.0F, 88.0F, 121.0F, 132.0F}, // rank1 expected
            kINPUT_SIZE / worldSize};
    }
    throw std::runtime_error("Unknown CollectiveOperation");
}

//! Build and execute a network with a specific collective operation
void testCollectiveOperation(
    int32_t rank, int32_t worldSize, CollectiveTestConfig const& config, ncclComm_t comm, cudaStream_t stream)
{
    sample::gLogInfo << "Rank " << rank << " - Testing " << toString(config.op) << std::endl;

    // Create builder and network
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(sample::gLogger.getTRTLogger()));
    ASSERT(builder != nullptr);

    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    ASSERT(network != nullptr);

    // Create input tensor
    constexpr int32_t kINPUT_ROWS = 3;
    constexpr int32_t kINPUT_COLS = 4;
    auto* input = network->addInput("input", DataType::kFLOAT, Dims2{kINPUT_ROWS, kINPUT_COLS});
    ASSERT(input != nullptr);

    auto* firstShuffle = network->addShuffle(*input);
    ASSERT(firstShuffle != nullptr);
    firstShuffle->setFirstTranspose({{1, 0}});

    ReduceOperation reduceOp = ReduceOperation::kNONE;
    if (config.op == CollectiveOperation::kALL_REDUCE || config.op == CollectiveOperation::kREDUCE
        || config.op == CollectiveOperation::kREDUCE_SCATTER)
    {
        reduceOp = ReduceOperation::kSUM;
    }
    int64_t root = -1;
    if (config.op == CollectiveOperation::kBROADCAST || config.op == CollectiveOperation::kREDUCE)
    {
        root = 0;
    }
    auto* collectiveLayer
        = network->addDistCollective(*firstShuffle->getOutput(0), config.op, reduceOp, root, nullptr, 0);
    ASSERT(collectiveLayer != nullptr);

    // Set the number of ranks for the collective operation
    if (!collectiveLayer->setNbRanks(worldSize))
    {
        throw std::runtime_error("Failed to set the number of ranks for the collective layer");
    }

    auto* secondShuffle = network->addShuffle(*collectiveLayer->getOutput(0));
    ASSERT(secondShuffle != nullptr);
    secondShuffle->setFirstTranspose({{1, 0}});

    // Mark the reshape layer's output as the network output
    network->markOutput(*secondShuffle->getOutput(0));

    // Build engine
    auto builderConfig = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    ASSERT(builderConfig != nullptr);
    builderConfig->setPreviewFeature(PreviewFeature::kMULTIDEVICE_RUNTIME_10_16, true);
    auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *builderConfig));
    ASSERT(serializedEngine != nullptr);

    // Create runtime and deserialize engine
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    ASSERT(runtime != nullptr);

    // Deserialize the CUDA engine
    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    ASSERT(engine != nullptr);

    // Create execution context for the engine
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    ASSERT(context != nullptr);

    // Prepare input and output buffers
    char const* inputName = engine->getIOTensorName(0);
    char const* outputName = engine->getIOTensorName(1);

    std::vector<float> const& inputChunk = (rank == 0) ? config.rank0Input : config.rank1Input;
    std::vector<float> outputChunk(config.outputElementCount, 0.0F);

    size_t const inputBytes = inputChunk.size() * sizeof(float);
    size_t const outputBytes = outputChunk.size() * sizeof(float);

    void* dInput = nullptr;
    void* dOutput = nullptr;
    CHECK_CUDA(cudaMalloc(&dInput, inputBytes));
    CHECK_CUDA(cudaMalloc(&dOutput, outputBytes));

    // Copy input data to GPU asynchronously
    CHECK_CUDA(cudaMemcpyAsync(dInput, inputChunk.data(), inputBytes, cudaMemcpyHostToDevice, stream));

    // Set input/output tensor addresses in the execution context
    context->setInputTensorAddress(inputName, dInput);
    context->setTensorAddress(outputName, dOutput);
    context->setInputShape(inputName, Dims2{kINPUT_ROWS, kINPUT_COLS});

    // Set NCCL communicator
    if (!context->setCommunicator(comm))
    {
        cudaFree(dInput);
        cudaFree(dOutput);
        throw std::runtime_error("Failed to set communicator for " + toString(config.op));
    }

    // Run inference
    if (!context->enqueueV3(stream))
    {
        cudaFree(dInput);
        cudaFree(dOutput);
        throw std::runtime_error("Inference failed for " + toString(config.op));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy output data back to host asynchronously
    CHECK_CUDA(cudaMemcpyAsync(outputChunk.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Get the expected output for this rank
    std::vector<float> const& expectedOutput = (rank == 0) ? config.rank0ExpectedOutput : config.rank1ExpectedOutput;

    // Determine if this rank should verify output
    // REDUCE: only rank 0 receives valid result
    // All other ops: both ranks can verify (same or different expected values)
    bool const shouldVerify = !expectedOutput.empty();

    if (shouldVerify)
    {
        constexpr float kEPS = 1e-5F;
        for (size_t i = 0; i < outputChunk.size() && i < expectedOutput.size(); ++i)
        {
            if (std::abs(outputChunk[i] - expectedOutput[i]) > kEPS)
            {
                cudaFree(dInput);
                cudaFree(dOutput);
                throw std::runtime_error("Output mismatch for " + toString(config.op) + " at index " + std::to_string(i)
                    + ": expected " + std::to_string(expectedOutput[i]) + ", got " + std::to_string(outputChunk[i]));
            }
        }
        sample::gLogInfo << "Rank " << rank << " - " << toString(config.op) << " PASSED" << std::endl;
    }

    // Cleanup
    cudaFree(dInput);
    cudaFree(dOutput);
}

//! Main test function that runs a specific collective operation test
void runCollectiveTest(int32_t rank, int32_t worldSize, CollectiveOperation op)
{
    // Check GPU availability
    int32_t deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < worldSize)
    {
        throw std::runtime_error("Not enough GPUs available. Need " + std::to_string(worldSize) + " but found "
            + std::to_string(deviceCount));
    }

    // Use rank to select GPU
    CHECK_CUDA(cudaSetDevice(rank));

    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Set up NCCL - rank 0 generates ID and writes to file, others read from file
    ncclUniqueId const id = getNcclIdViaFile(rank);
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, worldSize, id, rank));

    // Get test configuration for the specified operation
    CollectiveTestConfig const config = getTestConfig(op, worldSize);

    // Run the collective operation test
    testCollectiveOperation(rank, worldSize, config, comm, stream);

    sample::gLogInfo << "Rank " << rank << " - " << toString(op) << " test completed successfully!" << std::endl;

    NCCLCHECK(ncclCommDestroy(comm));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[])
{
    constexpr int32_t kREQUIRED_WORLD_SIZE = 2;

    for (int32_t i = 1; i < argc; ++i)
    {
        std::string const arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Get rank and world size from TRT_MY_RANK and TRT_WORLD_SIZE environment variables.
    int32_t const rank = getRankFromEnv();
    int32_t const worldSize = getWorldSizeFromEnv();

    // Parse command line arguments
    CollectiveOperation selectedOp{};
    bool hasSelectedOp = false;

    for (int32_t i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--op" && i + 1 < argc)
        {
            ++i;
            auto parsedOp = parseCollectiveOp(argv[i]);
            if (!parsedOp)
            {
                if (rank == 0)
                {
                    sample::gLogError << "Invalid operation: " << argv[i] << std::endl;
                    printUsage(argv[0]);
                }
                return 1;
            }
            selectedOp = *parsedOp;
            hasSelectedOp = true;
        }
    }

    // --op is required
    if (!hasSelectedOp)
    {
        if (rank == 0)
        {
            sample::gLogError << "Error: --op argument is required." << std::endl;
            printUsage(argv[0]);
        }
        return 1;
    }

    // We need exactly 2 processes for this test
    if (worldSize != kREQUIRED_WORLD_SIZE)
    {
        if (rank == 0)
        {
            sample::gLogError << "This sample requires exactly 2 processes, but " << worldSize << " were provided."
                              << std::endl;
            sample::gLogError << "Please set TRT_WORLD_SIZE=2 and launch 2 processes." << std::endl;
            sample::gLogError << "Run with --help for example commands." << std::endl;
        }
        return 1;
    }

    try
    {
        runCollectiveTest(rank, worldSize, selectedOp);
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Rank " << rank << " - Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
