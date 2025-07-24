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

// cublasLT was introduced in CUDA 10.1
#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "common/serialize.hpp"
#include "fcPlugin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;
using namespace nvinfer1::pluginInternal;

// plugin specific constants
namespace
{
char const* const kFC_VERSION{"1"};
char const* const kFC_NAME{"CustomFCPluginDynamic"};
constexpr size_t kMAX_WORKSPACE_BYTES = 4 * 1024 * 1024; // 4MiB
} // namespace

REGISTER_TENSORRT_PLUGIN(FCPluginDynamicCreator);

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(customMatmulPerf_t const& perf, int32_t const m, int32_t const n, int32_t const k)
{
    AlgoProps p;
    p.populate(perf.algo);
    // Calculate GFLOPS
    double timeAvg
        = perf.time * 1e-3; // Convert to seconds. It has been divided by kNB_KERNEL_REPEATS in customMatmulRun().
    double gflop = (2 * static_cast<uint64_t>(m * n) * k) * 1e-9; // Real

    gLogVerbose << "Algo=" << p.algoId << " Tile=" << p.tile << " (" << matmulTileName[p.tile] << ") K=" << p.numSplitsK
                << " Red.Sch.=" << p.reductionScheme << " Swiz=" << p.swizzle << " Cust=" << p.customOption
                << " Stat=" << perf.status << " Time=" << perf.time << " WSbytes=" << perf.workspaceSize
                << " math=" << p.numericImpl << " waves=" << perf.wavesCount << "GFlops=" << (gflop / timeAvg)
                << std::endl;
}

static bool timeCompare(customMatmulPerf_t const& perf_a, customMatmulPerf_t const& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle, // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc, void const* alpha,       // host or device pointer
    void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc,
    void const* beta, // host or device pointer
    void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const& algo, void* workSpace, size_t workSpaceSizeInBytes, customMatmulPerf_t& perfResults,
    cudaStream_t stream, cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{

    cublasLtMatmulHeuristicResult_t heurResult;

    CublasLtWrapper& cublasLtWrapper = getCublasLtWrapper();
    // Looping over the Algo
    cublasStatus_t algoStatus = cublasLtWrapper.cublasLtMatmulAlgoCheck(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS)
    {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes)
        {
            if (cudaEventRecord(startEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            for (int32_t loop = 0; loop < kNB_KERNEL_REPEATS; loop++)
            {
                cublasStatus_t oneRunStatus
                    = cublasLtWrapper.cublasLtMatmul(ltHandle, operationDesc, alpha, // host or device pointer
                        A, Adesc, B, Bdesc, beta,                                    // host or device pointer
                        C, Cdesc, D, Ddesc, &algo, workSpace, workSpaceSizeInBytes, stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS)
                {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            if (cudaEventRecord(stopEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            if (cudaEventSynchronize(stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            float time;
            if (cudaEventElapsedTime(&time, startEvent, stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            perfResults.algo = algo;
            perfResults.time = time / kNB_KERNEL_REPEATS; // Average time
            perfResults.workspaceSize = heurResult.workspaceSize;
            perfResults.wavesCount = heurResult.wavesCount;
        }
        else
        {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
        }
    }
    return algoStatus;
}

// Sample wrapper running through multiple algo and config attributes
// combination for single precision gemm using cublasLt low-level API
void nvinfer1::plugin::bert::LtGemmSearch(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb,
    int32_t const& m, int32_t const& n, int32_t const& k, void const* alpha,                // host pointer
    void const* A, int32_t const& lda, void const* B, int32_t const& ldb, void const* beta, // host pointer
    void* C, int32_t const& ldc, void* workSpace, size_t workSpaceSize, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    std::vector<customMatmulPerf_t>& perfResults, cudaStream_t stream)
{

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    CublasLtWrapper& cublasLtWrapper = getCublasLtWrapper();

    // SplitK value that we are going to try when SplitK is supported for a given algo.
    int32_t const splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Let try a fixed number of combinations
    int32_t algoCount = 0;
    int32_t nbAlgoIds = 0;
    int32_t algoIdA[kNB_ALGO_IDS];

    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulPreferenceCreate(&preference));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

    uint64_t const numericImplPrefer
        = Ctype == CUDA_R_16F ? CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA : CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA;
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &numericImplPrefer, sizeof(numericImplPrefer)));

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details
    // about defaults; here we just need to set the transforms for A and B
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulDescSetAttribute(
        operationDesc, nvinfer1::pluginInternal::CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulDescSetAttribute(
        operationDesc, nvinfer1::pluginInternal::CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // Create matrix descriptors. We are good with the details here so no need to
    // set any extra attributes
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutCreate(
        &Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutCreate(
        &Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType =
    // Atype = Btype = Ctype = Dtype = CUDA_R_32F)
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, kNB_ALGO_IDS, algoIdA, &nbAlgoIds));

    gLogVerbose << "Number of algos" << nbAlgoIds << std::endl;

    // Create CUDA event to time the execution time of each algo
    PLUGIN_CUASSERT(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    PLUGIN_CUASSERT(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    // Loop over the Algo IDs
    for (int32_t idx = 0; (idx < nbAlgoIds) && (algoCount < kNB_ALGO_COMBINATIONS); idx++)
    {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        // Initialize algo structure with given Algp ID.
        status = cublasLtWrapper.cublasLtMatmulAlgoInit(
            ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }

        uint64_t numericImpl = -1;
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &numericImpl, sizeof(numericImpl), nullptr));
        if (Ctype == CUDA_R_32F && numericImpl == CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
        {
            // skip HMMA-fp32accu kernels
            continue;
        }

        // Query the tiles enums supported by that algo
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
        int32_t nbTiles = int32_t(sizeWritten / sizeof(int32_t));
        int32_t* tileA = new int32_t[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0)
        {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int32_t splitkSupport;
        int32_t redMask;
        int32_t swizzlingMax;
        int32_t customOptionMax;
        int32_t epilogueMask;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int32_t) * nbTiles, &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));

        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

        // Loop over the different tiles
        for (int32_t tileIdx = 0; tileIdx < nbTiles; tileIdx++)
        {
            // Loop over the different custom option if any
            for (int32_t customOption = 0; customOption <= customOptionMax; customOption++)
            {
                PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
                // Loop over the CTAs swizzling support
                for (int32_t k = 0; k <= swizzlingMax; k++)
                {
                    int32_t splitkTrial = 0;
                    if (splitkSupport)
                    {
                        splitkTrial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in
                    // addition to the case where splitK is not enabled
                    for (int32_t l = 0; (l < (1 + splitkTrial)) && (algoCount < kNB_ALGO_COMBINATIONS); l++)
                    {
                        // Setup attribute of the algo to run
                        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx])));
                        int32_t splitK_val = 0;
                        int32_t redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
                        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                        PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int32_t)));

                        if (l > 0)
                        { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1],
                                sizeof(splitKSequenceA[l - 1])));
                            // Going over all the reduction scheme
                            for (redScheme = 1; redScheme < static_cast<int32_t>(CUBLASLT_REDUCTION_SCHEME_MASK)
                                 && (algoCount < kNB_ALGO_COMBINATIONS);
                                 redScheme = redScheme << 1)
                            {
                                if (redScheme & redMask)
                                {
                                    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme)));

                                    status = customMatmulRun(ltHandle, operationDesc, alpha, // host or device pointer
                                        A, Adesc, B, Bdesc, beta,                            // host or device pointer
                                        C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount],
                                        stream, startEvent, stopEvent);
                                    perfResults[algoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS)
                                    {
                                        algoCount++;
                                    }
                                } // end if
                            }     // end for
                        }
                        else
                        { // Non-splitK case
                            // if user preference is ok with workspace
                            if (algoCount < kNB_ALGO_COMBINATIONS)
                            {
                                status = customMatmulRun(ltHandle, operationDesc, alpha, // host or device pointer
                                    A, Adesc, B, Bdesc, beta,                            // host or device pointer
                                    C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount], stream,
                                    startEvent, stopEvent);
                                perfResults[algoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS)
                                {
                                    algoCount++;
                                }
                            }
                        }
                    } // end l
                }     // end k
            }         // end customOption
        }             // end tileIdx
        delete[] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults.begin(), perfResults.end(), timeCompare);

    // Print timing and perf details of the fastest combinations
    for (int32_t i = 0; i < kPRINT_ALGOS; i++)
    {
        if (perfResults[i].time == customMatmulPerf_t::kMAX_TIME)
        {
            break;
        }
        printPerfStructure(perfResults[i], m, n, k);
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulPreferenceDestroy(preference));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutDestroy(Cdesc));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutDestroy(Bdesc));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatrixLayoutDestroy(Adesc));
    PLUGIN_CUBLASASSERT(cublasLtWrapper.cublasLtMatmulDescDestroy(operationDesc));
    PLUGIN_CUASSERT(cudaEventDestroy(startEvent));
    PLUGIN_CUASSERT(cudaEventDestroy(stopEvent));
}

FCPluginDynamic::FCPluginDynamic(std::string const name, DataType const type, int32_t const outDim, Weights const& W)
    : mLayerName(name)
    , mType(type)
    , mOutDim(outDim)
    , mNumParams(W.count)
    , mNmax(0)
    , mK(0)
    , mWdev(nullptr)
{
    memset(mAlgo.data, 0, sizeof(mAlgo.data));

    mW.convertAndCopy(W, mType);
    copyToDevice(mW, getWeightsSize(mW, mType), mWdev);
}

FCPluginDynamic::FCPluginDynamic(std::string const name, void const* data, size_t length)
    : mLayerName(name)
    , mWdev(nullptr)
{
    gLogVerbose << "FCPluginDynamic deserialize\n";

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mOutDim);
    deserialize_value(&data, &length, &mNumParams);
    deserialize_value(&data, &length, &mNmax);
    deserialize_value(&data, &length, &mK);
    deserialize_value(&data, &length, &mAlgo);

    char const* d = static_cast<char const*>(data);

    mW.convertAndCopy(d, mNumParams, mType);
    copyToDevice(mW, getWeightsSize(mW, mType), mWdev);
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FCPluginDynamic::clone() const noexcept
{
    try
    {
        gLogVerbose << "FCPluginDynamic clone\n";

        auto* p = new FCPluginDynamic(mLayerName, mType, mOutDim, mW);
        memcpy(p->mAlgo.data, mAlgo.data, sizeof(mAlgo.data));
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FCPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept
{
    mLtContext.attach();
}

void FCPluginDynamic::detachFromContext() noexcept
{
    mLtContext.detach();
}

DimsExprs FCPluginDynamic::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(inputs != nullptr);
        DimsExprs ret;
        ret.nbDims = 5;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = exprBuilder.constant(mOutDim);
        ret.d[3] = exprBuilder.constant(1);
        ret.d[4] = exprBuilder.constant(1);
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FCPluginDynamic::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inOut != nullptr);

    PluginTensorDesc const& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    PluginTensorDesc const& prev = inOut[pos - 1];

    // output
    return in.type == prev.type && in.format == prev.format;
}

void FCPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(mType == inputs[0].desc.type);
        auto const& inDims0 = inputs[0].desc.dims;

        PLUGIN_VALIDATE(inDims0.nbDims == 5);
        mK = inDims0.d[HDIM]; // hiddensize
        // PLUGIN_ASSERT(hiddenSize * mOutDim == mNumParams);
        PLUGIN_VALIDATE(inDims0.d[3] == 1);
        PLUGIN_VALIDATE(inDims0.d[4] == 1);

        // m and k are mOutDim
        // n is B*S
        int32_t const S = inputs->max.d[SDIM];
        int32_t const B = inputs->max.d[BDIM];

        mNmax = S * B;

        // Cleanup LtContext descriptors before creating new ones.
        mLtContext.destroy();

        if (mType == DataType::kFLOAT)
        {
            Gemm<float> g(mOutDim, mNmax, mK, false, false);
            mLtContext.create(g, kMAX_WORKSPACE_BYTES);
        }
        else if (mType == DataType::kHALF)
        {
            Gemm<half> g(mOutDim, mNmax, mK, false, false);
            mLtContext.create(g, kMAX_WORKSPACE_BYTES);
        }
        else
        {
            std::string const msg = "Unsupported type error, expected [kHALF,kFLOAT], but received ";
            PLUGIN_VALIDATE(false, (msg + std::to_string(static_cast<int32_t>(mType))).c_str());
        }

        gLogVerbose << "FCPluginDynamic configurePlugin m=" << mOutDim << ", n=" << mNmax << ", k=" << mK << std::endl;

        size_t actualWorkspace = 0;
        if (mAlgo.data[0] == 0 && memcmp(mAlgo.data, mAlgo.data + 1, sizeof(mAlgo.data) - sizeof(mAlgo.data[0])) == 0)
        {
            gLogVerbose << "FCPluginDynamic gemmSearch\n";
            if (mSharedStream == nullptr)
            {
                SharedStream ss{};
                mSharedStream = static_cast<SharedStream*>(
                    getPluginRegistry()->acquirePluginResource(kFCPLUGIN_SHARED_STREAM_KEY, &ss))
                                    ->mStream;
            }
            if (mType == DataType::kFLOAT)
            {
                mAlgo = gemmSearch<float>(mOutDim, mNmax, mK, kMAX_WORKSPACE_BYTES, actualWorkspace, mSharedStream);
            }
            else if (mType == DataType::kHALF)
            {
                mAlgo = gemmSearch<half>(mOutDim, mNmax, mK, kMAX_WORKSPACE_BYTES, actualWorkspace, mSharedStream);
            }
        }

        AlgoProps p;
        p.populate(mAlgo);

        if (mType == DataType::kFLOAT && p.numericImpl == CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
        {
            gLogWarning << "cuBLAS might use mixed precision instead of FP32" << std::endl;
        }

        if (mType == DataType::kHALF && p.numericImpl != CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
        {
            gLogWarning << "TensorCore support was not selected" << std::endl;
        }

        gLogVerbose << "FCPluginDynamic configuration Algo=" << p.algoId << " Tile=" << p.tile << " ("
                    << matmulTileName[p.tile] << ") K=" << p.numSplitsK << " Red.Sch.=" << p.reductionScheme
                    << " Swiz=" << p.swizzle << " Cust=" << p.customOption << " numericImpl=" << p.numericImpl
                    << " ws=" << actualWorkspace << std::endl;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t FCPluginDynamic::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return kMAX_WORKSPACE_BYTES;
}

int32_t FCPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workSpace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr
            && workSpace != nullptr);

        size_t const workspaceSize = getWorkspaceSize(inputDesc, 1, outputDesc, 1);

        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const n = S * B;
        PLUGIN_VALIDATE(n >= 0);
        mLtContext.setN(static_cast<uint64_t>(n));

        if (mType == DataType::kFLOAT)
        {
            auto const* const input = static_cast<float const*>(inputs[0]);
            auto* output = static_cast<float*>(outputs[0]);

            Gemm<float> g(mOutDim, n, mK, false, false);
            if (mWdev == nullptr)
            {
                return STATUS_FAILURE;
            }
            g.A = static_cast<float*>(mWdev.get());
            g.B = const_cast<float*>(input);
            g.C = output;

            return cublasLtMatmul(mLtContext, g, mAlgo, workSpace, workspaceSize, stream);
        }
        if (mType == DataType::kHALF)
        {
            auto const* const input = static_cast<half const*>(inputs[0]);
            auto* output = static_cast<half*>(outputs[0]);

            Gemm<half> g(mOutDim, n, mK, false, false);
            if (mWdev == nullptr)
            {
                return STATUS_FAILURE;
            }
            g.A = static_cast<half*>(mWdev.get());
            g.B = const_cast<half*>(input);
            g.C = output;
            return cublasLtMatmul(mLtContext, g, mAlgo, workSpace, workspaceSize, stream);
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int32_t>(mType)
                      << std::endl;
            return STATUS_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

// IPluginV2Ext Methods
DataType FCPluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
char const* FCPluginDynamic::getPluginType() const noexcept
{
    return kFC_NAME;
}

char const* FCPluginDynamic::getPluginVersion() const noexcept
{
    return kFC_VERSION;
}

int32_t FCPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int32_t FCPluginDynamic::initialize() noexcept
{
    gLogVerbose << "FCPluginDynamic initialize\n";
    return 0;
}

void FCPluginDynamic::terminate() noexcept
{
    gLogVerbose << "FCPluginDynamic terminate\n";
    if (mSharedStream)
    {
        TRT_UNUSED(getPluginRegistry()->releasePluginResource(kFCPLUGIN_SHARED_STREAM_KEY));
        mSharedStream = nullptr;
    }
}

size_t FCPluginDynamic::getSerializationSize() const noexcept
{
    size_t wordSize = getElementSize(mType);
    return wordSize * mNumParams + sizeof(mType) + sizeof(mOutDim) + sizeof(mNumParams) + sizeof(mAlgo) + sizeof(mNmax)
        + sizeof(mK);
}

void FCPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mOutDim);
    serialize_value(&buffer, mNumParams);
    serialize_value(&buffer, mNmax);
    serialize_value(&buffer, mK);
    serialize_value(&buffer, mAlgo);

    size_t wordSize = getElementSize(mType);
    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mWdev.get()), mNumParams * wordSize);
}

void FCPluginDynamic::destroy() noexcept
{
    gLogVerbose << "FCPluginDynamic destroy\n";
    // This gets called when the network containing plugin is destroyed
    mLtContext.destroy();
    mWdev.reset(nullptr);
    delete this;
}

void FCPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* FCPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

FCPluginDynamicCreator::FCPluginDynamicCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("out_dims", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("W", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FCPluginDynamicCreator::getPluginName() const noexcept
{
    return kFC_NAME;
}

char const* FCPluginDynamicCreator::getPluginVersion() const noexcept
{
    return kFC_VERSION;
}

PluginFieldCollection const* FCPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FCPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogVerbose << "Creating FCPluginDynamicCreator...\n";
        PLUGIN_VALIDATE(name != nullptr);
        PLUGIN_VALIDATE(fc != nullptr);

        int32_t outDims = 0;
        int32_t typeId = -1;
        Weights W{DataType::kFLOAT, nullptr, 0LL};
        plugin::validateRequiredAttributesExist({"out_dims", "type_id", "W"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("out_dims") == 0)
            {
                outDims = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogVerbose << "Building outDims: " << outDims << std::endl;
            }

            if (fieldName.compare("type_id") == 0)
            {
                typeId = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogVerbose << "Building typeId: " << outDims << std::endl;
            }

            if (fieldName.compare("W") == 0)
            {
                gLogVerbose << "Building W...\n";
                W.values = fc->fields[i].data;
                W.count = fc->fields[i].length;
                W.type = fieldTypeToDataType(fc->fields[i].type);
                gLogVerbose << "Is W float32: " << (W.type == DataType::kFLOAT) << std::endl;
            }
        }

        if (outDims <= 0)
        {
            gLogError << "Invalid output dimension" << std::endl;
        }
        if (typeId < 0 || typeId > 1)
        {
            gLogError << "Invalid type id" << typeId << std::endl;
        }
        if (W.count == 0 || W.values == nullptr || W.count < outDims)
        {
            gLogError << "Invalid weights" << std::endl;
        }

        DataType type = static_cast<DataType>(typeId);
        return new FCPluginDynamic(name, type, outDims, W);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FCPluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    try
    {
        return new FCPluginDynamic(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FCPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* FCPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // #if CUDA_VERSION >= 10010
