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

// cublasLT was introduced in CUDA 10.1
#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "common/serialize.hpp"
#include "fcPlugin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

// plugin specific constants
namespace
{
const char* FC_VERSION{"1"};
const char* FC_NAME{"CustomFCPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection FCPluginDynamicCreator::mFC{};
std::vector<PluginField> FCPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FCPluginDynamicCreator);

constexpr size_t maxWorkspaceBytes = 4194304; // 4MB

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t& perf, int const& m, int const& n, int const& k)
{
    AlgoProps p;
    p.populate(perf.algo);
    /* Calculate GFLOPS */
    double timeAvg = perf.time * 1e-3; // Convert to seconds. It has been divided by kernelRepeats in customMatmulRun().
    double gflop = (2 * static_cast<unsigned long long int>(m * n) * k) * 1e-9; // Real

    gLogVerbose << "Algo=" << p.algoId << " Tile=" << p.tile << " (" << matmulTileName[p.tile] << ") K=" << p.numSplitsK
                << " Red.Sch.=" << p.reductionScheme << " Swiz=" << p.swizzle << " Cust=" << p.customOption
                << " Stat=" << perf.status << " Time=" << perf.time << " WSbytes=" << perf.workspaceSize
                << " math=" << p.numericImpl << " waves=" << perf.wavesCount << "GFlops=" << (gflop / timeAvg)
                << std::endl;
}

static inline bool time_compare(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle, // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc, void const* alpha,       /* host or device pointer */
    void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc,
    void const* beta, /* host or device pointer */
    void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const& algo, void* workSpace, size_t workSpaceSizeInBytes, customMatmulPerf_t& perfResults,
    cudaStream_t stream, cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{

    cublasLtMatmulHeuristicResult_t heurResult;

    /* Looping over the Algo */
    cublasStatus_t algoStatus
        = cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS)
    {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes)
        {
            if (cudaEventRecord(startEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            for (int loop = 0; loop < kernelRepeats; loop++)
            {
                cublasStatus_t oneRunStatus
                    = cublasLtMatmul(ltHandle, operationDesc, alpha, /* host or device pointer */
                        A, Adesc, B, Bdesc, beta,                    /* host or device pointer */
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
            perfResults.time = time / kernelRepeats; // Average time
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
void LtGemmSearch(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int const& m,
    int const& n, int const& k, void const* alpha,                                  /* host pointer */
    void const* A, int const& lda, void const* B, int const& ldb, void const* beta, /* host pointer */
    void* C, int const& ldc, void* workSpace, size_t workSpaceSize,
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t computeType,
#else
    cublasComputeType_t computeType,
#endif
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    std::vector<customMatmulPerf_t>& perfResults)
{

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    cudaStream_t stream = nullptr;

    // SplitK value that we are going to try when SplitK is supported for a given
    // algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Let try a fixed number of combinations
    int algoCount = 0;
    int nbAlgoIds = 0;
    int algoIdA[algoIds];
    // customMatmulPerf_t perfResults[algoCombinations];

    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceCreate(&preference));
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

    uint64_t const numericImplPrefer
        = Ctype == CUDA_R_16F ? CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA : CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA;
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &numericImplPrefer, sizeof(numericImplPrefer)));
    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details
    // about defaults; here we just need to set the transforms for A and B
#if CUBLAS_VER_MAJOR < 11
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType));
#else
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#endif
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // Create matrix descriptors. We are good with the details here so no need to
    // set any extra attributes
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType =
    // Atype = Btype = Ctype = Dtype = CUDA_R_32F)
    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIds, algoIdA, &nbAlgoIds));

    gLogVerbose << "Number of algos" << nbAlgoIds << std::endl;

    // Create CUDA event to time the execution time of each algo
    PLUGIN_CUASSERT(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    PLUGIN_CUASSERT(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (algoCount < algoCombinations); idx++)
    {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status
            = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }

        uint64_t numericImpl = -1;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &numericImpl, sizeof(numericImpl), nullptr));
        if (Ctype == CUDA_R_32F && numericImpl == CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
        {
            // skip HMMA-fp32accu kernels
            continue;
        }

        // Query the tiles enums supported by that algo
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
        int nbTiles = int(sizeWritten / sizeof(int));
        int* tileA = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0)
        {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax, epilogueMask;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));

        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++)
        {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++)
            {
                PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
                /* Loop over the CTAs swizzling support */
                for (int k = 0; k <= swizzlingMax; k++)
                {
                    int splitK_trial = 0;
                    if (splitkSupport)
                    {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in
                    // addition to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (algoCount < algoCombinations); l++)
                    {
                        /* Setup attribute of the algo to run */
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx])));
                        int splitK_val = 0;
                        int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int)));

                        if (l > 0)
                        { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            PLUGIN_CUBLASASSERT(
                                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                    &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1])));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1; redScheme < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK)
                                 && (algoCount < algoCombinations);
                                 redScheme = redScheme << 1)
                            {
                                if (redScheme & redMask)
                                {
                                    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme)));

                                    status
                                        = customMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                            A, Adesc, B, Bdesc, beta,                     /* host or device pointer */
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
                            /* if user preference is ok with workspace */
                            if (algoCount < algoCombinations)
                            {
                                status = customMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                    A, Adesc, B, Bdesc, beta,                            /* host or device pointer */
                                    C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount], stream,
                                    startEvent, stopEvent);
                                perfResults[algoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS)
                                    algoCount++;
                            }
                        }
                    } // end l
                }     // end k
            }         // end customOption
        }             // end tileIdx
        delete[] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults.begin(), perfResults.end(), time_compare);
    // Print timing and perf details of the fastest combinations
    // for (int i = 0; i < perfResults.size(); i++){
    for (int i = 0; i < printAlgos; i++)
    {
        if (perfResults[i].time == 1000000.F)
            break;
        printPerfStructure(perfResults[i], m, n, k);
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceDestroy(preference));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Cdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Bdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Adesc));
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescDestroy(operationDesc));
    PLUGIN_CUASSERT(cudaEventDestroy(startEvent));
    PLUGIN_CUASSERT(cudaEventDestroy(stopEvent));
}

FCPluginDynamic::FCPluginDynamic(const std::string name, const DataType type, const int outDim, const Weights& W)
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

FCPluginDynamic::FCPluginDynamic(const std::string name, const void* data, size_t length)
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

    const char* d = static_cast<const char*>(data);

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
    catch (const std::exception& e)
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
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 1);
        PLUGIN_ASSERT(outputIndex == 0);
        DimsExprs ret;
        ret.nbDims = 5;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = exprBuilder.constant(mOutDim);
        ret.d[3] = exprBuilder.constant(1);
        ret.d[4] = exprBuilder.constant(1);
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FCPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    // output
    return in.type == prev.type && in.format == prev.format;
}

void FCPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_ASSERT(nbOutputs == 1);
        PLUGIN_ASSERT(nbInputs == 1);
        PLUGIN_ASSERT(mType == inputs[0].desc.type);
        const auto& inDims0 = inputs[0].desc.dims;

        PLUGIN_ASSERT(inDims0.nbDims == 5);
        mK = inDims0.d[HDIM]; // hiddensize
        // PLUGIN_ASSERT(hiddenSize * mOutDim == mNumParams);
        PLUGIN_ASSERT(inDims0.d[3] == 1);
        PLUGIN_ASSERT(inDims0.d[4] == 1);

        // m and k are mOutDim
        // n is B*S
        const int S = inputs->max.d[SDIM];
        const int B = inputs->max.d[BDIM];

        mNmax = S * B;

        // Cleanup LtContext descriptors before creating new ones.
        mLtContext.destroy();

        if (mType == DataType::kFLOAT)
        {
            Gemm<float> g(mOutDim, mNmax, mK, false, false);
            mLtContext.create(g, maxWorkspaceBytes);
        }
        else if (mType == DataType::kHALF)
        {
            Gemm<half> g(mOutDim, mNmax, mK, false, false);
            mLtContext.create(g, maxWorkspaceBytes);
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(mType)
                      << std::endl;
            PLUGIN_ASSERT(false);
        }

        gLogVerbose << "FCPluginDynamic configurePlugin m=" << mOutDim << ", n=" << mNmax << ", k=" << mK << std::endl;

        size_t actualWorkspace = 0;
        if (mAlgo.data[0] == 0 && memcmp(mAlgo.data, mAlgo.data + 1, sizeof(mAlgo.data) - sizeof(mAlgo.data[0])) == 0)
        {
            gLogVerbose << "FCPluginDynamic gemmSearch\n";
            if (mType == DataType::kFLOAT)
            {
                mAlgo = gemmSearch<float>(mOutDim, mNmax, mK, maxWorkspaceBytes, actualWorkspace);
            }
            else if (mType == DataType::kHALF)
            {
                mAlgo = gemmSearch<half>(mOutDim, mNmax, mK, maxWorkspaceBytes, actualWorkspace);
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
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

size_t FCPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return maxWorkspaceBytes;
}

int FCPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workSpace, cudaStream_t stream) noexcept
{
    try
    {
        const size_t workspaceSize = getWorkspaceSize(inputDesc, 1, outputDesc, 1);

        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const n = S * B;
        PLUGIN_ASSERT(n >= 0);
        mLtContext.setN(static_cast<uint64_t>(n));

        if (mType == DataType::kFLOAT)
        {
            const auto* const input = static_cast<const float*>(inputs[0]);
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
            const auto* const input = static_cast<const half*>(inputs[0]);
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
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(mType)
                      << std::endl;
            return STATUS_FAILURE;
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

// IPluginV2Ext Methods
DataType FCPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* FCPluginDynamic::getPluginType() const noexcept
{
    return FC_NAME;
}

const char* FCPluginDynamic::getPluginVersion() const noexcept
{
    return FC_VERSION;
}

int FCPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int FCPluginDynamic::initialize() noexcept
{
    gLogVerbose << "FCPluginDynamic initialize\n";
    return 0;
}

void FCPluginDynamic::terminate() noexcept
{
    gLogVerbose << "FCPluginDynamic terminate\n";
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

void FCPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* FCPluginDynamic::getPluginNamespace() const noexcept
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

const char* FCPluginDynamicCreator::getPluginName() const noexcept
{
    return FC_NAME;
}

const char* FCPluginDynamicCreator::getPluginVersion() const noexcept
{
    return FC_VERSION;
}

const PluginFieldCollection* FCPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FCPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        gLogVerbose << "Creating FCPluginDynamicCreator...\n";

        int32_t outDims = 0;
        int32_t typeId = -1;
        Weights W{DataType::kFLOAT, nullptr, 0ll};
        plugin::validateRequiredAttributesExist({"out_dims", "type_id", "W"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("out_dims") == 0)
            {
                outDims = static_cast<const int*>(fc->fields[i].data)[0];
                gLogVerbose << "Building outDims: " << outDims << std::endl;
            }

            if (field_name.compare("type_id") == 0)
            {
                typeId = static_cast<const int*>(fc->fields[i].data)[0];
                gLogVerbose << "Building typeId: " << outDims << std::endl;
            }

            if (field_name.compare("W") == 0)
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
        if (typeId < 0 || typeId > 3)
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
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FCPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    try
    {
        return new FCPluginDynamic(name, serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FCPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* FCPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


#endif // #if CUDA_VERSION >= 10010
