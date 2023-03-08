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

#ifndef TRT_FC_PLUGIN_H
#define TRT_FC_PLUGIN_H

#include "NvInferPlugin.h"

#include "common/bertCommon.h"
#include <cublasLt.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

template <typename T>
struct GemmTypes
{
};

template <>
struct GemmTypes<half>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_16F;
    using dataTypeI = half;
    static const cudaDataType_t cudaTypeO = CUDA_R_16F;
    using dataTypeO = half;
    static const cudaDataType_t cudaTypeS = CUDA_R_16F;
    using dataTypeS = half;
#if CUBLAS_VER_MAJOR < 11
    static const cudaDataType_t cudaTypeCom = CUDA_R_16F;
#else
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_16F;
#endif
};

template <>
struct GemmTypes<float>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_32F;
    using dataTypeI = float;
    static const cudaDataType_t cudaTypeO = CUDA_R_32F;
    using dataTypeO = float;
    static const cudaDataType_t cudaTypeS = CUDA_R_32F;
    using dataTypeS = float;
#if CUBLAS_VER_MAJOR < 11
    static const cudaDataType_t cudaTypeCom = CUDA_R_32F;
#else
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_32F;
#endif
};

template <typename T>
struct Gemm
{
    using Types = GemmTypes<T>;
    typename Types::dataTypeI* A{nullptr};
    typename Types::dataTypeI* B{nullptr};
    typename Types::dataTypeO* C{nullptr};
    int m, n, k, ldA, ldB, ldC, rA, rB, rC, cA, cB, cC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;

    size_t elemA;
    size_t elemB;
    size_t elemC;
    bool transA, transB;

    cublasOperation_t opA;
    cublasOperation_t opB;

    const int word_size{sizeof(T)};
    typename Types::dataTypeS alpha;
    typename Types::dataTypeS beta;

    Gemm() {}

    Gemm(int m_, int n_, int k_, bool tA, bool tB)
    {
        init(m_, n_, k_, tA, tB);
    }

    void init(int m_, int n_, int k_, bool tA, bool tB) noexcept
    {
        m = m_;
        n = n_;
        k = k_;
        transA = tA;
        transB = tB;
        ldA = transA ? k : m;
        ldB = transB ? n : k;
        ldC = m;

        rA = ldA;
        rB = ldB;
        rC = ldC;

        cA = transA ? m : k;
        cB = transB ? k : n;
        cC = n;

        opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        elemA = m * k;
        elemB = n * k;
        elemC = n * m;
        bytesA = word_size * elemA;
        bytesB = word_size * elemB;
        bytesC = word_size * elemC;
        alpha = T(1.f);
        beta = T(0.f);
    }
};

auto constexpr algoCombinations = 6000;
auto constexpr algoIds = 40;
auto constexpr printAlgos = 1;
auto constexpr kernelRepeats = 10;
auto constexpr threadsPerBlock = 1024;

/* Structure to store information about different run trials */
typedef struct customMatMultPerfType_t
{
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time{1000000.F};
    size_t workspaceSize; // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

// clang-format off
void LtGemmSearch(cublasLtHandle_t ltHandle,
                  cublasOperation_t transa,
                  cublasOperation_t transb,
                  int const &m,
                  int const &n,
                  int const &k,
                  void const *alpha,
                  void const *A,
                  int const &lda,
                  void const *B,
                  int const &ldb,
                  void const *beta,
                  void *C,
                  int const &ldc,
                  void *workSpace,
                  size_t workSpaceSize,
                  cudaDataType_t computeType,
                  cudaDataType_t scaleType,
                  cudaDataType_t Atype,
                  cudaDataType_t Btype,
                  cudaDataType_t Ctype,
                  std::vector<customMatmulPerf_t> &perfResults);
// clang-format on
template <typename T>
void LtGemmSearch(cublasLtHandle_t ltHandle, const Gemm<T>& g, void* workSpace, size_t workSpaceSize,
    std::vector<customMatmulPerf_t>& perfResults)
{
    // clang-format off
  LtGemmSearch(ltHandle,
               g.opA,
               g.opB,
               g.m,
               g.n,
               g.k,
               &g.alpha,
               g.A,
               g.ldA,
               g.B,
               g.ldB,
               &g.beta,
               g.C,
               g.ldC,
               workSpace,
               workSpaceSize,
               Gemm<T>::Types::cudaTypeCom,
               Gemm<T>::Types::cudaTypeS,
               Gemm<T>::Types::cudaTypeI,
               Gemm<T>::Types::cudaTypeI,
               Gemm<T>::Types::cudaTypeO,
               perfResults);
    // clang-format on
}

struct LtContext
{
    cublasLtHandle_t cublas{nullptr};
    cudaDataType_t typeA;
    cudaDataType_t typeB;
    cudaDataType_t typeC;
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t typeComp;
#else
    cublasComputeType_t typeComp;
#endif
    cudaDataType_t typeS;
    cublasLtMatmulDesc_t operationDesc{nullptr};
    cublasLtMatrixLayout_t Adesc{nullptr};
    cublasLtMatrixLayout_t Bdesc{nullptr};
    cublasLtMatrixLayout_t Cdesc{nullptr};
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    void attach()
    {
        PLUGIN_CUBLASASSERT(cublasLtCreate(&cublas));
    }

    void detach()
    {
        PLUGIN_CUBLASASSERT(cublasLtDestroy(cublas));
    }

    void destroy()
    {
        if (operationDesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatmulDescDestroy(operationDesc));
            operationDesc = nullptr;
        }
        if (Adesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Adesc));
            Adesc = nullptr;
        }
        if (Bdesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Bdesc));
            Bdesc = nullptr;
        }
        if (Cdesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Cdesc));
            Cdesc = nullptr;
        }
    }

    template <typename T>
    void create(Gemm<T>& g, size_t workspaceSize)
    {
        typeA = Gemm<T>::Types::cudaTypeI;
        typeB = Gemm<T>::Types::cudaTypeI;
        typeC = Gemm<T>::Types::cudaTypeO;
        typeS = Gemm<T>::Types::cudaTypeS;
        typeComp = Gemm<T>::Types::cudaTypeCom; // compute

        // OPERATION
#if CUBLAS_VER_MAJOR < 11
        PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, typeComp));
#else
        PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, typeComp, typeS));
#endif
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &g.opA, sizeof(g.opA)));
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &g.opB, sizeof(g.opB)));

        // MAT DESC
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Adesc, typeA, g.rA, g.cA, g.ldA));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Bdesc, typeB, g.rB, g.cB, g.ldB));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Cdesc, typeC, g.rC, g.cC, g.ldC));
    }

    void setN(uint64_t n)
    {
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
    }
};

template <typename T>
cublasStatus_t inline cublasLtMatmul(
    LtContext& ctx, Gemm<T>& g, cublasLtMatmulAlgo_t algo, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // clang-format off
     return cublasLtMatmul(ctx.cublas,
                    ctx.operationDesc,
                    &g.alpha,
                    g.A,
                    ctx.Adesc,
                    g.B,
                    ctx.Bdesc,
                    &g.beta,
                    g.C,
                    ctx.Cdesc,
                    g.C,
                    ctx.Cdesc,
                    &algo,
                    workspace,
                    workspaceSize,
                    stream
                    );
    // clang-format on
}

/* CAUTION : must match cublasLtMatmulTile_t */
const char* const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8",
    "8x32",
    "16x16",
    "32x8",
    "8x64",
    "16x32",
    "32x16",
    "64x8",
    "32x32",
    "32x64",
    "64x32",
    "32x128",
    "64x64",
    "128x32",
    "64x128",
    "128x64",
    "64x256",
    "128x128",
    "256x64",
    "64x512",
    "128x256",
    "256x128",
    "512x64",
};

struct AlgoProps
{
    int32_t algoId;
    int32_t tile;
    int32_t swizzle;
    int32_t customOption;
    int32_t numSplitsK;
    int32_t reductionScheme;
    uint64_t numericImpl;

    void populate(const cublasLtMatmulAlgo_t& algo)
    {
        const cublasLtMatmulAlgo_t* matmulAlgo = &algo;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &numericImpl, sizeof(numericImpl), nullptr));
    }
};

template <typename T>
inline cublasLtMatmulAlgo_t gemmSearch(
    const int m, const int n, const int k, const size_t workspaceSize, size_t& actualWorkspace)
{

    Gemm<T> g(m, n, k, false, false);
    std::vector<customMatmulPerf_t> perfResults(algoCombinations);

    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.A), g.bytesA));
    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.B), g.bytesB));
    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.C), g.bytesC));

    void* workspace;
    PLUGIN_CUASSERT(cudaMalloc(&workspace, workspaceSize));
    cublasLtHandle_t lt;
    PLUGIN_CUBLASASSERT(cublasLtCreate(&lt));
    LtGemmSearch(lt, g, workspace, workspaceSize, perfResults);
    PLUGIN_CUASSERT(cudaDeviceSynchronize());
    PLUGIN_CUBLASASSERT(cublasLtDestroy(lt));
    PLUGIN_CUASSERT(cudaFree(workspace));

    PLUGIN_CUASSERT(cudaFree(g.A));
    PLUGIN_CUASSERT(cudaFree(g.B));
    PLUGIN_CUASSERT(cudaFree(g.C));

    actualWorkspace = perfResults[0].workspaceSize;
    return perfResults[0].algo;
}

template <typename T>
inline cublasLtMatmulAlgo_t gemmSearch(Gemm<T>& g, const size_t workspaceSize, size_t& actualWorkspace)
{

    std::vector<customMatmulPerf_t> perfResults(algoCombinations);

    PLUGIN_CUASSERT(cudaMalloc(&g.A, g.bytesA));
    PLUGIN_CUASSERT(cudaMalloc(&g.B, g.bytesB));
    PLUGIN_CUASSERT(cudaMalloc(&g.C, g.bytesC));

    void* workspace;
    PLUGIN_CUASSERT(cudaMalloc(&workspace, workspaceSize));
    cublasLtHandle_t lt;
    PLUGIN_CUBLASASSERT(cublasLtCreate(&lt));
    LtGemmSearch(lt, g, workspace, workspaceSize, perfResults);
    PLUGIN_CUASSERT(cudaDeviceSynchronize());
    PLUGIN_CUBLASASSERT(cublasLtDestroy(lt));
    PLUGIN_CUASSERT(cudaFree(workspace));

    PLUGIN_CUASSERT(cudaFree(g.A));
    PLUGIN_CUASSERT(cudaFree(g.B));
    PLUGIN_CUASSERT(cudaFree(g.C));

    actualWorkspace = perfResults[0].workspaceSize;
    return perfResults[0].algo;
}

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class FCPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    FCPluginDynamic(
        const std::string name, const nvinfer1::DataType type, const int outDim, const nvinfer1::Weights& W);

    FCPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make FCPluginDynamic without arguments, so we
    // delete default constructor.
    FCPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    size_t mOutDim; // leading dim
    size_t mNumParams;
    int mNmax;
    int mK;

    cublasLtMatmulAlgo_t mAlgo;

    bert::WeightsWithOwnership mW;
    bert::cuda_unique_ptr<void> mWdev;

    LtContext mLtContext;
};

class FCPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    FCPluginDynamicCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_FC_PLUGIN_H

#endif // #if CUDA_VERSION >= 10010
