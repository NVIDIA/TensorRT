/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cudnnWrapper.h"
#include "common/checkMacrosPlugin.h"

namespace nvinfer1
{
namespace pluginInternal
{

#define CUDNN_MAJOR 8
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
// Ensure that macros appearing in multiple files are only defined once.
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
auto const kCUDNN_PLUGIN_LIBNAME = std::string("cudnn64_") + std::to_string(CUDNN_MAJOR) + ".dll";
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
auto const kCUDNN_PLUGIN_LIBNAME = std::string("libcudnn.so.") + std::to_string(CUDNN_MAJOR);
#endif

// If tryLoadingCudnn failed, the CudnnWrapper object won't be created.
CudnnWrapper::CudnnWrapper(bool initHandle)
    : mLibrary(tryLoadingCudnn())
{
    auto load_sym = [](void* handle, char const* name) {
        void* ret = dllGetSym(handle, name);
        std::string loadError = "Fail to load symbol " + std::string(name) + " from the cudnn library.";
        PLUGIN_VALIDATE(ret != nullptr, loadError.c_str());
        return ret;
    };

    *(void**) (&_cudnnCreate) = load_sym(mLibrary, "cudnnCreate");
    *(void**) (&_cudnnDestroy) = load_sym(mLibrary, "cudnnDestroy");
    *(void**) (&_cudnnCreateTensorDescriptor) = load_sym(mLibrary, "cudnnCreateTensorDescriptor");
    *(void**) (&_cudnnDestroyTensorDescriptor) = load_sym(mLibrary, "cudnnDestroyTensorDescriptor");
    *(void**) (&_cudnnSetStream) = load_sym(mLibrary, "cudnnSetStream");
    *(void**) (&_cudnnBatchNormalizationForwardTraining) = load_sym(mLibrary, "cudnnBatchNormalizationForwardTraining");
    *(void**) (&_cudnnSetTensor4dDescriptor) = load_sym(mLibrary, "cudnnSetTensor4dDescriptor");
    *(void**) (&_cudnnSetTensorNdDescriptor) = load_sym(mLibrary, "cudnnSetTensorNdDescriptor");
    *(void**) (&_cudnnSetTensorNdDescriptorEx) = load_sym(mLibrary, "cudnnSetTensorNdDescriptorEx");
    *(void**) (&_cudnnDeriveBNTensorDescriptor) = load_sym(mLibrary, "cudnnDeriveBNTensorDescriptor");
    *(void**) (&_cudnnGetErrorString) = load_sym(mLibrary, "cudnnGetErrorString");

    if (initHandle)
    {
        PLUGIN_CUDNNASSERT(cudnnCreate(&mHandle));
        PLUGIN_VALIDATE(mHandle != nullptr);
    }
}

CudnnWrapper::~CudnnWrapper()
{
    if (mHandle != nullptr)
    {
        PLUGIN_CUDNNASSERT(cudnnDestroy(mHandle));
        mHandle = nullptr;
    }

    dllClose(mLibrary);
}

void* CudnnWrapper::tryLoadingCudnn()
{
    void* cudnnLib = dllOpen(kCUDNN_PLUGIN_LIBNAME.c_str());
    std::string errorMsg = "Failed to load " + kCUDNN_PLUGIN_LIBNAME + ".";
    PLUGIN_VALIDATE(cudnnLib != nullptr, errorMsg.c_str());
    return cudnnLib;
}

cudnnContext* CudnnWrapper::getCudnnHandle()
{
    return mHandle;
}

bool CudnnWrapper::isValid() const
{
    return mHandle != nullptr;
}

cudnnStatus_t CudnnWrapper::cudnnCreate(cudnnContext** handle)
{
    return (*_cudnnCreate)(handle);
}

cudnnStatus_t CudnnWrapper::cudnnDestroy(cudnnContext* handle)
{
    return (*_cudnnDestroy)(handle);
}

cudnnStatus_t CudnnWrapper::cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc)
{
    return (*_cudnnCreateTensorDescriptor)(tensorDesc);
}

cudnnStatus_t CudnnWrapper::cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
{
    return (*_cudnnDestroyTensorDescriptor)(tensorDesc);
}

cudnnStatus_t CudnnWrapper::cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId)
{
    return (*_cudnnSetStream)(handle, streamId);
}

cudnnStatus_t CudnnWrapper::cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    void const* alpha, void const* beta, cudnnTensorStruct const* xDesc, void const* x, cudnnTensorStruct const* yDesc,
    void* y, cudnnTensorStruct const* bnScaleBiasMeanVarDesc, void const* bnScale, void const* bnBias,
    double exponentialAverageFactor, void* resultRunningMean, void* resultRunningVariance, double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance)
{
    return (*_cudnnBatchNormalizationForwardTraining)(handle, mode, alpha, beta, xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance);
}

cudnnStatus_t CudnnWrapper::cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n, int c, int h, int w)
{
    return (*_cudnnSetTensor4dDescriptor)(tensorDesc, format, dataType, n, c, h, w);
}

cudnnStatus_t CudnnWrapper::cudnnSetTensorNdDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, int const dimA[], int const strideA[])
{
    return (*_cudnnSetTensorNdDescriptor)(tensorDesc, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t CudnnWrapper::cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims, int const dimA[])
{
    return (*_cudnnSetTensorNdDescriptorEx)(tensorDesc, format, dataType, nbDims, dimA);
}

cudnnStatus_t CudnnWrapper::cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t derivedBnDesc, cudnnTensorStruct const* xDesc, cudnnBatchNormMode_t mode)
{
    return (*_cudnnDeriveBNTensorDescriptor)(derivedBnDesc, xDesc, mode);
}

char const* CudnnWrapper::cudnnGetErrorString(cudnnStatus_t status)
{
    return (*_cudnnGetErrorString)(status);
}

CudnnWrapper& getCudnnWrapper()
{
    // Initialize a global cublasWrapper instance to be used to call cublas functions.
    static CudnnWrapper sGCudnnWrapper;
    return sGCudnnWrapper;
}

} // namespace pluginInternal
} // namespace nvinfer1
