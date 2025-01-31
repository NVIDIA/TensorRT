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

#ifndef TRT_PLUGIN_CUDNN_WRAPPER_H
#define TRT_PLUGIN_CUDNN_WRAPPER_H

#include "NvInferPlugin.h"
#include <functional>
#include <string>

extern "C"
{
    //! Forward declaration of cudnnTensorStruct to use in other interfaces.
    struct cudnnTensorStruct;
}

namespace nvinfer1
{
namespace pluginInternal
{
/*
 * Copy of the CUDNN return codes
 */
enum CudnnStatus
{
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14,
};

/*
 * Copy of the CUDNN cudnnBatchNormMode_t
 */
enum cudnnBatchNormMode
{
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL = 1,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
};

/*
 * Copy of the CUDNN cudnnTensorFormat_t
 */
enum cudnnTensorFormat
{
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
};

/*
 * Copy of CUDNN data type
 */
enum cudnnDataType
{
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
    CUDNN_DATA_BOOLEAN = 11,
    CUDNN_DATA_FP8_E4M3 = 12,
    CUDNN_DATA_FP8_E5M2 = 13,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 14,
};

using cudnnStatus_t = CudnnStatus;
using cudnnBatchNormMode_t = cudnnBatchNormMode;
using cudnnTensorFormat_t = cudnnTensorFormat;
using cudnnDataType_t = cudnnDataType;

using cudnnHandle_t = struct cudnnContext*;
using cudnnTensorDescriptor_t = struct cudnnTensorStruct*;

class CudnnWrapper
{
public:
    explicit CudnnWrapper(bool initHandle = false, char const* callerPluginName = nullptr);
    ~CudnnWrapper();

    cudnnContext* getCudnnHandle();
    bool isValid() const;

    /*
     * Copy of the CUDNN APIs
     */
    cudnnStatus_t cudnnCreate(cudnnContext** handle);
    cudnnStatus_t cudnnDestroy(cudnnContext* handle);
    cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
    cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
    cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
    cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle, cudnnBatchNormMode_t mode,
        void const* alpha, void const* beta, cudnnTensorStruct const* xDesc, void const* x,
        cudnnTensorStruct const* yDesc, void* y, cudnnTensorStruct const* bnScaleBiasMeanVarDesc, void const* bnScale,
        void const* bnBias, double exponentialAverageFactor, void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean, void* resultSaveInvVariance);
    cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
        cudnnDataType_t dataType, int n, int c, int h, int w);
    cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
        int const dimA[], int const strideA[]);
    cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
        cudnnDataType_t dataType, int nbDims, int const dimA[]);
    cudnnStatus_t cudnnDeriveBNTensorDescriptor(
        cudnnTensorDescriptor_t derivedBnDesc, cudnnTensorStruct const* xDesc, cudnnBatchNormMode_t mode);
    char const* cudnnGetErrorString(cudnnStatus_t status);

private:
    void* mLibrary{nullptr};
    cudnnContext* mHandle{nullptr};
    void* tryLoadingCudnn(char const*);

    cudnnStatus_t (*_cudnnCreate)(cudnnContext**);
    cudnnStatus_t (*_cudnnDestroy)(cudnnContext*);
    cudnnStatus_t (*_cudnnCreateTensorDescriptor)(cudnnTensorDescriptor_t*);
    cudnnStatus_t (*_cudnnDestroyTensorDescriptor)(cudnnTensorDescriptor_t);
    cudnnStatus_t (*_cudnnSetStream)(cudnnHandle_t, cudaStream_t);
    cudnnStatus_t (*_cudnnBatchNormalizationForwardTraining)(cudnnHandle_t, cudnnBatchNormMode_t, void const*,
        void const*, cudnnTensorStruct const*, void const*, cudnnTensorStruct const*, void*, cudnnTensorStruct const*,
        void const*, void const*, double, void*, void*, double, void*, void*);
    cudnnStatus_t (*_cudnnSetTensor4dDescriptor)(
        cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int);
    cudnnStatus_t (*_cudnnSetTensorNdDescriptor)(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
        int nbDims, int const dimA[], int const strideA[]);
    cudnnStatus_t (*_cudnnSetTensorNdDescriptorEx)(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
        cudnnDataType_t dataType, int nbDims, int const dimA[]);
    cudnnStatus_t (*_cudnnDeriveBNTensorDescriptor)(
        cudnnTensorDescriptor_t, cudnnTensorStruct const*, cudnnBatchNormMode_t);
    char const* (*_cudnnGetErrorString)(cudnnStatus_t status);
};

CudnnWrapper& getCudnnWrapper(char const* callerPluginName);

} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUDNN_WRAPPER_H
